"""Tripwire Q.2.2: Full-layer QAT (attn + experts) beats PTQ on one OLMoE layer.

Scope: quantize q/k/v/o projections AND all 64 experts in layer 0. Layernorms,
q_norm, k_norm, and router stay FP. Teacher match at decoder-layer boundary:
MSE(student_layer(X, pos_emb), Y_layer) where (X, Y) are captured from FP16.

Recipe:
  - collect_layer_io for layer 0, max_shards=3 → 3 shards of (8,1024,2048)
  - Split 2 train / 1 val
  - QATDecoderLayer wraps a fresh FP decoder layer
  - PTQ baseline = val MSE after first prime_cache (eval forward w/ cached W_q)
  - Train N steps on sub-batches of sequences; mid-train reprime every REPRIME_EVERY
  - val_final after final prime_cache
  - Gate: val_final < 0.9 × val_init

Sub-tests:
  Q.2.2.1: val_final < 0.9 × val_init
  Q.2.2.2: trajectory stable (<= 2 up-bumps)
  Q.2.2.3: >= 50/64 experts moved measurably (rel drift > 1e-5)
  Q.2.2.4: all 4 attn projs moved measurably (rel drift > 1e-5)
  Q.2.2.5: final val MSE finite

Run: python3 -m src.tripwires.test_qat_decoder_train
"""
import sys
import copy
import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.finetune.collect_activations import collect_layer_io
from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_decoder_layer import QATDecoderLayer, ATTN_PROJ_NAMES


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
LAYER_IDX = 0
MAX_SHARDS = 3            # 2 train + 1 val, total 24 sequences × 1024 tokens
SEED = 0
LR = 1e-4                 # 10× smaller than Q.2.1 — attention backward has
                          # bigger effective gradients than MoE-alone, and
                          # LR=1e-3 walked W_latent into a bad Viterbi region.
N_STEPS = 16
BATCH_SEQS = 2            # sequences per step (2 × 1024 = 2048 tokens)
REPRIME_EVERY = 4         # More frequent reprimes bound drift per window.
REENCODE_EVERY_INNER = 10_000  # effectively infinite; outer loop drives reprime


def _make_codebook(seed=0):
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)
    return make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)


def _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs=2):
    """val_X, val_Y: (B, T, H) CPU fp32. Runs under no_grad; forward uses cached W_q
    because step counter > 0 and REENCODE_EVERY_INNER is huge."""
    B, T, H = val_X.shape
    pos_ids = torch.arange(T, device="cuda").unsqueeze(0)
    se = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(0, B, batch_seqs):
            xb = val_X[i:i + batch_seqs].to("cuda", dtype=torch.float32, non_blocking=True)
            yb = val_Y[i:i + batch_seqs].to("cuda", dtype=torch.float32, non_blocking=True)
            pos = pos_ids.expand(xb.shape[0], -1)
            cos, sin = rotary(xb, pos)
            y = qat_layer(xb, position_embeddings=(cos, sin))
            if isinstance(y, tuple):
                y = y[0]
            se += F.mse_loss(y, yb, reduction="sum").item()
            cnt += yb.numel()
    return se / cnt


def _snapshot_latents(qat_layer):
    """Snapshot all QAT latents to CPU. Returns dict."""
    snap = {"attn": {}, "experts": []}
    for name in ATTN_PROJ_NAMES:
        snap["attn"][name] = (
            getattr(qat_layer.layer.self_attn, name).W_latent.detach().cpu().clone()
        )
    moe = qat_layer.layer.mlp
    for e in range(moe.num_experts):
        snap["experts"].append((
            moe.experts[e].gate_up.W_latent.detach().cpu().clone(),
            moe.experts[e].down.W_latent.detach().cpu().clone(),
        ))
    return snap


def _rel_drift(qat_layer, snap):
    """Return (attn_drifts dict, expert_drifts list)."""
    attn_d = {}
    for name in ATTN_PROJ_NAMES:
        now = getattr(qat_layer.layer.self_attn, name).W_latent.detach().cpu()
        ref = snap["attn"][name]
        drift = (now - ref).abs().mean().item()
        rel = drift / max(ref.abs().mean().item(), 1e-30)
        attn_d[name] = rel
    moe = qat_layer.layer.mlp
    exp_d = []
    for e in range(moe.num_experts):
        gu_now = moe.experts[e].gate_up.W_latent.detach().cpu()
        d_now = moe.experts[e].down.W_latent.detach().cpu()
        gu0, d0 = snap["experts"][e]
        gu_drift = (gu_now - gu0).abs().mean().item() / max(gu0.abs().mean().item(), 1e-30)
        d_drift = (d_now - d0).abs().mean().item() / max(d0.abs().mean().item(), 1e-30)
        exp_d.append((gu_drift, d_drift))
    return attn_d, exp_d


def train_one(qat_layer, train_X, train_Y, val_X, val_Y, rotary,
              lr, n_steps, batch_seqs, verbose=True):
    """train_X/Y: (B_train, T, H) CPU fp32. val_X/Y: same."""
    if verbose:
        print(f"      pre-prime GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    t_p = time.time()
    qat_layer.prime_cache()
    if verbose:
        print(f"      prime_cache: {time.time() - t_p:.1f}s  "
              f"GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    opt = torch.optim.Adam(qat_layer.trainable_parameters(), lr=lr)
    snap = _snapshot_latents(qat_layer)

    val_init = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs=batch_seqs)
    trajectory = [val_init]
    if verbose:
        print(f"      val_init={val_init:.4e}", flush=True)

    B_train, T, H = train_X.shape
    pos_ids = torch.arange(T, device="cuda").unsqueeze(0)
    for step in range(n_steps):
        torch.cuda.synchronize()
        t_step = time.time()
        idx = torch.randperm(B_train)[:batch_seqs]
        xb = train_X[idx].to("cuda", dtype=torch.float32, non_blocking=True)
        yb = train_Y[idx].to("cuda", dtype=torch.float32, non_blocking=True)
        pos = pos_ids.expand(xb.shape[0], -1)
        cos, sin = rotary(xb, pos)

        t_fwd0 = time.time()
        y = qat_layer(xb, position_embeddings=(cos, sin))
        if isinstance(y, tuple):
            y = y[0]
        torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd0
        loss = F.mse_loss(y, yb)
        opt.zero_grad(set_to_none=True)
        t_bwd0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t_bwd0
        t_opt0 = time.time()
        opt.step()
        torch.cuda.synchronize()
        t_opt = time.time() - t_opt0
        total = time.time() - t_step
        if verbose:
            print(f"      step {step+1:3d}  loss={loss.item():.4e}  "
                  f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s opt={t_opt:.2f}s total={total:.2f}s",
                  flush=True)
        if (step + 1) % 5 == 0:
            v = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs=batch_seqs)
            trajectory.append(v)
            if verbose:
                print(f"        val={v:.4e}", flush=True)

        if (step + 1) % REPRIME_EVERY == 0 and (step + 1) < n_steps:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            t_r = time.time()
            qat_layer.prime_cache()
            if verbose:
                print(f"        reprime: {time.time() - t_r:.1f}s  "
                      f"GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                      flush=True)

    opt.zero_grad(set_to_none=True)
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print(f"      pre-final-prime GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    t_p = time.time()
    qat_layer.prime_cache()
    if verbose:
        print(f"      final prime_cache: {time.time() - t_p:.1f}s  "
              f"GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    val_final = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs=batch_seqs)
    attn_drifts, exp_drifts = _rel_drift(qat_layer, snap)

    return {
        "val_init": val_init,
        "val_final": val_final,
        "trajectory": trajectory,
        "attn_drifts": attn_drifts,
        "exp_drifts": exp_drifts,
    }


def main():
    print("=" * 60)
    print("Tripwire Q.2.2: full-layer QAT (attn + experts) on one OLMoE layer")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA required")
        sys.exit(1)

    print("Loading FP16 OLMoE (bf16 CPU)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    cfg = model.config
    rotary = copy.deepcopy(model.model.rotary_emb).to("cuda", dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"\nCollecting layer {LAYER_IDX} I/O (max_shards={MAX_SHARDS})...")
    shards_X, shards_Y, seq_len = collect_layer_io(
        model=model, rotary_emb=rotary, layer_idx=LAYER_IDX,
        cfg=cfg, device="cuda", max_shards=MAX_SHARDS,
    )
    # Concat all shards along batch dim: (B_total, T, H) on CPU.
    all_X = torch.cat(shards_X, dim=0)
    all_Y = torch.cat(shards_Y, dim=0)
    n_seqs = all_X.shape[0]
    n_val = max(n_seqs // 4, 2)
    perm = torch.randperm(n_seqs)
    idx_val, idx_train = perm[:n_val], perm[n_val:]
    train_X, train_Y = all_X[idx_train], all_Y[idx_train]
    val_X, val_Y = all_X[idx_val], all_Y[idx_val]
    print(f"  train seqs={train_X.shape[0]}  val seqs={val_X.shape[0]}  T={seq_len}",
          flush=True)
    del shards_X, shards_Y, all_X, all_Y

    # Build fresh FP layer for QAT wrap.
    print("\nBuilding QATDecoderLayer (fresh FP layer copy)...")
    fp_layer = copy.deepcopy(model.model.layers[LAYER_IDX]).to(
        device="cuda", dtype=torch.float32,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    cb = _make_codebook(seed=SEED)
    qat_layer = QATDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=SEED, reencode_every_n_steps=REENCODE_EVERY_INNER,
    )
    n_params = sum(p.numel() for p in qat_layer.trainable_parameters())
    print(f"  trainable params: {n_params/1e6:.1f}M", flush=True)

    print(f"\nTraining (LR={LR}, N_STEPS={N_STEPS}, BATCH_SEQS={BATCH_SEQS})...")
    res = train_one(
        qat_layer, train_X, train_Y, val_X, val_Y, rotary,
        lr=LR, n_steps=N_STEPS, batch_seqs=BATCH_SEQS, verbose=True,
    )
    ratio = res["val_final"] / max(res["val_init"], 1e-30)
    print(f"\n  val_init={res['val_init']:.4e}  "
          f"val_final={res['val_final']:.4e}  ratio={ratio:.3f}  (gate: < 0.9)")

    # === Sub-test gates ===
    ok1 = ratio < 0.9
    print(f"\nQ.2.2.1: QAT beats PTQ by >= 10%   [{'PASS' if ok1 else 'FAIL'}]")

    traj = res["trajectory"]
    bumps = sum(1 for i in range(1, len(traj)) if traj[i] > traj[i-1])
    ok2 = bumps <= 2
    print(f"Q.2.2.2: trajectory stable (<=2 up-bumps, saw {bumps})  "
          f"[{'PASS' if ok2 else 'FAIL'}]")

    moved_exp = sum(1 for gu, d in res["exp_drifts"] if gu > 1e-5 and d > 1e-5)
    ok3 = moved_exp >= 50
    print(f"Q.2.2.3: >=50/64 experts moved measurably (saw {moved_exp})  "
          f"[{'PASS' if ok3 else 'FAIL'}]")

    moved_attn = sum(1 for n in ATTN_PROJ_NAMES if res["attn_drifts"][n] > 1e-5)
    ok4 = moved_attn == 4
    drifts_str = "  ".join(f"{n}={res['attn_drifts'][n]:.1e}" for n in ATTN_PROJ_NAMES)
    print(f"Q.2.2.4: all 4 attn projs moved (saw {moved_attn}/4)  [{'PASS' if ok4 else 'FAIL'}]")
    print(f"          drifts: {drifts_str}")

    ok5 = np.isfinite(res["val_final"])
    print(f"Q.2.2.5: final val MSE finite ({res['val_final']:.4e})  "
          f"[{'PASS' if ok5 else 'FAIL'}]")

    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    print("\n" + "=" * 60)
    if all_ok:
        print("Q.2.2 GATE: PASS — full-layer QAT beats PTQ on one OLMoE layer.")
        sys.exit(0)
    else:
        print("Q.2.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
