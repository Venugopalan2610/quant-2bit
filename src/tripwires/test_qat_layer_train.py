"""Tripwire Q.2.1: Experts-only QAT beats PTQ on one OLMoE layer.

Scope: quantize all 64 experts in layer 0 jointly. Router and attention stay
FP16. Teacher match at MoE-block boundary: MSE(student_moe(X_moe), Y_moe)
where (X_moe, Y_moe) are captured from the FP16 model.

Recipe:
  - Collect (X_moe, Y_moe) for layer 0 (max_tokens=8192)
  - Split train / val
  - QATMoELayer.from_mlp_block(fp16 mlp block)
  - PTQ baseline = MSE on val at step 0
  - LR sweep {1e-5, 5e-5}, N_STEPS each
  - Gate: best val MSE < 0.9 × PTQ baseline

Sub-tests:
  Q.2.1.1: best LR achieves val_final < 0.9 × val_init
  Q.2.1.2: trajectory stable (<= 2 up-bumps on best LR)
  Q.2.1.3: at least 50 experts moved measurably (rel drift > 1e-5)
  Q.2.1.4: final val MSE finite

Run: python3 -m src.tripwires.test_qat_layer_train
"""
import sys
import copy
import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.finetune.collect_activations import collect_moe_io
from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_moe_layer import QATMoELayer


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
LAYER_IDX = 0
MAX_TOKENS = 8192
SEED = 0
LR_SWEEP = (1e-3,)
N_STEPS = 16
BATCH = 1024
# Explicit re-prime cadence. In-training reencode is driven by our outer loop,
# not by the QuantizedLinear step counter (REENCODE_EVERY_INNER is effectively
# infinite so forward()s always use cached W_q under the hood).
REPRIME_EVERY = 8
REENCODE_EVERY_INNER = 10_000


def _make_codebook(seed=0):
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)
    return make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)


def _val_mse(qmoe, X_val, Y_val, batch=1024):
    with torch.no_grad():
        n = X_val.shape[0]
        se = 0.0
        cnt = 0
        for i in range(0, n, batch):
            xb = X_val[i:i + batch]
            yb = Y_val[i:i + batch]
            y = qmoe.eval_forward(xb)
            se += F.mse_loss(y, yb, reduction="sum").item()
            cnt += yb.numel()
        return se / cnt


def _snapshot_latents(qmoe):
    """Return (gu_clone, d_clone) per expert, moved to CPU to save VRAM."""
    snaps = []
    for e in range(qmoe.num_experts):
        snaps.append((
            qmoe.experts[e].gate_up.W_latent.detach().cpu().clone(),
            qmoe.experts[e].down.W_latent.detach().cpu().clone(),
        ))
    return snaps


def _rel_drift(qmoe, snaps):
    """Return per-expert (gu_rel, d_rel) list. snaps are CPU tensors."""
    drifts = []
    for e in range(qmoe.num_experts):
        gu_now = qmoe.experts[e].gate_up.W_latent.detach().cpu()
        d_now  = qmoe.experts[e].down.W_latent.detach().cpu()
        gu0, d0 = snaps[e]
        gu_drift = (gu_now - gu0).abs().mean().item()
        d_drift  = (d_now  - d0).abs().mean().item()
        gu_ref = gu0.abs().mean().item()
        d_ref  = d0.abs().mean().item()
        drifts.append((
            gu_drift / max(gu_ref, 1e-30),
            d_drift  / max(d_ref,  1e-30),
        ))
    return drifts


def train_one(config, mlp_block, X_train, Y_train, X_val, Y_val,
              codebook, lr, n_steps, batch, seed, verbose=True):
    qmoe = QATMoELayer.from_mlp_block(
        mlp_block=mlp_block, config=config,
        codebook_gpu=codebook, seed=seed,
        reencode_every_n_steps=REENCODE_EVERY_INNER,
    )
    if verbose:
        print(f"      pre-prime GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    t_prime = time.time()
    qmoe.prime_cache()
    if verbose:
        print(f"      prime_cache: {time.time() - t_prime:.1f}s  "
              f"post-prime GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    opt = torch.optim.Adam(qmoe.trainable_parameters(), lr=lr)
    snaps = _snapshot_latents(qmoe)

    val_init = _val_mse(qmoe, X_val, Y_val)
    trajectory = [val_init]

    n_train = X_train.shape[0]
    for step in range(n_steps):
        torch.cuda.synchronize()
        t_step = time.time()
        idx = torch.randint(0, n_train, (batch,), device=X_train.device)
        xb = X_train[idx]
        yb = Y_train[idx]
        t_fwd0 = time.time()
        y = qmoe(xb)
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
            v = _val_mse(qmoe, X_val, Y_val)
            trajectory.append(v)
            if verbose:
                print(f"        val={v:.4e}", flush=True)
        # Mid-training reencode: run OUTSIDE autograd graph so Viterbi's
        # transient allocation doesn't spike alongside Adam + grad tensors.
        if (step + 1) % REPRIME_EVERY == 0 and (step + 1) < n_steps:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            t_r = time.time()
            qmoe.prime_cache()
            if verbose:
                print(f"        reprime: {time.time() - t_r:.1f}s  "
                      f"GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
                      flush=True)

    # Free Adam + grads so second prime has room for Viterbi's transient peak.
    opt.zero_grad(set_to_none=True)
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print(f"      pre-final-prime GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    t_prime = time.time()
    qmoe.prime_cache()
    if verbose:
        print(f"      final prime_cache: {time.time() - t_prime:.1f}s  "
              f"post GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    val_final = _val_mse(qmoe, X_val, Y_val)
    drifts = _rel_drift(qmoe, snaps)

    # Free this sweep slot's resources
    del snaps, qmoe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "val_init": val_init,
        "val_final": val_final,
        "trajectory": trajectory,
        "drifts": drifts,
    }


def main():
    print("=" * 60)
    print("Tripwire Q.2.1: experts-only QAT on one OLMoE layer")
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

    print(f"\nCollecting layer {LAYER_IDX} MoE I/O (max_tokens={MAX_TOKENS})...")
    X_cpu, Y_cpu = collect_moe_io(
        model=model, rotary_emb=rotary, layer_idx=LAYER_IDX,
        cfg=cfg, device="cuda", max_tokens=MAX_TOKENS,
    )
    n = X_cpu.shape[0]
    n_val = max(n // 4, 256)
    perm = torch.randperm(n)
    idx_val, idx_train = perm[:n_val], perm[n_val:]
    X_train = X_cpu[idx_train].to("cuda", dtype=torch.float32)
    Y_train = Y_cpu[idx_train].to("cuda", dtype=torch.float32)
    X_val   = X_cpu[idx_val].to("cuda", dtype=torch.float32)
    Y_val   = Y_cpu[idx_val].to("cuda", dtype=torch.float32)
    print(f"  train n={X_train.shape[0]}  val n={X_val.shape[0]}", flush=True)

    # Snapshot the FP16 mlp block on CPU — we only need it to init QATMoELayer.
    mlp_block_cpu = copy.deepcopy(model.model.layers[LAYER_IDX].mlp).to(
        dtype=torch.float32
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    cb = _make_codebook(seed=SEED)

    print("\nLR sweep:")
    sweep = {}
    best = None
    for lr in LR_SWEEP:
        print(f"\n  >>> lr={lr}", flush=True)
        res = train_one(
            cfg, mlp_block_cpu, X_train, Y_train, X_val, Y_val,
            codebook=cb, lr=lr, n_steps=N_STEPS, batch=BATCH,
            seed=SEED, verbose=True,
        )
        ratio = res["val_final"] / max(res["val_init"], 1e-30)
        sweep[lr] = res
        print(f"     val_init={res['val_init']:.4e}  "
              f"val_final={res['val_final']:.4e}  ratio={ratio:.3f}", flush=True)
        if best is None or res["val_final"] < best["val_final"]:
            best = res
            best["lr"] = lr

    print(f"\n  sweep summary (val_final / val_init, <1 means QAT helped):")
    for lr, r in sweep.items():
        ratio = r["val_final"] / max(r["val_init"], 1e-30)
        print(f"     lr={lr:.1e}: init={r['val_init']:.4e}  "
              f"final={r['val_final']:.4e}  ratio={ratio:.3f}")
    best_ratio = best["val_final"] / max(best["val_init"], 1e-30)
    print(f"\n  best lr={best['lr']}  ratio={best_ratio:.3f}  (gate: < 0.9)")

    # === Sub-test gates ===
    ok1 = best_ratio < 0.9
    print(f"\nQ.2.1.1: QAT beats PTQ by >= 10%   [{'PASS' if ok1 else 'FAIL'}]")

    traj = best["trajectory"]
    bumps = sum(1 for i in range(1, len(traj)) if traj[i] > traj[i-1])
    ok2 = bumps <= 2
    print(f"Q.2.1.2: trajectory stable (<=2 up-bumps, saw {bumps})  "
          f"[{'PASS' if ok2 else 'FAIL'}]")

    moved = sum(1 for gu, d in best["drifts"] if gu > 1e-5 and d > 1e-5)
    ok3 = moved >= 50
    print(f"Q.2.1.3: >=50/64 experts moved measurably (saw {moved})  "
          f"[{'PASS' if ok3 else 'FAIL'}]")

    ok4 = np.isfinite(best["val_final"])
    print(f"Q.2.1.4: final val MSE finite ({best['val_final']:.4e})  "
          f"[{'PASS' if ok4 else 'FAIL'}]")

    all_ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 60)
    if all_ok:
        print("Q.2.1 GATE: PASS — experts-only QAT beats PTQ on one layer.")
        sys.exit(0)
    else:
        print("Q.2.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
