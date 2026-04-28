"""qat_one_layer: train QATDecoderLayer for a single OLMoE decoder layer.

Factored out of src.tripwires.test_qat_decoder_train so the orchestrator and
tripwires share a codepath. Mirrors the Q.2.2 recipe:
  - collect_layer_io for the target layer (teacher-forced from FP16)
  - Build QATDecoderLayer wrapping a fresh FP32 GPU copy
  - Prime cache, train N_STEPS, reprime every REPRIME_EVERY, final prime
  - Return the trained QATDecoderLayer + stats; caller handles save/install.
"""
import copy
import gc
import time

import torch
import torch.nn.functional as F

from src.finetune.collect_activations import collect_layer_io
from src.qat.qat_decoder_layer import QATDecoderLayer, ATTN_PROJ_NAMES
from src.bcjr.anneal import (
    cosine_temperature_schedule,
    measure_ptq_residual,
    convert_layer_to_bcjr,
    convert_layer_to_ste,
    set_layer_temperature,
    entropy_diagnostic,
)


# Defaults match what passed Q.2.2 GATE (ratio=0.644).
DEFAULT_LR = 1e-4
DEFAULT_N_STEPS = 16
DEFAULT_BATCH_SEQS = 2
DEFAULT_REPRIME_EVERY = 4
DEFAULT_REENCODE_EVERY_INNER = 10_000
DEFAULT_MAX_SHARDS = 3
# Robustness knobs (added after L01 regression at reprime boundaries).
DEFAULT_GRAD_CLIP = 1.0        # global-norm clip on trainable params
DEFAULT_PROX_LAMBDA = 0.01     # weight of sum ||W_latent - W_q||^2 term

# BCJR defaults: prox_lambda=0 because BCJR's soft codeword already pulls
# W_latent toward favored codewords (proximal term becomes redundant and
# can over-regularize — see findings.md on the Hail Mary failure).
# T_min=1e-2 is the fp32 stability floor for BCJR logsumexp.
DEFAULT_BCJR_T_MIN = 1e-2
DEFAULT_BCJR_PROX_LAMBDA = 0.0


def _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs):
    """Run val in hard-STE cache mode.

    For BCJR runs, we temporarily flip all QLs to STE mode so forward() uses
    the cached W_q (which prime_cache just filled with hard STE output). This
    keeps val measurements on a consistent scale across STE and BCJR runs:
    both read hard quantized output, so val ratios are directly comparable.
    Flip-back happens in a finally block to survive OOM / exceptions.
    """
    saved_modes = []
    for _, ql in qat_layer._all_qls():
        saved_modes.append((ql, ql.quant_mode))
        if ql.quant_mode == "bcjr":
            ql.quant_mode = "ste"
    try:
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
    finally:
        for ql, mode in saved_modes:
            ql.quant_mode = mode


def qat_one_layer(model, rotary, cfg, codebook_gpu, layer_idx,
                  lr=DEFAULT_LR, n_steps=DEFAULT_N_STEPS,
                  batch_seqs=DEFAULT_BATCH_SEQS,
                  reprime_every=DEFAULT_REPRIME_EVERY,
                  reencode_every_inner=DEFAULT_REENCODE_EVERY_INNER,
                  max_shards=DEFAULT_MAX_SHARDS,
                  grad_clip=DEFAULT_GRAD_CLIP,
                  prox_lambda=DEFAULT_PROX_LAMBDA,
                  quant_mode="ste", bcjr_T_min=DEFAULT_BCJR_T_MIN,
                  bcjr_chunk=8, log_entropy_every=4,
                  seed=0, verbose=True):
    """Train one decoder layer's QAT wrapper. Returns (qat_layer, stats_dict).

    Caller owns the returned qat_layer — extract weights via
    qat_layer.layer.self_attn.{q,k,v,o}_proj._W_q_cache and
    qat_layer.layer.mlp.experts[e].{gate_up,down}._W_q_cache.
    """
    if verbose:
        print(f"  [L{layer_idx:02d}] collecting teacher I/O "
              f"(max_shards={max_shards})...", flush=True)
    t0 = time.time()
    shards_X, shards_Y, seq_len = collect_layer_io(
        model=model, rotary_emb=rotary, layer_idx=layer_idx,
        cfg=cfg, device="cuda", max_shards=max_shards,
    )
    all_X = torch.cat(shards_X, dim=0)
    all_Y = torch.cat(shards_Y, dim=0)
    n_seqs = all_X.shape[0]
    n_val = max(n_seqs // 4, 2)
    perm = torch.randperm(n_seqs, generator=torch.Generator().manual_seed(seed))
    idx_val, idx_train = perm[:n_val], perm[n_val:]
    train_X, train_Y = all_X[idx_train], all_Y[idx_train]
    val_X, val_Y = all_X[idx_val], all_Y[idx_val]
    del shards_X, shards_Y, all_X, all_Y
    if verbose:
        print(f"    train seqs={train_X.shape[0]} val seqs={val_X.shape[0]} "
              f"T={seq_len} ({time.time() - t0:.1f}s)", flush=True)

    if verbose:
        print(f"  [L{layer_idx:02d}] wrapping FP layer...", flush=True)
    fp_layer = copy.deepcopy(model.model.layers[layer_idx]).to(
        device="cuda", dtype=torch.float32,
    )
    qat_layer = QATDecoderLayer(
        fp_layer, codebook_gpu=codebook_gpu, config=cfg,
        seed=seed + layer_idx * 997,
        reencode_every_n_steps=reencode_every_inner,
    )

    # BCJR setup: measure T_0 from PTQ residual, flip all QLs to BCJR mode.
    # Must happen BEFORE prime_cache so the initial cache is computed via
    # STE (crisp baseline) rather than soft BCJR at T=T_0.
    bcjr_T_0 = None
    if quant_mode == "bcjr":
        if verbose:
            print(f"  [L{layer_idx:02d}] measuring PTQ residual for T_0...",
                  flush=True)
        bcjr_T_0 = measure_ptq_residual(qat_layer)
        if bcjr_T_0 < bcjr_T_min:
            if verbose:
                print(f"    measured T_0={bcjr_T_0:.3e} < T_min={bcjr_T_min:.3e}; "
                      f"using T_min as T_0", flush=True)
            bcjr_T_0 = bcjr_T_min * 2.0  # give the schedule room to decrease
        if verbose:
            print(f"    T_0={bcjr_T_0:.3e}  T_min={bcjr_T_min:.3e}", flush=True)

    if verbose:
        print(f"  [L{layer_idx:02d}] prime_cache (initial)...", flush=True)
    t_p = time.time()
    qat_layer.prime_cache()
    if verbose:
        print(f"    done in {time.time() - t_p:.1f}s  "
              f"GPU: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # val_init must be measured with STE hard cache for apples-to-apples
    # ratio reporting with STE baseline runs. Measure FIRST, then flip.
    val_init = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs)

    if quant_mode == "bcjr":
        if verbose:
            print(f"  [L{layer_idx:02d}] converting to BCJR mode at T_0={bcjr_T_0:.3e}...",
                  flush=True)
        convert_layer_to_bcjr(qat_layer, T_init=bcjr_T_0, bcjr_chunk=bcjr_chunk)

    params = qat_layer.trainable_parameters()
    opt = torch.optim.Adam(params, lr=lr)
    # No-regret tracking: keep the best W_q snapshot seen across all reprime
    # boundaries (including the initial prime_cache baseline).
    best_val = val_init
    best_snap = qat_layer.snapshot_W_q_cpu()
    best_tag = "init"
    trajectory = [("init", val_init)]
    if verbose:
        print(f"    val_init={val_init:.4e}  (best baseline)", flush=True)

    B_train, T, H = train_X.shape
    pos_ids = torch.arange(T, device="cuda").unsqueeze(0)
    bcjr_T_history = []
    for step in range(n_steps):
        # BCJR temperature update — cosine from T_0 down to T_min over n_steps
        if quant_mode == "bcjr":
            T_now = cosine_temperature_schedule(step, n_steps, bcjr_T_0, bcjr_T_min)
            set_layer_temperature(qat_layer, T_now)
            bcjr_T_history.append(T_now)

        idx = torch.randperm(B_train)[:batch_seqs]
        xb = train_X[idx].to("cuda", dtype=torch.float32, non_blocking=True)
        yb = train_Y[idx].to("cuda", dtype=torch.float32, non_blocking=True)
        pos = pos_ids.expand(xb.shape[0], -1)
        cos, sin = rotary(xb, pos)
        y = qat_layer(xb, position_embeddings=(cos, sin))
        if isinstance(y, tuple):
            y = y[0]
        mse = F.mse_loss(y, yb)
        prox = qat_layer.proximal_loss() if prox_lambda > 0 else 0.0
        loss = mse + prox_lambda * prox
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        opt.step()
        if verbose and (step + 1) % 4 == 0:
            prox_val = float(prox) if prox_lambda > 0 else 0.0
            extra = ""
            if quant_mode == "bcjr" and (step + 1) % log_entropy_every == 0:
                H = entropy_diagnostic(qat_layer, max_tiles=8)
                extra = f"  T={T_now:.3e}  H={H:.3f}"
            print(f"    step {step+1}/{n_steps}  "
                  f"mse={mse.item():.4e}  prox={prox_val:.4e}{extra}", flush=True)
        if (step + 1) % reprime_every == 0 and (step + 1) < n_steps:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            t_r = time.time()
            qat_layer.prime_cache()
            val_now = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs)
            tag = f"reprime@{step+1}"
            trajectory.append((tag, val_now))
            improved = val_now < best_val
            if improved:
                best_val = val_now
                best_snap = qat_layer.snapshot_W_q_cpu()
                best_tag = tag
            if verbose:
                marker = "  ← best" if improved else ""
                print(f"    reprime: {time.time() - t_r:.1f}s  "
                      f"val={val_now:.4e}{marker}", flush=True)

    opt.zero_grad(set_to_none=True)
    del opt
    gc.collect()
    torch.cuda.empty_cache()

    # Final hardening: flip BCJR→STE so prime_cache installs hard Viterbi W_q.
    if quant_mode == "bcjr":
        if verbose:
            print(f"  [L{layer_idx:02d}] final hardening: BCJR → STE...", flush=True)
        convert_layer_to_ste(qat_layer)

    if verbose:
        print(f"  [L{layer_idx:02d}] final prime_cache...", flush=True)
    t_p = time.time()
    qat_layer.prime_cache()
    val_final_raw = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs)
    trajectory.append(("final_prime", val_final_raw))
    if val_final_raw < best_val:
        best_val = val_final_raw
        best_snap = qat_layer.snapshot_W_q_cpu()
        best_tag = "final_prime"
    if verbose:
        print(f"    done in {time.time() - t_p:.1f}s  "
              f"val={val_final_raw:.4e}", flush=True)

    # Restore the best snapshot before returning — this is what we save.
    qat_layer.restore_W_q_from_cpu(best_snap)
    val_final = _val_mse(qat_layer, val_X, val_Y, rotary, batch_seqs)
    ratio = val_final / max(val_init, 1e-30)
    if verbose:
        print(f"  [L{layer_idx:02d}] val_init={val_init:.4e}  "
              f"val_final={val_final:.4e}  ratio={ratio:.3f}  "
              f"(best={best_tag})", flush=True)

    stats = {
        "layer_idx": layer_idx,
        "val_init": val_init,
        "val_final": val_final,
        "val_final_raw": val_final_raw,
        "ratio": ratio,
        "best_tag": best_tag,
        "trajectory": trajectory,
        "n_steps": n_steps,
        "lr": lr,
        "grad_clip": grad_clip,
        "prox_lambda": prox_lambda,
        "quant_mode": quant_mode,
        "bcjr_T_0": bcjr_T_0,
        "bcjr_T_min": bcjr_T_min if quant_mode == "bcjr" else None,
        "bcjr_T_history": bcjr_T_history if quant_mode == "bcjr" else None,
    }
    del best_snap
    # Free training data before returning
    del train_X, train_Y, val_X, val_Y
    gc.collect()
    torch.cuda.empty_cache()
    return qat_layer, stats
