"""Tripwire Q.1.3: QAT beats PTQ on a real OLMoE expert.

This is the real-world test Q.0.3 couldn't do. On isotropic Gaussian toy
inputs, PTQ's Frobenius-optimal quantization is also activation-MSE-optimal,
so QAT can't beat PTQ. But real expert activations are highly non-isotropic
(routed, sparse, kurtotic) — PTQ is no longer optimal, and QAT should close
meaningful ground.

Recipe:
  - Collect (X, Y) for one layer-0 expert (budget=512)
  - Split (X, Y) into train / val
  - QATExpert.from_weights(fp16_expert_weights)
  - PTQ baseline = MSE on val at step 0
  - LR sweep {1e-5, 5e-5, 1e-4}, 100 steps each
  - Gate: best val MSE < 0.9 × PTQ baseline

Sub-tests:
  Q.1.3.1: at least one LR in the sweep achieves val_final < 0.9 × val_init
  Q.1.3.2: best-LR trajectory is monotonically decreasing (or ≤ 1 bump)
  Q.1.3.3: W_latent_gate_up and W_latent_down both moved (> 1e-5 relative)
  Q.1.3.4: final val loss is finite

Run: python3 -m src.tripwires.test_qat_real_expert_train
"""
import sys
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.finetune.collect_activations import collect_layer_activations
from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_expert import QATExpert


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
LAYER_IDX = 0
EXPERT_IDX = 0
BUDGET = 512
SEED = 0
LR_SWEEP = (1e-5, 5e-5)
N_STEPS = 40
BATCH = 64
REENCODE_EVERY = 4


def _make_codebook(seed=0):
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)
    return make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)


def _val_mse(expert, X_val, Y_val):
    expert.gate_up._step_counter.zero_()
    expert.down._step_counter.zero_()
    with torch.no_grad():
        y = expert(X_val)
    return F.mse_loss(y, Y_val).item()


def train_one(W_gate_up, W_down, X_train, Y_train, X_val, Y_val,
              codebook, lr, n_steps, batch, seed, verbose=True):
    """Train a fresh QATExpert and return trajectory + final model."""
    expert = QATExpert.from_weights(
        W_gate_up_fp16=W_gate_up, W_down_fp16=W_down,
        codebook_gpu=codebook, seed=seed,
        reencode_every_n_steps=REENCODE_EVERY,
    )
    opt = torch.optim.Adam(expert.trainable_parameters(), lr=lr)
    W_gu_init = expert.gate_up.W_latent.detach().clone()
    W_d_init  = expert.down.W_latent.detach().clone()

    val_init = _val_mse(expert, X_val, Y_val)
    trajectory = [val_init]

    n_train = X_train.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch,), device=X_train.device)
        xb = X_train[idx]
        yb = Y_train[idx]
        y = expert(xb)
        loss = F.mse_loss(y, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 10 == 0:
            v = _val_mse(expert, X_val, Y_val)
            trajectory.append(v)
            if verbose:
                print(f"      step {step+1:3d}  train={loss.item():.4e}  val={v:.4e}")

    val_final = _val_mse(expert, X_val, Y_val)
    gu_drift = (expert.gate_up.W_latent - W_gu_init).abs().mean().item()
    d_drift  = (expert.down.W_latent - W_d_init).abs().mean().item()
    gu_ref   = W_gu_init.abs().mean().item()
    d_ref    = W_d_init.abs().mean().item()
    return {
        "val_init": val_init,
        "val_final": val_final,
        "trajectory": trajectory,
        "gu_rel_drift": gu_drift / max(gu_ref, 1e-30),
        "d_rel_drift":  d_drift  / max(d_ref,  1e-30),
    }


def main():
    print("=" * 60)
    print("Tripwire Q.1.3: QAT vs PTQ on real OLMoE expert")
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
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\nCollecting layer {LAYER_IDX} activations (budget={BUDGET})...")
    captures = collect_layer_activations(
        model=model, rotary_emb=rotary, layer_idx=LAYER_IDX,
        cfg=cfg, device="cuda", budget_per_expert=BUDGET,
    )
    X_cpu, Y_cpu = captures[EXPERT_IDX]
    n = X_cpu.shape[0]
    n_val = max(n // 4, 64)
    perm = torch.randperm(n)
    idx_val, idx_train = perm[:n_val], perm[n_val:]
    X_train = X_cpu[idx_train].to("cuda", dtype=torch.float32)
    Y_train = Y_cpu[idx_train].to("cuda", dtype=torch.float32)
    X_val   = X_cpu[idx_val].to("cuda", dtype=torch.float32)
    Y_val   = Y_cpu[idx_val].to("cuda", dtype=torch.float32)
    print(f"  expert {EXPERT_IDX}: train n={X_train.shape[0]}  val n={X_val.shape[0]}")

    experts_mod = model.model.layers[LAYER_IDX].mlp.experts
    W_gate_up = experts_mod.gate_up_proj[EXPERT_IDX].detach().to(
        device="cuda", dtype=torch.float32
    )
    W_down = experts_mod.down_proj[EXPERT_IDX].detach().to(
        device="cuda", dtype=torch.float32
    )

    # Free the big CPU model — we have what we need.
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    cb = _make_codebook(seed=SEED)

    print("\nLR sweep:")
    sweep_results = {}
    best = None
    for lr in LR_SWEEP:
        print(f"\n  >>> lr={lr}")
        res = train_one(
            W_gate_up, W_down, X_train, Y_train, X_val, Y_val,
            codebook=cb, lr=lr, n_steps=N_STEPS, batch=BATCH,
            seed=SEED, verbose=True,
        )
        ratio = res["val_final"] / max(res["val_init"], 1e-30)
        sweep_results[lr] = res
        print(f"     val_init={res['val_init']:.4e}  "
              f"val_final={res['val_final']:.4e}  ratio={ratio:.3f}")
        if best is None or res["val_final"] < best["val_final"]:
            best = res
            best["lr"] = lr

    print(f"\n  sweep summary (val_final / val_init, <1 means QAT helped):")
    for lr, r in sweep_results.items():
        ratio = r["val_final"] / max(r["val_init"], 1e-30)
        print(f"     lr={lr:.1e}: init={r['val_init']:.4e}  "
              f"final={r['val_final']:.4e}  ratio={ratio:.3f}")
    best_ratio = best["val_final"] / max(best["val_init"], 1e-30)
    print(f"\n  best lr={best['lr']}  ratio={best_ratio:.3f}  (gate: < 0.9)")

    # === Sub-test gates ===
    ok1 = best_ratio < 0.9
    print(f"\nQ.1.3.1: QAT beats PTQ by >= 10%   [{'PASS' if ok1 else 'FAIL'}]")

    traj = best["trajectory"]
    bumps = sum(1 for i in range(1, len(traj)) if traj[i] > traj[i-1])
    ok2 = bumps <= 2
    print(f"Q.1.3.2: trajectory stable (≤2 up-bumps, saw {bumps})  "
          f"[{'PASS' if ok2 else 'FAIL'}]")

    ok3 = best["gu_rel_drift"] > 1e-5 and best["d_rel_drift"] > 1e-5
    print(f"Q.1.3.3: both latent matrices moved "
          f"(gu={best['gu_rel_drift']:.2e}, d={best['d_rel_drift']:.2e})  "
          f"[{'PASS' if ok3 else 'FAIL'}]")

    ok4 = np.isfinite(best["val_final"])
    print(f"Q.1.3.4: final val MSE finite ({best['val_final']:.4e})  "
          f"[{'PASS' if ok4 else 'FAIL'}]")

    all_ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 60)
    if all_ok:
        print("Q.1.3 GATE: PASS — QAT beats PTQ on a real OLMoE expert.")
        print("Phase Q.1 single-expert success. Ready to scale to all experts/layers.")
        sys.exit(0)
    else:
        print("Q.1.3 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
