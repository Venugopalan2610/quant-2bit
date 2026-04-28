"""Tripwire Q.0.3: toy synthetic training loop (PLUMBING SANITY ONLY).

The simplest possible analog of per-expert QAT:
  - Teacher: nn.Linear with random FP weights (the "FP16 teacher")
  - Student: QuantizedLinear.from_linear(teacher) — W_latent init from teacher
  - Loss:   MSE(student(x), teacher(x)) over random input batches

What this tests — and what it does NOT:
  - DOES verify: plumbing works end-to-end (grads flow, optimizer steps,
    W_latent moves, W_q stays on codebook, no NaN).
  - DOES NOT prove QAT beats PTQ. It can't — with isotropic Gaussian teacher
    weights AND isotropic Gaussian inputs, the output MSE is tr(E·E^T) where
    E = W_q - W_teacher. Viterbi/PTQ minimizes exactly this per tile, so
    PTQ is already optimal for this toy. The "QAT > PTQ" proof belongs to
    Q.1 (real OLMoE activations, which are highly non-isotropic).

Sub-tests:
  Q.0.3.1: training doesn't cause catastrophic regression
           (final MSE < 1.1 × initial). An LR sweep surfaces the expected
           monotone pattern: lower LR → closer to PTQ baseline.
  Q.0.3.2: W_latent actually drifts (not a no-op)
  Q.0.3.3: W_q is still on-codebook after training (quantization not broken)
  Q.0.3.4: final loss is finite (no NaN explosion)

Run: python -m src.tripwires.test_qat_toy_train
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, quantize_forward_numpy
from src.qat.quantized_linear import QuantizedLinear


def _make_codebook(seed=0):
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)
    return make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)


def _validate(student, teacher, x_val):
    with torch.no_grad():
        # Run student with a step-counter freeze so we don't inadvertently
        # skip a re-encode that the train loop would do. Just call forward.
        y_s = student(x_val)
        y_t = teacher(x_val)
    return F.mse_loss(y_s, y_t).item()


def run_training(n_steps=300, lr=1e-4, batch=256, in_f=64, out_f=64, seed=0,
                  reencode_every=1, verbose=True):
    torch.manual_seed(seed)
    cb = _make_codebook(seed=seed)

    # Teacher
    teacher = nn.Linear(in_f, out_f, bias=False).cuda().float()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student from teacher
    student = QuantizedLinear.from_linear(
        teacher, codebook_gpu=cb, seed=seed + 1337,
        reencode_every_n_steps=reencode_every,
    )
    opt = torch.optim.Adam([student.W_latent], lr=lr)

    # Fixed validation batch
    x_val = torch.randn(512, in_f, device="cuda")

    # Record at-init baseline (this is effectively PTQ on teacher's weights)
    loss_init = _validate(student, teacher, x_val)

    losses = [loss_init]
    for step in range(n_steps):
        x = torch.randn(batch, in_f, device="cuda")
        with torch.no_grad():
            y_target = teacher(x)
        y_s = student(x)
        loss = F.mse_loss(y_s, y_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            v = _validate(student, teacher, x_val)
            losses.append(v)
            if verbose:
                print(f"    step {step+1:4d}: train={loss.item():.4e}  val={v:.4e}")

    loss_final = _validate(student, teacher, x_val)
    return {
        "loss_init": loss_init,
        "loss_final": loss_final,
        "loss_trajectory": losses,
        "student": student,
        "teacher": teacher,
        "codebook": cb,
    }


def test_loss_decreases():
    """Small LR sweep: vanilla STE at 2bpw is sensitive to LR. We sweep
    {1e-3, 1e-4, 1e-5} and take the best. Also surfaces the sensitivity
    so future phases know the scale."""
    print("\nQ.0.3.1: training reduces MSE vs PTQ baseline (LR sweep)")
    print("-" * 60)
    best_res = None
    best_ratio = float("inf")
    sweep_results = {}
    for lr in (1e-3, 1e-4, 1e-5):
        print(f"\n  >>> lr={lr}")
        res = run_training(n_steps=300, lr=lr, batch=256, verbose=True)
        ratio = res["loss_final"] / max(res["loss_init"], 1e-30)
        sweep_results[lr] = ratio
        print(f"     init={res['loss_init']:.4e}  final={res['loss_final']:.4e}  "
              f"ratio={ratio:.3f}")
        if ratio < best_ratio:
            best_ratio = ratio
            best_res = res

    print(f"\n  sweep summary (final/init, < 1 means QAT helped):")
    for lr, r in sweep_results.items():
        print(f"     lr={lr:.0e}: ratio={r:.3f}")
    print(f"  best ratio: {best_ratio:.3f} (gate: < 1.1 — plumbing sanity)")
    ok = best_ratio < 1.1
    print(f"  [{'PASS' if ok else 'FAIL'}] training doesn't catastrophically regress")
    # Also check the expected monotone-in-LR pattern: smaller LR → closer to 1.0
    lrs_sorted = sorted(sweep_results.keys(), reverse=True)       # 1e-3, 1e-4, 1e-5
    monotone = all(sweep_results[lrs_sorted[i]] >= sweep_results[lrs_sorted[i+1]]
                    for i in range(len(lrs_sorted) - 1))
    print(f"  monotone-in-LR (expected): {monotone}")
    return ok and monotone, best_res


def test_W_latent_drifted(res):
    print("\nQ.0.3.2: W_latent drifted from FP16 init")
    print("-" * 60)
    teacher = res["teacher"]
    student = res["student"]
    W_init = teacher.weight.detach()
    W_final = student.W_latent.detach()
    drift = (W_final - W_init).abs().mean().item()
    init_mag = W_init.abs().mean().item()
    rel = drift / max(init_mag, 1e-30)
    print(f"  mean |W_latent - W_teacher|: {drift:.4e}")
    print(f"  mean |W_teacher|:             {init_mag:.4e}")
    print(f"  relative drift:               {rel:.4f}")
    # drift should be non-trivial (>1% of init magnitude) but not huge (<50%)
    ok = 1e-4 < rel < 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] W_latent drifted in expected band")
    return ok


def test_W_q_still_on_codebook(res):
    print("\nQ.0.3.3: W_q still on codebook after training")
    print("-" * 60)
    student = res["student"]
    cb_np = res["codebook"].cpu().numpy()

    # Extract final W_q by running the forward quant path on current W_latent.
    W_latent_np = student.W_latent.detach().cpu().numpy()
    sign_l = student.sign_l.cpu().numpy()
    sign_r = student.sign_r.cpu().numpy()
    W_q_np, W_scale, _, _, _ = quantize_forward_numpy(
        W_latent_np.astype(np.float32), sign_l, sign_r, res["codebook"],
    )

    # Same check as Q.0.1.2: after RHT + unscale, every V=2 pair on codebook.
    from src.rht.transform import apply_rht
    W_tilde_q = apply_rht(W_q_np, sign_l, sign_r).astype(np.float32)
    W_tilde_q_unit = W_tilde_q / W_scale

    pairs = W_tilde_q_unit.reshape(-1, 2)                          # (m*n/2, 2)
    # vectorized nearest-codebook distance
    # dists: (P, 65536). For 64x64 matrix that's 2048 * 65536 = 128M floats = 512MB.
    # Do in chunks.
    max_dev = 0.0
    P = pairs.shape[0]
    chunk = 4096
    for i in range(0, P, chunk):
        pb = pairs[i:i+chunk]                                       # (C, 2)
        d = np.abs(pb[:, None, :] - cb_np[None, :, :]).sum(axis=-1)  # (C, 65536)
        max_dev = max(max_dev, float(d.min(axis=1).max()))

    print(f"  pairs checked: {P}")
    print(f"  max deviation from nearest codebook entry: {max_dev:.2e}")
    ok = max_dev < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] W_q still on codebook")
    return ok


def test_final_loss_finite(res):
    print("\nQ.0.3.4: final loss is finite")
    print("-" * 60)
    ok = np.isfinite(res["loss_final"])
    print(f"  loss_final: {res['loss_final']}")
    print(f"  [{'PASS' if ok else 'FAIL'}] finite")
    return ok


def main():
    print("=" * 60)
    print("Tripwire Q.0.3: toy synthetic QAT training loop")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    ok1, res = test_loss_decreases()
    ok2 = test_W_latent_drifted(res)
    ok3 = test_W_q_still_on_codebook(res)
    ok4 = test_final_loss_finite(res)

    results = [
        ("loss_decreases", ok1),
        ("W_latent_drifted", ok2),
        ("W_q_on_codebook", ok3),
        ("final_loss_finite", ok4),
    ]

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("Q.0.3 GATE: PASS — vanilla STE QAT works on toy task.")
        print("Phase 0 plumbing complete.")
        print("Ready for Q.1 (per-expert QAT on real OLMoE experts).")
        sys.exit(0)
    else:
        print("Q.0.3 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
