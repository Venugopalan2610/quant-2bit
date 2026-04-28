"""Tripwire Q.0.1: STE forward/backward sanity.

Sub-tests:
  Q.0.1.1: forward produces finite, correctly-shaped W_q
  Q.0.1.2: W_q is on-codebook — i.e. inverse_rht(W_q)/W_scale has every
           (Tx,Ty) tile that, when read out as (V=2) pairs, matches a
           row of the 65536-entry codebook exactly (up to fp32 noise).
  Q.0.1.3: STE backward is identity — for any upstream grad g on W_q,
           W_latent.grad should equal g exactly.
  Q.0.1.4: end-to-end autograd — a scalar loss on W_q produces non-zero
           finite gradients on W_latent, and those gradients differ from
           noise (i.e. directional signal is real).

Run: python -m src.tripwires.test_qat_ste
"""
import sys
import numpy as np
import torch

from src.codes.lut_init import init_hyb_lut
from src.rht.transform import make_sign_vector
from src.qat.ste import (
    TrellisQuantSTE, quantize_forward_numpy, make_hyb_codebook_gpu, L_BITS,
)


def _setup(m=64, n=64, seed=0):
    """Build a random W_latent, sign vectors, and a codebook."""
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=seed)        # (512, 2)
    codebook_gpu = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)
    sign_l = make_sign_vector(m, seed=seed * 10 + 1)
    sign_r = make_sign_vector(n, seed=seed * 10 + 2)
    rng = np.random.default_rng(seed)
    W_np = rng.standard_normal(size=(m, n)).astype(np.float32)
    W_latent = torch.from_numpy(W_np).cuda().requires_grad_(True)
    return W_latent, sign_l, sign_r, codebook_gpu, lut


def test_forward_shape_finite():
    print("\nQ.0.1.1: forward shape + finiteness")
    print("-" * 60)
    W_latent, sign_l, sign_r, cb, _ = _setup(m=64, n=64)

    W_q = TrellisQuantSTE.apply(W_latent, sign_l, sign_r, cb)

    shape_ok = W_q.shape == W_latent.shape
    finite_ok = bool(torch.isfinite(W_q).all().item())
    not_constant = float(W_q.std().item()) > 1e-6

    print(f"  W_q shape: {tuple(W_q.shape)} (expected {tuple(W_latent.shape)})")
    print(f"  finite: {finite_ok}")
    print(f"  std(W_q): {W_q.std().item():.4f}")
    print(f"  mean abs diff |W_q - W_latent|: {(W_q - W_latent).abs().mean().item():.4f}")

    ok = shape_ok and finite_ok and not_constant
    print(f"  [{'PASS' if shape_ok else 'FAIL'}] shape matches")
    print(f"  [{'PASS' if finite_ok else 'FAIL'}] W_q is finite")
    print(f"  [{'PASS' if not_constant else 'FAIL'}] W_q is non-constant")
    return ok


def test_on_codebook():
    """After RHT + scale, every (V=2) pair of W_tilde_q_unit must match some
    codebook entry to fp32 precision."""
    print("\nQ.0.1.2: W_q is on-codebook (after RHT round-trip)")
    print("-" * 60)
    W_latent, sign_l, sign_r, cb, lut = _setup(m=64, n=64)

    W_np = W_latent.detach().cpu().numpy()
    W_q_np, W_scale, walks, _bs, _ss = quantize_forward_numpy(
        W_np, sign_l, sign_r, cb,
    )

    # Reverse RHT + scale on W_q to get back to codebook-space.
    from src.rht.transform import apply_rht
    W_tilde_q = apply_rht(W_q_np, sign_l, sign_r).astype(np.float32)
    W_tilde_q_unit = W_tilde_q / W_scale

    # Every consecutive (V=2) pair in row-major order of each (Tx=16,Ty=16)
    # tile should match a codebook entry. Flatten tile-wise and compare.
    cb_np = cb.cpu().numpy()                                       # (65536, 2)
    Tx, Ty, V = 16, 16, 2
    m, n = W_tilde_q_unit.shape
    n_row_blocks = m // Tx
    n_col_blocks = n // Ty

    max_dev = 0.0
    total_pairs = 0
    mismatches = 0
    for j in range(n_col_blocks):
        col_block = W_tilde_q_unit[:, j*Ty:(j+1)*Ty]              # (m, Ty)
        for i in range(n_row_blocks):
            tile = col_block[i*Tx:(i+1)*Tx, :].reshape(-1)          # (Tx*Ty,)
            pairs = tile.reshape(-1, V)                            # (Tx*Ty/V, 2)
            # For each pair, find nearest codebook entry and record distance.
            for p in pairs:
                dists = np.abs(cb_np - p).sum(axis=1)
                nearest = dists.min()
                max_dev = max(max_dev, float(nearest))
                total_pairs += 1
                if nearest > 1e-4:
                    mismatches += 1

    print(f"  pairs checked: {total_pairs}")
    print(f"  max deviation from nearest codebook entry: {max_dev:.2e}")
    print(f"  mismatches (> 1e-4): {mismatches}")
    ok = max_dev < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] all pairs on codebook (< 1e-4)")
    return ok


def test_ste_backward_identity():
    print("\nQ.0.1.3: STE backward is identity")
    print("-" * 60)
    W_latent, sign_l, sign_r, cb, _ = _setup(m=64, n=64)
    W_q = TrellisQuantSTE.apply(W_latent, sign_l, sign_r, cb)

    # Arbitrary upstream gradient
    torch.manual_seed(123)
    g = torch.randn_like(W_q)
    W_q.backward(g)

    grad = W_latent.grad
    max_diff = (grad - g).abs().max().item()
    print(f"  max |W_latent.grad - g|: {max_diff:.2e}")
    print(f"  grad is finite: {bool(torch.isfinite(grad).all().item())}")
    ok = max_diff < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] identity backward (diff < 1e-6)")
    return ok


def test_end_to_end_autograd():
    print("\nQ.0.1.4: end-to-end scalar loss -> finite grad on W_latent")
    print("-" * 60)
    W_latent, sign_l, sign_r, cb, _ = _setup(m=64, n=64)
    W_q = TrellisQuantSTE.apply(W_latent, sign_l, sign_r, cb)

    loss = (W_q ** 2).sum()
    loss.backward()

    grad = W_latent.grad
    finite = bool(torch.isfinite(grad).all().item())
    nonzero = float(grad.abs().mean().item()) > 1e-6
    # Expected STE gradient for (W_q**2).sum() is 2*W_q (as an identity passthrough),
    # i.e. grad(W_latent) = 2 * W_q. Verify.
    expected = 2.0 * W_q.detach()
    match_err = (grad - expected).abs().max().item()

    print(f"  loss: {loss.item():.4f}")
    print(f"  grad finite: {finite}")
    print(f"  grad mean abs: {grad.abs().mean().item():.4f}")
    print(f"  max |grad - 2*W_q|: {match_err:.2e}")
    ok = finite and nonzero and match_err < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] end-to-end grad correct")
    return ok


def main():
    print("=" * 60)
    print("Tripwire Q.0.1: STE forward/backward sanity")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available (kernel requires CUDA)")
        sys.exit(1)

    results = []
    results.append(("forward_shape_finite", test_forward_shape_finite()))
    results.append(("on_codebook", test_on_codebook()))
    results.append(("ste_backward_identity", test_ste_backward_identity()))
    results.append(("end_to_end_autograd", test_end_to_end_autograd()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("Q.0.1 GATE: PASS — STE wiring is sound.")
        print("Ready for Q.0.2 (QuantizedLinear module).")
        sys.exit(0)
    else:
        print("Q.0.1 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
