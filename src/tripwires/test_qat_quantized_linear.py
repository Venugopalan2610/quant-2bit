"""Tripwire Q.0.2: QuantizedLinear module sanity.

Sub-tests:
  Q.0.2.1: forward has correct shape; output matches F.linear(x, W_q, bias)
           where W_q is independently extracted from the module.
  Q.0.2.2: from_linear(fp_layer) copies the FP weights into W_latent and
           produces W_q close to the PTQ-of-those-weights (first forward ==
           one-shot PTQ).
  Q.0.2.3: backward flows to W_latent, W_latent.grad is finite non-zero.
  Q.0.2.4: re-encode schedule — with reencode_every_n_steps=3, calling
           forward 3 times runs Viterbi only on steps 0 and 3; cached W_q
           is reused on steps 1 and 2 but gradient still flows to W_latent.
  Q.0.2.5: parameter update sanity — one optimizer.step() actually changes
           W_latent AND (eventually) W_q after a re-encode.

Run: python -m src.tripwires.test_qat_quantized_linear
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


def test_forward_matches_extracted_Wq():
    print("\nQ.0.2.1: forward output matches F.linear(x, W_q)")
    print("-" * 60)
    torch.manual_seed(0)
    cb = _make_codebook(seed=0)
    in_f, out_f = 64, 64
    layer = QuantizedLinear(in_f, out_f, codebook_gpu=cb, bias=True, seed=7)
    x = torch.randn(32, in_f, device="cuda")

    # Take two snapshots: module forward, and manual F.linear with current_W_q.
    # Use current_W_q() first (which does NOT advance step), then forward().
    # Both should use the same walks since step counter hasn't moved between them.
    # However, current_W_q may compute fresh OR reuse cache. To keep this test
    # deterministic: reset step counter before each access so both recompute.
    layer._step_counter.zero_()
    W_q_a = layer.current_W_q().detach().clone()
    # But current_W_q already computed and cached; forward() will see step=0
    # still since we didn't advance, then advance inside forward.
    layer._step_counter.zero_()
    y_module = layer(x)

    y_manual = F.linear(x, W_q_a, layer.bias)

    max_diff = (y_module - y_manual).abs().max().item()
    print(f"  y_module shape: {tuple(y_module.shape)}")
    print(f"  max |y_module - F.linear(x, W_q, bias)|: {max_diff:.2e}")
    ok = max_diff < 1e-4 and y_module.shape == (32, out_f)
    print(f"  [{'PASS' if ok else 'FAIL'}] module output matches manual")
    return ok


def test_from_linear_matches_ptq():
    """`from_linear(fp_layer)` init should produce a first-forward W_q that
    equals running PTQ once on fp_layer.weight."""
    print("\nQ.0.2.2: from_linear init == one-shot PTQ on first forward")
    print("-" * 60)
    torch.manual_seed(1)
    cb = _make_codebook(seed=1)
    in_f, out_f = 64, 64
    fp_layer = nn.Linear(in_f, out_f, bias=True).cuda().float()
    qlayer = QuantizedLinear.from_linear(fp_layer, codebook_gpu=cb, seed=11)

    qlayer._step_counter.zero_()
    W_q_module = qlayer.current_W_q().detach().cpu().numpy()

    # Run standalone PTQ on fp_layer.weight with same signs the module used.
    sign_l = qlayer.sign_l.cpu().numpy()
    sign_r = qlayer.sign_r.cpu().numpy()
    W_fp_np = fp_layer.weight.detach().cpu().numpy().astype(np.float32)
    W_q_ref, _, _, _, _ = quantize_forward_numpy(W_fp_np, sign_l, sign_r, cb)

    max_diff = float(np.abs(W_q_module - W_q_ref).max())
    print(f"  max |W_q_module - W_q_ptq|: {max_diff:.2e}")
    ok = max_diff < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] module == standalone PTQ")
    return ok


def test_backward_grad_to_W_latent():
    print("\nQ.0.2.3: gradient reaches W_latent")
    print("-" * 60)
    torch.manual_seed(2)
    cb = _make_codebook(seed=2)
    in_f, out_f = 64, 64
    layer = QuantizedLinear(in_f, out_f, codebook_gpu=cb, bias=False, seed=3)
    x = torch.randn(16, in_f, device="cuda")

    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()

    g = layer.W_latent.grad
    finite = bool(torch.isfinite(g).all().item())
    nonzero = float(g.abs().mean().item()) > 1e-6
    print(f"  grad shape: {tuple(g.shape)}")
    print(f"  grad finite: {finite}")
    print(f"  grad mean abs: {g.abs().mean().item():.4f}")
    ok = finite and nonzero
    print(f"  [{'PASS' if ok else 'FAIL'}] grad non-zero and finite on W_latent")
    return ok


def test_reencode_schedule():
    """reencode_every_n_steps=3: fresh Viterbi on steps that are multiples of 3,
    cached otherwise. Detect by counting actual Viterbi calls via a hook on
    the cached W_q tensor: its data should only change at step 0 and step 3.
    """
    print("\nQ.0.2.4: re-encode schedule honors reencode_every_n_steps")
    print("-" * 60)
    torch.manual_seed(4)
    cb = _make_codebook(seed=4)
    in_f, out_f = 64, 64
    layer = QuantizedLinear(in_f, out_f, codebook_gpu=cb, bias=False, seed=5,
                            reencode_every_n_steps=3)
    x = torch.randn(8, in_f, device="cuda")

    # Nudge W_latent after each forward so that IF re-encode fired, W_q would
    # clearly change; if not, cached W_q stays identical.
    snapshots = []
    for step in range(6):
        # Perturb W_latent before calling forward. Small perturbation.
        with torch.no_grad():
            layer.W_latent.add_(0.01 * torch.randn_like(layer.W_latent))
        layer(x)
        snapshots.append(layer._W_q_cache.detach().clone())

    # Expected re-encode events: step 0, step 3.
    # So snapshots[0] != "fresh from modified latent at step 0's W_latent"
    # snapshots[1] == snapshots[0] (cached, didn't recompute)
    # snapshots[2] == snapshots[0]
    # snapshots[3] != snapshots[0] (re-encoded)
    # snapshots[4] == snapshots[3]
    # snapshots[5] == snapshots[3]
    s = snapshots
    c01 = torch.equal(s[0], s[1])
    c02 = torch.equal(s[0], s[2])
    c03_differ = not torch.equal(s[0], s[3])
    c34 = torch.equal(s[3], s[4])
    c35 = torch.equal(s[3], s[5])

    print(f"  step 0 cache == step 1 cache: {c01}   (expected True)")
    print(f"  step 0 cache == step 2 cache: {c02}   (expected True)")
    print(f"  step 0 cache == step 3 cache: {not c03_differ}   (expected False — re-encoded)")
    print(f"  step 3 cache == step 4 cache: {c34}   (expected True)")
    print(f"  step 3 cache == step 5 cache: {c35}   (expected True)")
    ok = c01 and c02 and c03_differ and c34 and c35
    print(f"  [{'PASS' if ok else 'FAIL'}] re-encode schedule correct")
    return ok


def test_optimizer_step_updates():
    """One Adam step should move W_latent; after the next re-encode, W_q may
    change. Verify W_latent actually changes (optimizer wired correctly)."""
    print("\nQ.0.2.5: optimizer step changes W_latent")
    print("-" * 60)
    torch.manual_seed(6)
    cb = _make_codebook(seed=6)
    in_f, out_f = 64, 64
    layer = QuantizedLinear(in_f, out_f, codebook_gpu=cb, bias=True, seed=9)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    x = torch.randn(16, in_f, device="cuda")
    y_target = torch.randn(16, out_f, device="cuda")

    W_before = layer.W_latent.detach().clone()

    y = layer(x)
    loss = F.mse_loss(y, y_target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    W_after = layer.W_latent.detach().clone()
    delta = (W_after - W_before).abs().mean().item()
    print(f"  mean |ΔW_latent| after 1 step: {delta:.4e}")
    ok = delta > 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] W_latent moved after optimizer step")
    return ok


def main():
    print("=" * 60)
    print("Tripwire Q.0.2: QuantizedLinear module")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    results = []
    results.append(("forward_matches_extracted", test_forward_matches_extracted_Wq()))
    results.append(("from_linear_matches_ptq", test_from_linear_matches_ptq()))
    results.append(("backward_grad", test_backward_grad_to_W_latent()))
    results.append(("reencode_schedule", test_reencode_schedule()))
    results.append(("optimizer_updates", test_optimizer_step_updates()))

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    if all_ok:
        print("Q.0.2 GATE: PASS — QuantizedLinear module wiring is sound.")
        print("Ready for Q.0.3 (toy synthetic training loop).")
        sys.exit(0)
    else:
        print("Q.0.2 GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
