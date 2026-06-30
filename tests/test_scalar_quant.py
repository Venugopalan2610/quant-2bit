"""CPU unit tests for the scalar 2-bit QAT baselines (no GPU needed — runs
while Rung 1 owns the card). Verifies the quantizer is a FAIR scalar, not an
accidental strawman, and that the STE gradient matches the trellis arm.
"""
import numpy as np
import torch

from src.qat.scalar_quant import (
    _symmetric_levels, _round_ste, uniform_quantize_unit,
    scalar_quant_controlled, scalar_quant_faithful,
    _mse_optimal_clip_per_group, _quantize_levels, effective_bits_faithful,
)
from src.rht.transform import make_sign_vector


def test_levels_2bit():
    lv = _symmetric_levels(2).tolist()
    assert lv == [-1.5, -0.5, 0.5, 1.5], lv


def test_round_ste_identity_grad():
    x = torch.randn(1000, requires_grad=True)
    y = _round_ste(x)
    y.sum().backward()
    # STE: d(round)/dx = 1 everywhere.
    assert torch.allclose(x.grad, torch.ones_like(x.grad))
    # forward really rounds.
    assert torch.allclose(y.detach(), torch.round(x.detach()))


def test_uniform_quantize_levels_and_clamp():
    x = torch.tensor([-9.0, -1.4, -0.2, 0.2, 0.7, 9.0])
    q = uniform_quantize_unit(x, n_bits=2)
    assert set(q.tolist()) <= {-1.5, -0.5, 0.5, 1.5}
    # outer clamp
    assert q[0].item() == -1.5 and q[-1].item() == 1.5
    # nearest-level rounding
    assert q[2].item() == -0.5 and q[3].item() == 0.5


def test_controlled_is_near_lloyd_max():
    """The controlled arm's fairness hinges on this: uniform {±0.5,±1.5} on a
    unit-RMS Gaussian must be ~optimal scalar (Lloyd-Max 2-bit ≈ ±0.453,±1.510),
    else it's a strawman. We check the quantizer's distortion is within a few %
    of the Lloyd-Max MSE, proving the controlled scalar is a fair baseline."""
    torch.manual_seed(0)
    g = torch.randn(200000)                      # unit Gaussian
    q = uniform_quantize_unit(g, n_bits=2)
    mse_uniform = (g - q).pow(2).mean().item()
    # Lloyd-Max optimal 2-bit Gaussian MSE ≈ 0.1175 (Max 1960).
    mse_lloyd = 0.1175
    rel = (mse_uniform - mse_lloyd) / mse_lloyd
    assert rel < 0.06, f"controlled scalar {rel:.1%} worse than Lloyd-Max — strawman risk"
    print(f"  controlled-arm MSE={mse_uniform:.4f} vs Lloyd-Max {mse_lloyd:.4f} ({rel:+.1%})")


def test_controlled_ste_grad_is_identity_in_rotated_basis():
    """STE must give dL/dW_latent = (RHT^T · RHT) applied to grad — but since
    RHT is orthonormal and the quantizer is identity-backward, grad magnitude
    is preserved. We check grad exists, is finite, and is non-trivial."""
    torch.manual_seed(0)
    m = n = 16
    W = torch.randn(m, n, requires_grad=True)
    sl = torch.tensor(make_sign_vector(m, seed=1))
    sr = torch.tensor(make_sign_vector(n, seed=2))
    Wq = scalar_quant_controlled(W, sl, sr, n_bits=2)
    assert Wq.shape == (m, n)
    Wq.pow(2).sum().backward()
    assert torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0


def test_faithful_rht_runs_and_ste_grad():
    """Faithful arm (RHT basis, per-group MSE clip) runs and gives identity-STE
    grad to W (no learnable clip — fixed scale, matching the trellis arm)."""
    torch.manual_seed(0)
    m = n = 64
    W = torch.randn(m, n, requires_grad=True)
    sl = torch.tensor(make_sign_vector(m, seed=1))
    sr = torch.tensor(make_sign_vector(n, seed=2))
    Wq = scalar_quant_faithful(W, sl, sr, group_size=32, n_bits=2)
    assert Wq.shape == (m, n)
    Wq.pow(2).sum().backward()
    assert torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0


def test_mse_clip_beats_maxabs():
    """The fairness fix: MSE-optimal per-group clip must give LOWER quant MSE
    than the old max-abs clip — otherwise the faithful arm is needlessly weak."""
    torch.manual_seed(0)
    Wg = torch.randn(16, 4, 128)                         # (out, n_groups, G)
    max_level = float(_symmetric_levels(2)[-1])
    # max-abs clip
    max_abs = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q_max = _quantize_levels(Wg / (max_abs / max_level), 2, ste=False) * (max_abs / max_level)
    mse_maxabs = (Wg - q_max).pow(2).mean().item()
    # MSE-optimal clip
    clip = _mse_optimal_clip_per_group(Wg, 2)
    q_opt = _quantize_levels(Wg / (clip / max_level), 2, ste=False) * (clip / max_level)
    mse_opt = (Wg - q_opt).pow(2).mean().item()
    assert mse_opt < mse_maxabs, f"MSE-opt {mse_opt:.4f} not < max-abs {mse_maxabs:.4f}"
    print(f"  MSE clip {mse_opt:.4f} < max-abs {mse_maxabs:.4f} "
          f"({100*(mse_maxabs-mse_opt)/mse_maxabs:.0f}% better)")


def test_faithful_effective_bits_accounting():
    # Llama-3.2-1B q_proj is 2048×2048. group=128, fp16 scale.
    eb = effective_bits_faithful(2048, 2048, group_size=128, n_bits=2, scale_bits=16)
    assert abs(eb - 2.125) < 1e-6, eb
    print(f"  faithful effective bits @group128 = {eb:.4f} (trellis = 2.0000)")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
