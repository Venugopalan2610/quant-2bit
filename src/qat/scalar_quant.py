"""Scalar (uniform) 2-bit QAT baselines — the ParetoQ-analog comparison arms.

These exist to answer the load-bearing question BCJR-QAT has NOT yet tested:
does the *trellis representation* beat a *scalar* representation at matched
2-bit QAT budget? Beating QTIP-PTQ (the paper's −0.084 result) only shows
trellis-trained > trellis-untrained. It says nothing about trellis vs scalar.

Two arms, proving two different things (see scripts/rung_scalar_*.sh):

  CONTROLLED (scalar_quant_controlled):
    Same RHT rotation + same single global RMS scale as the trellis path
    (src/qat/ste.py). ONLY the quantizer differs: uniform 4-level + STE
    instead of the trellis codebook. Exactly 2.000 bits/weight, both arms.
    Isolates the representation as the single variable. Weakness: uniform
    levels on RHT-Gaussianized weights is a weak scalar by construction.

  FAITHFUL (scalar_quant_faithful):
    The real deployed scalar recipe: NATIVE basis (no RHT), per-group scale
    from a LEARNED clip, uniform 4-level + STE. This is what ParetoQ/BitNet
    actually do. Costs scale-overhead bits (see effective_bits_faithful):
    a per-group fp16 scale at group=128 is +0.125 b/w → ~2.125 effective.
    Report that honestly — if trellis@2.000b beats faithful-scalar@2.125b
    (MORE bits), the "you rigged the bitrate" objection is dead.

Gradient: straight-through estimator on the weights (round is identity in
backward), identical to TrellisQuantSTE. The faithful arm's clip is a normal
learnable parameter (real gradient, not STE) trained jointly.
"""
import numpy as np
import torch

from src.rht.transform import apply_rht, apply_inverse_rht


def _symmetric_levels(n_bits):
    """Mid-rise symmetric uniform levels for an n-bit quantizer, in units of
    the step Δ=1. For 2-bit: 4 levels {-1.5, -0.5, +0.5, +1.5}. Symmetric
    (no exact zero) — matches the standard symmetric int quantizer used by
    weight-only 2-bit QAT (ParetoQ's symmetric setting)."""
    n = 1 << n_bits                       # 4 levels for 2-bit
    # mid-rise: centers at (k - (n-1)/2) for k=0..n-1  → {-1.5,-0.5,0.5,1.5}
    return torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0


def _round_ste(x):
    """Round-to-nearest with a straight-through (identity) backward."""
    return (torch.round(x) - x).detach() + x


def _quantize_levels(W_unit, n_bits, ste):
    """Round unit-scaled weights to n-bit symmetric mid-rise levels.

    n_bits=2 → {-1.5,-0.5,+0.5,+1.5}, step Δ=1. If ste=True the round has an
    identity backward (gradient flows straight through); if False it is a hard
    no-grad round (used inside custom autograd Functions where backward is
    defined separately). Returns values in level units (Δ=1)."""
    levels = _symmetric_levels(n_bits).to(W_unit.device, W_unit.dtype)
    lo, hi = float(levels[0]), float(levels[-1])
    shifted = W_unit - 0.5                          # mid-rise → integer grid
    r = _round_ste(shifted) if ste else torch.round(shifted)
    return (r + 0.5).clamp(min=lo, max=hi)


def uniform_quantize_unit(W_unit, n_bits=2):
    """STE uniform quantize of unit-scaled weights (used by the faithful arm,
    which needs gradient to flow to both the weights (STE) and the clip)."""
    return _quantize_levels(W_unit, n_bits, ste=True)


# ── Arm: CONTROLLED (RHT + single global scale, only quantizer swapped) ──────

class _ScalarControlledSTE(torch.autograd.Function):
    """Uniform 2-bit in the trellis path's RHT-rotated basis + global RMS scale.

    Forward: RHT-rotate → /global-rms → uniform round → ×rms → inverse-RHT.
    Backward: identity (dL/dW_latent = dL/dW_q) — EXACTLY the convention
    TrellisQuantSTE uses (src/qat/ste.py:330), so the only thing differing
    between this arm and the trellis arm is the quantizer, never the gradient.
    """
    @staticmethod
    def forward(ctx, W_latent, sign_l, sign_r, n_bits):
        dev, dt = W_latent.device, W_latent.dtype
        W = W_latent.detach().float()
        rht, inv_rht = _rht_fns(W)            # GPU-native FHT on CUDA, numpy on CPU
        W_tilde = rht(W, sign_l, sign_r)
        scale = W_tilde.pow(2).mean().sqrt().clamp(min=1e-30)
        W_q_unit = _quantize_levels(W_tilde / scale, n_bits, ste=False)
        W_tilde_q = W_q_unit * scale
        W_q = inv_rht(W_tilde_q, sign_l, sign_r)
        return W_q.to(device=dev, dtype=dt)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None     # identity STE; non-W args → None


def scalar_quant_controlled(W_latent, sign_l, sign_r, n_bits=2):
    """Uniform 2-bit in the SAME RHT basis + SAME global RMS scale as the
    trellis path. Only the quantizer differs; bit budget is exactly n_bits/w
    (the single global scale is negligible). Identity-STE gradient, matching
    TrellisQuantSTE. CPU (numpy RHT) or CUDA."""
    return _ScalarControlledSTE.apply(W_latent, sign_l, sign_r, n_bits)


# ── Arm: FAITHFUL (RHT basis, per-group MSE-optimal clip — fair single var) ──
#
# Corrected design (see PREREGISTRATION.md / results/matched_l4_ladder.md):
# the faithful scalar runs in the SAME RHT basis as the trellis (so the only
# variable is the quantizer, not the rotation — native 2-bit without RHT is
# catastrophic and would be an unfair strawman), and uses a FIXED MSE-optimal
# per-group clip. Fixed, not learned: the trellis arm's scale is also fixed
# (global RMS), so a learned clip would give scalar an extra DoF the trellis
# lacks. Per-group adaptivity is scalar's real advantage; MSE-optimal captures
# it without the 1/clip² training instability. Costs scale-overhead bits
# (~2.125b @group128, reported via effective_bits_faithful).

def _mse_optimal_clip_per_group(Wg, n_bits, n_grid=25):
    """Per-group clip minimizing uniform-quant MSE. Wg: (out, n_groups, G).
    Grid-searches clip = α·max|w| over α∈[0.4,1.0]. Returns (out, n_groups, 1)."""
    max_abs = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    max_level = float(_symmetric_levels(n_bits)[-1])
    best_clip = max_abs.clone()
    best_mse = torch.full_like(max_abs, float("inf"))
    for alpha in torch.linspace(0.4, 1.0, n_grid).tolist():
        clip = alpha * max_abs
        step = clip / max_level
        q = _quantize_levels(Wg / step, n_bits, ste=False) * step
        mse = (Wg - q).pow(2).mean(dim=-1, keepdim=True)
        better = mse < best_mse
        best_clip = torch.where(better, clip, best_clip)
        best_mse = torch.where(better, mse, best_mse)
    return best_clip


class _ScalarFaithfulRHTSTE(torch.autograd.Function):
    """Per-group uniform n-bit in the RHT basis with fixed MSE-optimal clip.
    Identity-STE backward (matches the trellis/controlled arms)."""
    @staticmethod
    def forward(ctx, W_latent, sign_l, sign_r, group_size, n_bits):
        dev, dt = W_latent.device, W_latent.dtype
        W = W_latent.detach().float()
        rht, inv_rht = _rht_fns(W)
        W_tilde = rht(W, sign_l, sign_r)
        out_f, in_f = W_tilde.shape
        assert in_f % group_size == 0, f"in {in_f} not divisible by group {group_size}"
        Wg = W_tilde.view(out_f, in_f // group_size, group_size)
        clip = _mse_optimal_clip_per_group(Wg, n_bits)
        step = clip / float(_symmetric_levels(n_bits)[-1])
        W_q_unit = _quantize_levels(Wg / step, n_bits, ste=False)
        W_tilde_q = (W_q_unit * step).view(out_f, in_f)
        W_q = inv_rht(W_tilde_q, sign_l, sign_r)
        return W_q.to(device=dev, dtype=dt)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def scalar_quant_faithful(W_latent, sign_l, sign_r, group_size=128, n_bits=2):
    """Per-group MSE-optimal uniform n-bit in the RHT basis. STE to W_latent.
    Single-variable vs the trellis (same basis, fixed scale; only quantizer
    differs). Clip is recomputed per call from the current weights — PTQ-style,
    matching how the trellis recomputes its RMS scale each step."""
    return _ScalarFaithfulRHTSTE.apply(W_latent, sign_l, sign_r, group_size, n_bits)


def effective_bits_faithful(out_f, in_f, group_size, n_bits=2, scale_bits=16):
    """Honest effective bits/weight for the faithful arm, counting the
    per-group clip/scale overhead. This is the number you REPORT next to the
    trellis arm's clean n_bits."""
    weights = out_f * in_f
    n_groups = out_f * (in_f // group_size)
    overhead = n_groups * scale_bits
    return n_bits + overhead / weights


# ── helpers ──────────────────────────────────────────────────────────────────

def _np(x):
    """RHT helpers in src.rht.transform want ±1 numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _rht_fns(W):
    """Return (rht, inv_rht) closures matching the trellis path's RHT.

    On CUDA with fast-hadamard-transform available, use the GPU-native FHT
    (src.qat.ste.apply_rht_gpu) — NO CPU round-trip (pre-mortem fix #7). On
    CPU, fall back to the numpy FWHT. Both closures take (W, sign_l, sign_r)
    and return a torch tensor on W's device."""
    if W.is_cuda:
        try:
            from src.qat.ste import apply_rht_gpu, apply_inverse_rht_gpu, _HAS_FHT
        except Exception:
            _HAS_FHT = False
        if _HAS_FHT:
            def rht(w, sl, sr):
                return apply_rht_gpu(w, _t(sl, w), _t(sr, w))
            def inv_rht(w, sl, sr):
                return apply_inverse_rht_gpu(w, _t(sl, w), _t(sr, w))
            return rht, inv_rht

    def rht(w, sl, sr):
        out = apply_rht(w.detach().cpu(), _np(sl), _np(sr))
        return torch.as_tensor(out, dtype=torch.float32, device=w.device)

    def inv_rht(w, sl, sr):
        out = apply_inverse_rht(w.detach().cpu(), _np(sl), _np(sr))
        return torch.as_tensor(out, dtype=torch.float32, device=w.device)
    return rht, inv_rht


def _t(sign, ref):
    """±1 sign vector → float32 tensor on ref's device (for the GPU RHT)."""
    if isinstance(sign, torch.Tensor):
        return sign.to(device=ref.device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(sign, dtype=np.float32), device=ref.device)
