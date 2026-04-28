"""Temperature annealing + BCJR-mode switching for QAT.

Usage in a training driver:

    from src.bcjr.anneal import (
        measure_ptq_residual, convert_layer_to_bcjr, convert_layer_to_ste,
        cosine_temperature_schedule, set_layer_temperature,
        entropy_diagnostic,
    )

    T_0 = measure_ptq_residual(qat_layer)       # principled starting T
    T_min = max(1e-2, T_0 * 1e-2)               # fp32 floor
    convert_layer_to_bcjr(qat_layer, T_init=T_0)

    for step in range(n_steps):
        T = cosine_temperature_schedule(step, n_steps, T_0, T_min)
        set_layer_temperature(qat_layer, T)
        # ... forward, loss, backward, opt.step ...

    convert_layer_to_ste(qat_layer)             # hard Viterbi from here on
    qat_layer.prime_cache()                     # cache now holds hard W_q

Philosophy:
  T_0 = mean ||W - W_q_STE||² / n_weights (the "natural" quant noise scale).
      At this T, Boltzmann weights e^{-E/T} treat codewords within ~√T of
      the input as comparable. Gradient pressure is maximally informative
      since many paths are non-trivially populated.

  Anneal schedule is cosine (not linear) so T spends more time in the
  productive low-T regime where marginals are sharp and gradients give
  real codebook preferences rather than soft averages.

  T_min ≥ 1e-2 for fp32 stability (log_alpha magnitudes reach ~1e7 below
  this; catastrophic cancellation in log_alpha + log_beta - log_Z).
"""
import math

import torch

from src.bcjr.forward_backward import bcjr_forward_backward, build_pred_succ_tables
from src.bcjr.bcjr_quant import BCJR_CHUNK
from src.qat.ste import (
    TrellisQuantSTE,
    apply_rht_gpu,
    _tile_for_viterbi_gpu,
)


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------

def cosine_temperature_schedule(step, total_steps, T_0, T_min):
    """Cosine decay from T_0 at step=0 to T_min at step=total_steps-1.

    Returns T_min at/after the final step. Pure function — no side effects.
    """
    if total_steps <= 1:
        return T_min
    s = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return T_min + 0.5 * (T_0 - T_min) * (1.0 + math.cos(math.pi * s))


# ----------------------------------------------------------------------
# Mode switching
# ----------------------------------------------------------------------

def _all_qls(qat_layer):
    """Unified iterator over every QuantizedLinear in a QATDecoderLayer.
    Uses the existing _all_qls() if present; otherwise falls back to a
    plain attribute walk (for single-target tests)."""
    if hasattr(qat_layer, "_all_qls"):
        yield from qat_layer._all_qls()
    else:
        yield ("ql", qat_layer)


def convert_layer_to_bcjr(qat_layer, T_init, bcjr_chunk=BCJR_CHUNK):
    """Flip every QL in the layer to BCJR mode at temperature T_init."""
    for _, ql in _all_qls(qat_layer):
        ql.set_quant_mode("bcjr")
        ql.set_temperature(T_init)
        ql.bcjr_chunk = int(bcjr_chunk)


def convert_layer_to_ste(qat_layer):
    """Flip every QL in the layer back to STE (hard Viterbi)."""
    for _, ql in _all_qls(qat_layer):
        ql.set_quant_mode("ste")


def set_layer_temperature(qat_layer, T):
    """Update BCJR temperature on every QL in the layer."""
    for _, ql in _all_qls(qat_layer):
        ql.set_temperature(T)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def measure_ptq_residual(qat_layer):
    """Return mean ||W_tilde - W_q_tilde||² / n_weights in UNIT (rotated+RMS-
    normalized) scale — the scale BCJR actually sees.

    BCJR's local energy E_t(s) = ||codebook[s] - w_unit_t||² is computed
    after rotation and W_scale normalization. So the natural T_0 must
    also be in that unit scale — otherwise mixing QLs with different
    W_scale gives inconsistent annealing.

    This is the natural T_0 for BCJR annealing: it puts the Boltzmann
    thermal noise on the same scale as the quantization noise floor, so
    near-optimal paths get comparable weight and soft-averaging is
    informative. At T ≫ this, all paths get uniform weight (flat
    gradient). At T ≪ this, only the Viterbi path survives (STE-like).
    """
    accum = 0.0
    count = 0
    with torch.no_grad():
        for _, ql in _all_qls(qat_layer):
            W_tilde = apply_rht_gpu(ql.W_latent, ql.sign_l, ql.sign_r)
            W_scale = W_tilde.pow(2).mean().sqrt().clamp(min=1e-30)
            W_tilde_unit = W_tilde / W_scale
            W_q = TrellisQuantSTE.apply(
                ql.W_latent, ql.sign_l, ql.sign_r, ql.codebook_gpu,
            )
            W_q_tilde = apply_rht_gpu(W_q, ql.sign_l, ql.sign_r)
            W_q_tilde_unit = W_q_tilde / W_scale
            diff = W_tilde_unit - W_q_tilde_unit
            accum += diff.pow(2).sum().item()
            count += diff.numel()
    return accum / max(count, 1)


def entropy_diagnostic(qat_layer, max_tiles=32):
    """Average marginal entropy H_t = -Σ_s P_t(s) log P_t(s) across a sample.

    Runs BCJR on at most `max_tiles` tiles per QL (in fp32, no grad) and
    reports entropy in nats. Ranges:
        H ≈ log(65536) ≈ 11.09  → uniform (very soft, T → ∞)
        H ≈ 0                    → concentrated on one state (hard, T → 0)
    """
    total_h = 0.0
    total_tiles = 0
    preds = None
    succs = None
    with torch.no_grad():
        for _, ql in _all_qls(qat_layer):
            if ql.quant_mode != "bcjr":
                continue
            if preds is None:
                preds, succs = build_pred_succ_tables(device=ql.W_latent.device)
            # Rotate + scale + tile — same as bcjr_quant
            W_tilde = apply_rht_gpu(ql.W_latent, ql.sign_l, ql.sign_r)
            W_scale = W_tilde.pow(2).mean().sqrt().clamp(min=1e-30)
            W_unit = W_tilde / W_scale
            seqs_flat, _ = _tile_for_viterbi_gpu(W_unit)
            B, T_seq = seqs_flat.shape
            N = T_seq // 2
            seqs = seqs_flat.view(B, N, 2)
            if B > max_tiles:
                idx = torch.randperm(B, device=seqs.device)[:max_tiles]
                seqs = seqs[idx]
            _, _, P, _, _ = bcjr_forward_backward(
                seqs, ql.codebook_gpu, ql.T_temp,
                preds=preds, succs=succs,
                return_internals=True,
            )
            # H per (b, t): -Σ_s P[b,t,s] log P[b,t,s]
            logP = (P.clamp(min=1e-30)).log()
            H_bt = -(P * logP).sum(dim=-1)                           # (B', N)
            total_h += H_bt.sum().item()
            total_tiles += H_bt.numel()
    if total_tiles == 0:
        return float("nan")
    return total_h / total_tiles
