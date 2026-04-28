"""BCJR-based trellis quantizer: a differentiable replacement for Viterbi+STE.

Pipeline (mirrors pre-IP STE in src/qat/ste.py):
    W_latent (original basis)
      --apply_rht_gpu-->  W_tilde (rotated)
      --scale by rms-->   W_tilde_unit
      --tile (Tx, Ty)-->  sequences (B, N, V)       B = n_row_blocks*n_col_blocks
      --BCJR forward-backward at T-->  soft_cw (B, N, V)
      --untile-->         W_tilde_q_unit (m, n)
      --* W_scale-->      W_tilde_q (rotated)
      --apply_inverse_rht_gpu-->  W_q (original basis)

The BCJR core is autograd-native (no custom backward). Per-chunk memory is
bounded by processing tiles in chunks of `chunk_size` and wrapping each chunk
in torch.utils.checkpoint so only chunk inputs are retained for backward.

Memory per chunk during backward recompute:
    log_alpha + log_beta + log_local + P ≈ 4 * (chunk_size * N * S * 4 bytes)
    = 4 * (chunk_size * 128 * 65536 * 4) ≈ chunk_size * 128 MB
chunk_size=8 → ~1 GB peak per chunk during backward (fits on 12 GB).

Temperature annealing:
    Pass T_temp as a tensor argument (scalar). Outside code schedules T.
    At T → 0, soft_cw concentrates on the Viterbi MAP state → hard quant.

Integration point: QuantizedLinear with `quant_mode='bcjr'` calls
bcjr_quantize_weight_with_rht() in place of TrellisQuantSTE.apply().
"""
import math

import torch
import torch.utils.checkpoint as checkpoint

import os

from src.bcjr.forward_backward import (
    L_BITS, K_BITS, V_DIM,
    build_pred_succ_tables,
    bcjr_forward_backward,
)

# Opt-in fast path via env var so we don't silently change numerics. Enable
# with BCJR_FAST=1. Parity tested in src/tripwires/test_bcjr_fast_parity.py.
if os.environ.get("BCJR_FAST") == "1":
    from src.bcjr.forward_backward_fast import bcjr_forward_backward_fast as _bcjr_fb
else:
    _bcjr_fb = bcjr_forward_backward
from src.qat.ste import (
    TX, TY,
    _tile_for_viterbi_gpu,
    _untile_from_viterbi_gpu,
    apply_rht_gpu,
    apply_inverse_rht_gpu,
)


# Per-chunk batch size (number of tiles processed in one BCJR call).
# chunk_size=8 keeps peak memory ~1 GB on 12 GB GPUs. Tune upward on
# bigger cards for fewer kernel launches.
BCJR_CHUNK = 8


def _bcjr_chunk_forward(sequences_chunk, codebook, T_temp, preds, succs):
    """One BCJR call. Returns soft codewords (B, N, V).

    Dispatch:
      BCJR_MONOLITHIC=1 → single-autograd-node Triton path (6.57× on 4080)
      BCJR_FAST=1       → no-per-step-checkpoint Triton path
      else              → reference eager implementation
    """
    if os.environ.get("BCJR_MONOLITHIC") == "1":
        from src.bcjr.bcjr_monolithic import bcjr_forward_backward_monolithic
        soft, _ = bcjr_forward_backward_monolithic(
            sequences_chunk, codebook, T_temp,
            preds=preds, succs=succs,
        )
        return soft
    soft, _ = _bcjr_fb(
        sequences_chunk, codebook, T_temp,
        preds=preds, succs=succs,
    )
    return soft


def bcjr_quantize_tiles(sequences, codebook, T_temp, preds, succs,
                        chunk_size=BCJR_CHUNK, use_checkpoint=True):
    """Run BCJR on (B, N, V) sequences in chunks. Returns (B, N, V) soft outputs.

    Each chunk is wrapped in torch.utils.checkpoint when `use_checkpoint=True`
    so autograd stores only the chunk inputs — the full forward-backward
    tensors are recomputed in backward. Saves ~4× memory per chunk at a
    ~2× compute cost.

    If BCJR_GRAPH=1 env is set, the chunk forward+backward is replaced with
    a CUDA-graph-captured version (see src/bcjr/bcjr_graph.py). In this
    mode we MUST bypass torch.utils.checkpoint — checkpoint's conditional
    re-forward is not capturable.
    """
    use_graph = os.environ.get("BCJR_GRAPH") == "1"
    if use_graph:
        # Import lazily to avoid a cycle at module load.
        from src.bcjr.bcjr_graph import graphed_bcjr_chunk_forward
        chunk_fn = graphed_bcjr_chunk_forward
    else:
        chunk_fn = _bcjr_chunk_forward

    B = sequences.shape[0]
    outs = []
    for i in range(0, B, chunk_size):
        chunk = sequences[i:i + chunk_size]
        if use_graph:
            # Graphed path handles its own autograd — no outer checkpoint.
            soft = chunk_fn(chunk, codebook, T_temp, preds, succs)
        elif use_checkpoint and chunk.requires_grad:
            soft = checkpoint.checkpoint(
                chunk_fn, chunk, codebook, T_temp, preds, succs,
                use_reentrant=False,
            )
        else:
            soft = chunk_fn(chunk, codebook, T_temp, preds, succs)
        outs.append(soft)
    return torch.cat(outs, dim=0)


def bcjr_quantize_weight_rotated(W_rotated, codebook, T_temp, preds, succs,
                                 chunk_size=BCJR_CHUNK, Tx=TX, Ty=TY):
    """Quantize an already-rotated weight via BCJR. Returns W_q in same basis.

    Mirrors quantize_forward_gpu_no_rht() in src/qat/ste.py:
      - scale by RMS (stop-grad on the scale so gradient flows straight through)
      - tile into (B, N, V) sequences
      - run BCJR at temperature T
      - untile and rescale
    """
    m, n = W_rotated.shape
    # Scale must be detached from the graph or else the BCJR soft output
    # scales with W_latent in a way that fights the finite-T soft behavior.
    # STE (quantize_forward_gpu_no_rht) similarly uses detach via pow/sqrt
    # on a detached tensor; here we match by explicit detach.
    W_scale = W_rotated.detach().pow(2).mean().sqrt().clamp(min=1e-30)
    W_tilde_unit = W_rotated / W_scale

    # Tile: (B, T_seq) → (B, N, V) where N = T_seq/V, V=2
    sequences_flat, shape_info = _tile_for_viterbi_gpu(W_tilde_unit, Tx=Tx, Ty=Ty)
    B, T_seq = sequences_flat.shape
    N = T_seq // V_DIM
    sequences = sequences_flat.view(B, N, V_DIM)

    soft = bcjr_quantize_tiles(
        sequences, codebook, T_temp, preds, succs, chunk_size=chunk_size,
    )

    # Back to (B, T_seq) then untile to (m, n)
    soft_flat = soft.contiguous().view(B, T_seq)
    W_tilde_q_unit = _untile_from_viterbi_gpu(soft_flat, shape_info, (m, n), Tx, Ty)
    return W_tilde_q_unit * W_scale


def bcjr_quantize_weight(W_latent, sign_l, sign_r, codebook, T_temp,
                         preds=None, succs=None, chunk_size=BCJR_CHUNK,
                         Tx=TX, Ty=TY):
    """Full BCJR quantize with RHT wrapping (pre-IP style).

    W_latent is in original basis; we rotate, run BCJR, unrotate. Matches
    src/qat/ste.py:quantize_forward_gpu()'s data flow but soft instead of hard.

    Gradient flows: through RHT (linear, differentiable), through BCJR
    (autograd-native), through inverse RHT (linear). No STE needed.
    """
    if preds is None or succs is None:
        preds, succs = build_pred_succ_tables(device=W_latent.device)
    W_tilde = apply_rht_gpu(W_latent, sign_l, sign_r)
    W_tilde_q = bcjr_quantize_weight_rotated(
        W_tilde, codebook, T_temp, preds, succs,
        chunk_size=chunk_size, Tx=Tx, Ty=Ty,
    )
    W_q = apply_inverse_rht_gpu(W_tilde_q, sign_l, sign_r)
    return W_q
