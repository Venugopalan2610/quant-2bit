"""CUDA Graph-captured BCJR forward + backward.

`torch.cuda.make_graphed_callables` records the BCJR forward pass AND its
backward pass into replayable CUDA graphs. The N=128 Python time loop
inside BCJR becomes a single graph.replay() — ~500 kernel launches
collapse to ~2 (one per graph), eliminating Python dispatch overhead
which dominates on consumer GPUs.

Gated by BCJR_GRAPH=1 env var. Default is the reference (eager) path,
so behavior is unchanged unless the user opts in.

Capture constraints:
  - Shapes/dtypes fixed per graph. A new graph is built when shape changes.
  - Input tensors copied into static buffers before replay.
  - `codebook`, `preds`, `succs` are closure variables at capture time —
    they are identical across calls in a training run, so this is fine.
  - Graph capture uses its own memory pool; peak memory may be slightly
    higher than eager due to pre-allocated buffers.

Parity: src/tripwires/test_bcjr_graph_parity.py checks numerics against the
reference implementation within fp32 tolerance.
"""
import os

import torch

from src.bcjr.forward_backward import bcjr_forward_backward


# (sequences.shape, sequences.dtype, T_temp.dtype) → graphed callable
_GRAPH_CACHE = {}


def _make_graphed(sample_sequences, codebook, sample_T, preds, succs):
    """Build a graphed version of bcjr_forward_backward for the given
    (shape, dtype) of sample_sequences + sample_T. `codebook`, `preds`,
    `succs` are captured as closure constants.

    Returns a callable graphed_fn(sequences, T_temp) -> soft.
    """
    def fn(seq, T):
        soft, _ = bcjr_forward_backward(
            seq, codebook, T, preds=preds, succs=succs,
        )
        return soft

    # make_graphed_callables runs warmup iterations internally, then captures
    # forward graph and backward graph (if inputs require_grad). Sample args
    # must match the live inputs' shape/dtype exactly.
    graphed_fn = torch.cuda.make_graphed_callables(
        fn, sample_args=(sample_sequences, sample_T),
    )
    return graphed_fn


def graphed_bcjr_chunk_forward(sequences, codebook, T_temp, preds, succs):
    """Drop-in replacement for `_bcjr_chunk_forward` (bcjr_quant.py:_bcjr_chunk_forward).

    First call with a given shape/dtype captures the graph (slow, ~seconds).
    Subsequent calls replay it (fast — zero Python dispatch overhead).

    Falls through to reference implementation if BCJR_GRAPH != "1".
    """
    if os.environ.get("BCJR_GRAPH") != "1":
        soft, _ = bcjr_forward_backward(
            sequences, codebook, T_temp, preds=preds, succs=succs,
        )
        return soft

    key = (tuple(sequences.shape), sequences.dtype, T_temp.dtype)
    if key not in _GRAPH_CACHE:
        # Warmup + capture uses detached clones of the live inputs as samples.
        sample_seq = sequences.detach().clone().requires_grad_(
            sequences.requires_grad
        )
        sample_T = T_temp.detach().clone()
        _GRAPH_CACHE[key] = _make_graphed(
            sample_seq, codebook, sample_T, preds, succs,
        )

    return _GRAPH_CACHE[key](sequences, T_temp)


def reset_graph_cache():
    """Drop all captured graphs. Use between runs or if memory pressure."""
    global _GRAPH_CACHE
    _GRAPH_CACHE = {}
    torch.cuda.empty_cache()
