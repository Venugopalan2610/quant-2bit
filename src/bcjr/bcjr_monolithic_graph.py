"""CUDA-graph-captured monolithic BCJR.

Wraps the monolithic forward+backward with torch.cuda.make_graphed_callables.
This captures the entire Python time loop (256+ kernel launches) into a
replayable CUDA graph, eliminating per-step Python + CUDA dispatch overhead.

Gated by BCJR_MONO_GRAPH=1 env var. Falls back to eager monolithic otherwise.
"""
import os

import torch

from src.bcjr.bcjr_monolithic import (
    _run_alpha_forward,
    _run_beta_forward,
    _backward_log_alpha,
    _backward_log_beta,
)
from src.bcjr.forward_backward import _local_log_prob, build_pred_succ_tables


def _bcjr_pure_forward(sequences, codebook, T_temp, preds, succs):
    """Pure-function version of the monolithic forward. No autograd.Function;
    autograd is built by PyTorch from the individual ops. Used by
    make_graphed_callables which captures forward + backward graphs.
    """
    log_local = _local_log_prob(sequences, codebook, T_temp)       # (B, N, S)
    log_alpha = _run_alpha_forward(log_local, preds)               # (B, N, S)
    log_beta = _run_beta_forward(log_local, succs)                 # (B, N, S)
    log_Z = torch.logsumexp(log_alpha[:, -1], dim=-1)              # (B,)
    log_P = log_alpha + log_beta - log_Z.view(-1, 1, 1)
    P = log_P.exp()
    soft = torch.einsum("bns,sv->bnv", P, codebook)
    return soft


# Cache of graphed callables per (shape, dtype) key
_GRAPH_CACHE = {}


def bcjr_forward_backward_monolithic_graphed(sequences, codebook, T_temp,
                                             preds=None, succs=None,
                                             return_internals=False):
    """Drop-in for bcjr_forward_backward. Uses CUDA-graph-captured monolithic
    if BCJR_MONO_GRAPH=1, else falls back to the eager monolithic.
    """
    if preds is None or succs is None:
        preds, succs = build_pred_succ_tables(device=sequences.device)
    if not torch.is_tensor(T_temp):
        T_temp = torch.tensor(float(T_temp), dtype=torch.float32,
                              device=sequences.device)

    if os.environ.get("BCJR_MONO_GRAPH") != "1":
        # Fallback: regular monolithic call
        soft = _bcjr_pure_forward(sequences.float(), codebook.float(),
                                  T_temp.float(), preds, succs)
        return soft, None

    # Graph-captured path.
    # Key includes shape/dtype — if they change, build a new graph.
    key = (tuple(sequences.shape), sequences.dtype, T_temp.dtype,
           codebook.shape, codebook.dtype)

    if key not in _GRAPH_CACHE:
        # Build sample inputs matching live shape/dtype; capture graph.
        sample_seq = sequences.detach().clone().float().requires_grad_(True)
        sample_cb = codebook.detach().clone().float()
        sample_T = T_temp.detach().clone().float()

        # make_graphed_callables warms up and captures both fwd + bwd graphs.
        # preds/succs are closure constants.
        def core_fn(seq, cb, T):
            return _bcjr_pure_forward(seq, cb, T, preds, succs)

        _GRAPH_CACHE[key] = torch.cuda.make_graphed_callables(
            core_fn, sample_args=(sample_seq, sample_cb, sample_T),
        )

    graphed_fn = _GRAPH_CACHE[key]
    soft = graphed_fn(sequences.float(), codebook.float(), T_temp.float())
    return soft, None


def reset_mono_graph_cache():
    global _GRAPH_CACHE
    _GRAPH_CACHE = {}
    torch.cuda.empty_cache()
