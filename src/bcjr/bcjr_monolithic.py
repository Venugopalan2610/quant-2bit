"""Monolithic BCJR: ONE torch.autograd.Function for the full forward-backward.

Collapses the entire BCJR algorithm (log_local + 128 alpha + 128 beta + marginals
+ soft codeword) into a single autograd node. Eliminates ~500 autograd-engine
overhead ops per chunk.

Forward:
  - Compute log_local inline
  - Run alpha recursion (Triton kernels, time-serialized)
  - Run beta recursion (Triton kernels)
  - Compute marginals + soft codeword
  - Save: log_alpha, log_beta, log_local, codebook, T_temp for backward

Backward (analytic gradient):
  - d_soft / d_P then d_P / d_log_P then d_log_P / d_alpha, d_beta, d_log_Z
  - d_log_Z backward contributes to d_alpha[..., -1, :]
  - Unroll alpha backward in reverse time — use existing Triton bwd kernel
  - Same for beta in forward time
  - d_log_local → d_sequences via the inverse of _local_log_prob

Triton kernels are still per-step (inherently sequential in time), but the
autograd overhead per step collapses to zero.
"""
import os

import torch

from src.bcjr.forward_backward import (
    L_BITS, K_BITS, V_DIM,
    build_pred_succ_tables, _local_log_prob,
)
from src.bcjr.bcjr_triton import (
    _alpha_step_fwd_kernel, _alpha_step_bwd_kernel, N_PRED,
)

import triton


def _run_alpha_forward(log_local, preds):
    """Alpha recursion with (N, B, S) buffer layout so every per-step tensor
    is a contiguous (B, S) view of the same allocation. Zero per-iteration
    allocations — required for CUDA graph capture and already faster on its own.
    """
    B, N, S = log_local.shape
    device = log_local.device

    # Transpose log_local once to (N, B, S) contiguous so log_local_nbs[t] is
    # a contiguous (B, S) view. Eliminates per-step .contiguous() allocation.
    log_local_nbs = log_local.transpose(0, 1).contiguous()       # (N, B, S)

    # Pre-allocate log_alpha in (N, B, S) layout so log_alpha_nbs[t] is a
    # contiguous (B, S) slice — the Triton kernel writes to it directly.
    log_alpha_nbs = torch.empty((N, B, S), device=device, dtype=torch.float32)

    # Initial log_alpha_prev = zeros (free, allocation cost negligible)
    zero_prev = torch.zeros((B, S), device=device, dtype=torch.float32)

    BLOCK_S = 128
    grid = (B, triton.cdiv(S, BLOCK_S))
    log_alpha_prev = zero_prev
    for t in range(N):
        out = log_alpha_nbs[t]                                    # (B, S) contiguous view
        _alpha_step_fwd_kernel[grid](
            log_alpha_prev, log_local_nbs[t], preds, out,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )
        log_alpha_prev = out

    # Return in (B, N, S) to match downstream consumers
    return log_alpha_nbs.transpose(0, 1).contiguous()


def _run_beta_forward(log_local, succs):
    """Beta recursion. Functional-style add (autograd-compatible). The
    autograd.Function path runs with grad-mode OFF inside ctx so the extra
    allocation is cheap; the pure-function path (under make_graphed_callables)
    uses its graph memory pool for bounded allocations.
    """
    B, N, S = log_local.shape
    device = log_local.device

    log_local_nbs = log_local.transpose(0, 1).contiguous()       # (N, B, S)
    log_beta_nbs = torch.empty((N, B, S), device=device, dtype=torch.float32)

    zero_beta = torch.zeros((B, S), device=device, dtype=torch.float32)
    log_beta_nbs[N - 1] = zero_beta
    zero_local = torch.zeros((B, S), device=device, dtype=torch.float32)

    BLOCK_S = 128
    grid = (B, triton.cdiv(S, BLOCK_S))
    log_beta_next = zero_beta
    for t in range(N - 2, -1, -1):
        x = log_local_nbs[t + 1] + log_beta_next
        out = log_beta_nbs[t]
        _alpha_step_fwd_kernel[grid](
            x, zero_local, succs, out,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )
        log_beta_next = out

    return log_beta_nbs.transpose(0, 1).contiguous()


def _backward_log_alpha(log_alpha, d_log_alpha, preds):
    """Propagate d_log_alpha back through alpha recursion."""
    B, N, S = log_alpha.shape
    device = log_alpha.device

    log_alpha_nbs = log_alpha.transpose(0, 1).contiguous()        # (N, B, S)
    d_log_alpha_nbs = d_log_alpha.transpose(0, 1).contiguous()    # (N, B, S)

    d_log_local_nbs = d_log_alpha_nbs.clone()                     # identity part

    BLOCK_S = 128
    grid = (B, triton.cdiv(S, BLOCK_S))
    grad_alpha_next = d_log_alpha_nbs[N - 1].clone()
    for t in range(N - 1, 0, -1):
        log_alpha_prev = log_alpha_nbs[t - 1]
        grad_prev = torch.zeros((B, S), device=device, dtype=torch.float32)
        _alpha_step_bwd_kernel[grid](
            log_alpha_prev, grad_alpha_next, preds, grad_prev,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )
        d_log_local_nbs[t - 1].add_(grad_prev)
        grad_alpha_next = d_log_alpha_nbs[t - 1] + grad_prev

    return d_log_local_nbs.transpose(0, 1).contiguous()


def _backward_log_beta(log_local, log_beta, d_log_beta, succs):
    """Propagate d_log_beta back through beta recursion."""
    B, N, S = log_beta.shape
    device = log_beta.device

    log_local_nbs = log_local.transpose(0, 1).contiguous()
    log_beta_nbs = log_beta.transpose(0, 1).contiguous()
    d_log_beta_nbs = d_log_beta.transpose(0, 1).contiguous()

    d_log_local_nbs = torch.zeros_like(log_local_nbs)

    BLOCK_S = 128
    grid = (B, triton.cdiv(S, BLOCK_S))
    grad_beta_prev = d_log_beta_nbs[0].clone()
    for t in range(0, N - 1):
        x = log_local_nbs[t + 1] + log_beta_nbs[t + 1]
        grad_x = torch.zeros((B, S), device=device, dtype=torch.float32)
        _alpha_step_bwd_kernel[grid](
            x, grad_beta_prev, succs, grad_x,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )
        d_log_local_nbs[t + 1].add_(grad_x)
        grad_beta_prev = d_log_beta_nbs[t + 1] + grad_x

    return d_log_local_nbs.transpose(0, 1).contiguous()


class BcjrForwardBackwardMonolithic(torch.autograd.Function):
    """One autograd node for the full BCJR forward-backward algorithm."""

    @staticmethod
    def forward(ctx, sequences, codebook, T_temp, preds, succs):
        assert sequences.dtype == torch.float32
        assert codebook.dtype == torch.float32

        # 1. log_local
        log_local = _local_log_prob(sequences, codebook, T_temp)        # (B, N, S)

        # 2. alpha recursion (Triton)
        log_alpha = _run_alpha_forward(log_local, preds)                # (B, N, S)

        # 3. beta recursion (Triton)
        log_beta = _run_beta_forward(log_local, succs)                  # (B, N, S)

        # 4. log_Z = logsumexp(log_alpha[:, -1], dim=-1)
        log_Z = torch.logsumexp(log_alpha[:, -1], dim=-1)               # (B,)

        # 5. marginal P = exp(alpha + beta - log_Z)
        log_P = log_alpha + log_beta - log_Z.view(-1, 1, 1)
        P = log_P.exp()                                                 # (B, N, S)

        # 6. soft codeword
        soft = torch.einsum("bns,sv->bnv", P, codebook)                 # (B, N, V)

        # Save for backward
        ctx.save_for_backward(
            sequences, codebook, T_temp, preds, succs,
            log_alpha, log_beta, log_Z, P,
        )
        return soft

    @staticmethod
    def backward(ctx, grad_soft):
        (sequences, codebook, T_temp, preds, succs,
         log_alpha, log_beta, log_Z, P) = ctx.saved_tensors
        B, N, S = log_alpha.shape

        # d_P from einsum: dL/dP[b, t, s] = sum_v grad_soft[b, t, v] * codebook[s, v]
        d_P = torch.einsum("bnv,sv->bns", grad_soft, codebook)           # (B, N, S)

        # d_log_P from P = exp(log_P): d_log_P = P * d_P
        d_log_P = P * d_P                                                # (B, N, S)

        # log_P = log_alpha + log_beta - log_Z.view(-1, 1, 1)
        d_log_alpha = d_log_P
        d_log_beta = d_log_P
        # d_log_Z = -sum over (n, s) of d_log_P[b, n, s]
        d_log_Z = -d_log_P.sum(dim=(1, 2))                               # (B,)

        # log_Z = logsumexp(log_alpha[:, -1], dim=-1)
        # d_log_Z back-prop: d_log_alpha[..., -1, :] += softmax(alpha[..., -1, :]) * d_log_Z
        last_alpha = log_alpha[:, -1]                                    # (B, S)
        # softmax: subtract log_Z then exp
        softmax_last = (last_alpha - log_Z.unsqueeze(-1)).exp()          # (B, S)
        d_log_alpha_last_add = softmax_last * d_log_Z.unsqueeze(-1)      # (B, S)
        d_log_alpha = d_log_alpha.clone()
        d_log_alpha[:, -1] = d_log_alpha[:, -1] + d_log_alpha_last_add

        # Propagate d_log_alpha through alpha recursion → d_log_local_alpha
        d_log_local_alpha = _backward_log_alpha(log_alpha, d_log_alpha, preds)

        # Propagate d_log_beta through beta recursion → d_log_local_beta
        # Need log_local for this — recompute (cheap, and we didn't save it)
        log_local = _local_log_prob(sequences, codebook, T_temp)
        d_log_local_beta = _backward_log_beta(log_local, log_beta, d_log_beta, succs)

        d_log_local = d_log_local_alpha + d_log_local_beta               # (B, N, S)

        # Propagate d_log_local through _local_log_prob to get d_sequences.
        # _local_log_prob: log_local[b, n, s] = -( |cb[s]|^2 - 2*<cb[s], w[b,n]> + |w[b,n]|^2 ) / T
        # d log_local / d w[b, n, v] = (-1/T) * (-2 * cb[s, v] + 2 * w[b, n, v])
        # d_w[b, n, v] = sum_s d_log_local[b, n, s] * (-1/T) * (-2*cb[s,v] + 2*w[b,n,v])
        inv_T = 1.0 / T_temp
        # d_w from cb term: sum_s d_log_local * (2/T) * cb[s, v]
        d_w_cb = (2.0 * inv_T) * torch.einsum("bns,sv->bnv", d_log_local, codebook)
        # d_w from w^2 term: sum_s d_log_local * (-2/T) * w[b, n, v]
        #   = (-2/T) * w[b, n, v] * sum_s d_log_local[b, n, s]
        d_log_local_sum = d_log_local.sum(dim=-1, keepdim=True)          # (B, N, 1)
        d_w_wsq = (-2.0 * inv_T) * sequences * d_log_local_sum
        d_sequences = d_w_cb + d_w_wsq

        # codebook and T_temp aren't differentiable in our setup
        # preds, succs are int, non-differentiable
        return d_sequences, None, None, None, None


def bcjr_forward_backward_monolithic(sequences, codebook, T_temp,
                                     L=L_BITS, k=K_BITS, V=V_DIM,
                                     preds=None, succs=None,
                                     return_internals=False):
    """Drop-in replacement for bcjr_forward_backward. Uses one big autograd
    node + Triton step kernels. Collapses ~500 autograd-op overhead per chunk.
    """
    if preds is None or succs is None:
        preds, succs = build_pred_succ_tables(device=sequences.device)
    if not torch.is_tensor(T_temp):
        T_temp = torch.tensor(float(T_temp), dtype=torch.float32,
                              device=sequences.device)
    soft = BcjrForwardBackwardMonolithic.apply(
        sequences.float(), codebook.float(), T_temp.float(), preds, succs,
    )
    # Match reference signature: (soft, log_Z). log_Z not needed downstream.
    return soft, None
