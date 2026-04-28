"""Triton-fused BCJR alpha/beta step kernels.

Replaces the eager-mode python ops:
    gathered = log_alpha_prev[:, preds]        # O(B*S*16) memory allocation
    log_sum  = torch.logsumexp(gathered, -1)   # separate kernel
    return log_local_t + log_sum               # separate kernel

with a single fused Triton kernel per step. Eliminates:
  - Intermediate `gathered` tensor (64 MB per call, not allocated)
  - Two extra kernel launches per step
  - Python dispatch overhead between the three ops

Forward: alpha_step_triton_fwd
Backward: alpha_step_triton_bwd (via BackwardStep class)

For B=16, S=65536, n_pred=16:
  - Each program processes a tile of (1 batch, BLOCK_S states)
  - Total programs = B × (S / BLOCK_S)
  - BLOCK_S chosen for SM occupancy

Numerics: max-subtraction logsumexp trick for numerical stability.
"""
import torch
import triton
import triton.language as tl


# n_pred is fixed by the trellis topology: 16 predecessors per state
N_PRED = 16


@triton.jit
def _alpha_step_fwd_kernel(
    log_alpha_prev_ptr,    # (B, S) fp32
    log_local_t_ptr,       # (B, S) fp32
    preds_ptr,             # (S, N_PRED) int64
    out_ptr,               # (B, S) fp32 — log_alpha_t
    B: tl.constexpr,
    S: tl.constexpr,
    N_PRED_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Forward alpha step. Does NOT save softmax for backward — backward
    recomputes it from log_alpha_prev + preds (cheap) to avoid storing
    (B, S, N_PRED) = 64 MB per step × 128 steps = 8 GB of intermediates.
    """
    pid_b = tl.program_id(0)
    pid_s_block = tl.program_id(1)

    s_start = pid_s_block * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    pred_ptrs = preds_ptr + s_offs[:, None] * N_PRED_C + tl.arange(0, N_PRED_C)[None, :]
    preds = tl.load(pred_ptrs, mask=s_mask[:, None], other=0)

    alpha_base = log_alpha_prev_ptr + pid_b * S
    gather_ptrs = alpha_base + preds
    gathered = tl.load(gather_ptrs, mask=s_mask[:, None], other=-1e30)

    row_max = tl.max(gathered, axis=1)
    shifted = gathered - row_max[:, None]
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=1)
    log_sum = row_max + tl.log(sum_exp)

    local_ptrs = log_local_t_ptr + pid_b * S + s_offs
    log_local_t = tl.load(local_ptrs, mask=s_mask, other=0.0)
    log_alpha_t = log_local_t + log_sum

    out_ptrs = out_ptr + pid_b * S + s_offs
    tl.store(out_ptrs, log_alpha_t, mask=s_mask)


@triton.jit
def _alpha_step_bwd_kernel(
    log_alpha_prev_ptr,    # (B, S) fp32 — saved input (small, ~4MB per step)
    grad_alpha_ptr,        # (B, S) fp32 — dL/d_alpha_t
    preds_ptr,             # (S, N_PRED) int64
    grad_alpha_prev_ptr,   # (B, S) fp32 — accumulator (atomic_add)
    B: tl.constexpr,
    S: tl.constexpr,
    N_PRED_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Backward re-computes softmax on the fly from log_alpha_prev + preds.
    This saves ~64 MB per step (vs saving softmax in forward), trading a
    tiny amount of compute for a huge memory win. N=128 steps × 64 MB
    saved = 8 GB not consumed.
    """
    pid_b = tl.program_id(0)
    pid_s_block = tl.program_id(1)

    s_start = pid_s_block * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # Load predecessors and gather log_alpha_prev[b, preds[s, :]]
    pred_ptrs = preds_ptr + s_offs[:, None] * N_PRED_C + tl.arange(0, N_PRED_C)[None, :]
    preds = tl.load(pred_ptrs, mask=s_mask[:, None], other=0)
    alpha_base = log_alpha_prev_ptr + pid_b * S
    gather_ptrs = alpha_base + preds
    gathered = tl.load(gather_ptrs, mask=s_mask[:, None], other=-1e30)

    # Recompute softmax = exp(gathered - logsumexp(gathered))
    row_max = tl.max(gathered, axis=1)
    shifted = gathered - row_max[:, None]
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=1)
    softmax = exp_shifted / sum_exp[:, None]                    # (BLOCK_S, N_PRED)

    # d_gathered[s, i] = grad_a[s] * softmax[s, i]
    grad_a_ptrs = grad_alpha_ptr + pid_b * S + s_offs
    grad_a = tl.load(grad_a_ptrs, mask=s_mask, other=0.0)
    d_gathered = grad_a[:, None] * softmax

    # Scatter-add to grad_alpha_prev[pid_b, preds[s, i]]
    scatter_ptrs = grad_alpha_prev_ptr + pid_b * S + preds
    tl.atomic_add(scatter_ptrs, d_gathered, mask=s_mask[:, None])


class _AlphaStepTriton(torch.autograd.Function):
    """Forward: log_alpha_t = log_local_t + logsumexp(log_alpha_prev[:, preds], dim=-1).
    Backward: recomputes softmax from saved log_alpha_prev (small, 4 MB/step)
    instead of stashing softmax (would be 64 MB/step × 128 = 8 GB).
    """

    @staticmethod
    def forward(ctx, log_alpha_prev, log_local_t, preds):
        assert log_alpha_prev.is_cuda and log_local_t.is_cuda and preds.is_cuda
        assert log_alpha_prev.dtype == torch.float32
        assert log_local_t.dtype == torch.float32
        assert preds.dtype == torch.int64
        # Kernel reads with stride (S, 1); caller may pass views (e.g., log_local[:, t])
        # which are non-contiguous. Force contiguous to avoid reading wrong memory.
        log_alpha_prev = log_alpha_prev.contiguous()
        log_local_t = log_local_t.contiguous()
        B, S = log_alpha_prev.shape
        assert log_local_t.shape == (B, S)
        assert preds.shape[0] == S and preds.shape[1] == N_PRED

        out = torch.empty((B, S), device=log_alpha_prev.device, dtype=torch.float32)

        BLOCK_S = 128
        grid = (B, triton.cdiv(S, BLOCK_S))
        _alpha_step_fwd_kernel[grid](
            log_alpha_prev, log_local_t, preds, out,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )

        ctx.save_for_backward(log_alpha_prev, preds)
        ctx.alpha_prev_shape = log_alpha_prev.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        log_alpha_prev, preds = ctx.saved_tensors
        B, S = ctx.alpha_prev_shape

        # Identity grad through the add
        grad_log_local_t = grad_out.contiguous()

        # Scatter-add via Triton kernel (recomputes softmax internally)
        grad_log_alpha_prev = torch.zeros(
            (B, S), device=grad_out.device, dtype=torch.float32
        )

        BLOCK_S = 128
        grid = (B, triton.cdiv(S, BLOCK_S))
        _alpha_step_bwd_kernel[grid](
            log_alpha_prev, grad_out.contiguous(), preds, grad_log_alpha_prev,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )

        return grad_log_alpha_prev, grad_log_local_t, None


# Public API matching the eager _alpha_step signature.
def alpha_step_triton(log_alpha_prev, log_local_t, preds):
    """Drop-in Triton replacement for src.bcjr.forward_backward._alpha_step."""
    return _AlphaStepTriton.apply(log_alpha_prev, log_local_t, preds)


# Beta step is symmetric to alpha:
#   log_beta_t[s] = logsumexp_{s' in succs(s)}(log_local_{t+1}[s'] + log_beta_{t+1}[s'])
# Same kernel structure, but the "gather" combines log_local AND log_beta at
# successor indices. We reuse the alpha kernels by pre-computing the sum
# (log_local_next + log_beta_next) and then applying the same gather+logsumexp.


class _BetaStepTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_beta_next, log_local_next, succs):
        """log_beta_t[s] = logsumexp(gather(log_local_next + log_beta_next, succs[s, :])).
        Saves (log_local_next + log_beta_next) for backward; recomputes softmax.
        """
        assert log_beta_next.is_cuda and log_local_next.is_cuda and succs.is_cuda
        log_beta_next = log_beta_next.contiguous()
        log_local_next = log_local_next.contiguous()
        B, S = log_beta_next.shape

        x = (log_local_next + log_beta_next).contiguous()
        zero_local = torch.zeros((B, S), device=log_beta_next.device,
                                 dtype=torch.float32)

        out = torch.empty((B, S), device=log_beta_next.device, dtype=torch.float32)

        BLOCK_S = 128
        grid = (B, triton.cdiv(S, BLOCK_S))
        _alpha_step_fwd_kernel[grid](
            x, zero_local, succs, out,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )

        # Save combined x (4 MB per step) instead of softmax (64 MB).
        ctx.save_for_backward(x, succs)
        ctx.input_shape = log_beta_next.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, succs = ctx.saved_tensors
        B, S = ctx.input_shape

        grad_x = torch.zeros((B, S), device=grad_out.device, dtype=torch.float32)

        BLOCK_S = 128
        grid = (B, triton.cdiv(S, BLOCK_S))
        _alpha_step_bwd_kernel[grid](
            x, grad_out.contiguous(), succs, grad_x,
            B=B, S=S, N_PRED_C=N_PRED, BLOCK_S=BLOCK_S,
        )

        # d log_local_next = d log_beta_next = grad_x (sum splits identity)
        return grad_x, grad_x, None


def beta_step_triton(log_beta_next, log_local_next, succs):
    """Drop-in Triton replacement for src.bcjr.forward_backward._beta_step."""
    return _BetaStepTriton.apply(log_beta_next, log_local_next, succs)
