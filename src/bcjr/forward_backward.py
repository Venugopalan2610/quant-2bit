"""BCJR forward-backward on the HYB bitshift trellis (QTIP parameters).

Trellis setup (matches src/viterbi/encode.py):
  L = 16  state width in bits  →  S = 2^L = 65536 states
  k = 2   bits emitted per step
  V = 2   vector dimension  →  codebook: (S, V)
  kV = 4  so each state has 2^kV = 16 predecessors / 16 successors
  N       number of steps per tile = T_seq / V (e.g. 128 when T_seq=256)

Trellis transition rule:
  next_state = ((prev_state << kV) | c) & mask,   c in [0, 2^kV)
Equivalently, for a destination state s:
  predecessors preds(s) = { (p_top << (L - kV)) | (s >> kV) : p_top in [0, 2^kV) }
  successors   succs(s) = { ((s << kV) | c)        & mask   : c     in [0, 2^kV) }

Emission model (matches Viterbi local_cost):
  Each destination state s emits the V-dim codeword codebook[s].
  Local energy at step t for state s:
      E_t(s) = || codebook[s] - w_t ||^2         (w_t is V-dim)
  Local probability in the Boltzmann measure:
      p_t(s)  ∝ exp( -E_t(s) / T )

The chain's joint is
  P(s_0, ..., s_{N-1}, w) ∝ prod_t p_t(s_t) * 1[ transition_ok(s_{t-1}, s_t) ]
with start state unconstrained (matches Viterbi's zero-initialized cum_err)
and end state unconstrained (matches Viterbi's argmin over all final states).

Forward-backward in log space:
  log_alpha_t(s) = log_local_t(s) + logsumexp_{s' in preds(s)} log_alpha_{t-1}(s')
  log_beta_t(s)  = logsumexp_{s' in succs(s)} [ log_local_{t+1}(s') + log_beta_{t+1}(s') ]
  log_Z          = logsumexp_s log_alpha_{N-1}(s)
  log_P_t(s)     = log_alpha_t(s) + log_beta_t(s) - log_Z

Soft codeword (expected emission):
  W_soft_t = sum_s P_t(s) * codebook[s]     shape (B, N, V)

At T -> 0:
  P_t concentrates on the Viterbi-winning state → W_soft_t → Viterbi recon.
At T -> infinity:
  P_t uniform over reachable states → W_soft_t → mean of codebook.

Gradient: W_soft_t is a smooth function of w_t (via local_log_prob). Autograd
handles the backward pass. Memory for autograd is dominated by log_alpha and
log_beta storage: O(B * N * S). For OLMoE q_proj, process tiles in chunks.
"""
import math
import os

import torch
import torch.utils.checkpoint as checkpoint


L_BITS = 16
K_BITS = 2
V_DIM = 2


def _use_triton():
    return os.environ.get("BCJR_TRITON") == "1"


def build_pred_succ_tables(L=L_BITS, k=K_BITS, V=V_DIM, device="cuda"):
    """Return (preds, succs) index tensors on `device`.

    preds: (S, n_pred) int64 — preds[s, p] is the p-th predecessor of state s
    succs: (S, n_succ) int64 — succs[s, c] is the c-th successor of state s

    Both have n_pred = n_succ = 2^(kV).
    """
    S = 1 << L
    kV = k * V
    n_branches = 1 << kV
    mask_L = (1 << L) - 1

    s_arr = torch.arange(S, dtype=torch.int64, device=device)
    c_arr = torch.arange(n_branches, dtype=torch.int64, device=device)

    # preds[s, p] = (p << (L - kV)) | (s >> kV)
    top_of_s = s_arr >> kV                                       # (S,)
    preds = (c_arr.unsqueeze(0) << (L - kV)) | top_of_s.unsqueeze(1)

    # succs[s, c] = ((s << kV) | c) & mask_L
    shifted = (s_arr.unsqueeze(1) << kV) & mask_L                # (S, 1)
    succs = shifted | c_arr.unsqueeze(0)                         # (S, n_branches)

    return preds, succs


def _local_log_prob(sequences, codebook, T_temp):
    """Compute log_local_t[b, s] = -||codebook[s] - w_{b,t}||^2 / T.

    sequences: (B, N, V)
    codebook:  (S, V)
    T_temp:    scalar tensor (broadcastable) — temperature > 0

    Returns: (B, N, S) float32
    """
    # ||cb[s] - w||^2 = ||cb[s]||^2 - 2*<cb[s], w> + ||w||^2
    cb_sq = (codebook * codebook).sum(dim=-1)                    # (S,)
    w_dot_cb = torch.einsum("bnv,sv->bns", sequences, codebook)  # (B, N, S)
    w_sq = (sequences * sequences).sum(dim=-1, keepdim=True)     # (B, N, 1)
    local_cost = cb_sq.view(1, 1, -1) - 2.0 * w_dot_cb + w_sq    # (B, N, S)
    return -local_cost / T_temp


def _alpha_step(log_alpha_prev, log_local_t, preds):
    """One step of forward recursion. Extracted so per-step checkpointing can
    re-run just this block during backward, freeing the (B, S, n_pred) gather
    tensor between steps instead of holding all N of them live."""
    gathered = log_alpha_prev[:, preds]                          # (B, S, n_pred)
    log_sum = torch.logsumexp(gathered, dim=-1)                  # (B, S)
    return log_local_t + log_sum


def _beta_step(log_beta_next, log_local_next, succs):
    """One step of backward recursion (see _alpha_step)."""
    succ_local = log_local_next[:, succs]                        # (B, S, n_succ)
    succ_beta  = log_beta_next[:, succs]                         # (B, S, n_succ)
    return torch.logsumexp(succ_local + succ_beta, dim=-1)       # (B, S)


def _forward_log_alpha(log_local, preds, use_checkpoint=True):
    """Run forward recursion. Returns log_alpha of shape (B, N, S).

    log_alpha_t[s] = log_local_t[s] + logsumexp_{p in preds(s)} log_alpha_{t-1}[p]
    with log_alpha_{-1}[s] = 0 (uniform — matches Viterbi's free start).

    If BCJR_TRITON=1, uses the fused Triton kernel (src/bcjr/bcjr_triton.py)
    for each step — ~6× faster than eager. The per-step checkpoint is bypassed
    in this mode because the kernel already fuses gather+logsumexp+add, so
    the memory/compute trade-off from checkpoint is obsolete.
    """
    B, N, S = log_local.shape
    use_triton = _use_triton()
    if use_triton:
        from src.bcjr.bcjr_triton import alpha_step_triton
        alpha_step_fn = alpha_step_triton
        do_ckpt = False
    else:
        alpha_step_fn = _alpha_step
        do_ckpt = use_checkpoint and log_local.requires_grad

    log_alpha_prev = torch.zeros((B, S), dtype=log_local.dtype, device=log_local.device)
    outs = []
    for t in range(N):
        if do_ckpt:
            log_alpha_prev = checkpoint.checkpoint(
                alpha_step_fn, log_alpha_prev, log_local[:, t], preds,
                use_reentrant=False,
            )
        else:
            log_alpha_prev = alpha_step_fn(log_alpha_prev, log_local[:, t], preds)
        outs.append(log_alpha_prev)
    return torch.stack(outs, dim=1)                              # (B, N, S)


def _backward_log_beta(log_local, succs, use_checkpoint=True):
    """Run backward recursion. Returns log_beta of shape (B, N, S).

    log_beta_t[s] = logsumexp_{s' in succs(s)} [ log_local_{t+1}[s'] + log_beta_{t+1}[s'] ]
    with log_beta_{N-1}[s] = 0 (uniform — matches Viterbi's free end).

    BCJR_TRITON=1 routes to the Triton kernel — same speedup as alpha.
    """
    B, N, S = log_local.shape
    use_triton = _use_triton()
    if use_triton:
        from src.bcjr.bcjr_triton import beta_step_triton
        beta_step_fn = beta_step_triton
        do_ckpt = False
    else:
        beta_step_fn = _beta_step
        do_ckpt = use_checkpoint and log_local.requires_grad

    log_beta_next = torch.zeros((B, S), dtype=log_local.dtype, device=log_local.device)
    outs_rev = [log_beta_next]
    for t in range(N - 2, -1, -1):
        if do_ckpt:
            log_beta_next = checkpoint.checkpoint(
                beta_step_fn, log_beta_next, log_local[:, t + 1], succs,
                use_reentrant=False,
            )
        else:
            log_beta_next = beta_step_fn(log_beta_next, log_local[:, t + 1], succs)
        outs_rev.append(log_beta_next)
    # outs_rev is [beta_{N-1}, beta_{N-2}, ..., beta_0] → reverse → (B, N, S)
    return torch.stack(outs_rev[::-1], dim=1)


def bcjr_forward_backward(sequences, codebook, T_temp,
                          L=L_BITS, k=K_BITS, V=V_DIM,
                          preds=None, succs=None, return_internals=False):
    """Full BCJR forward-backward.

    sequences: (B, N, V) float32/float, requires_grad allowed.
    codebook:  (S, V) float32 (no grad through codebook).
    T_temp:    scalar temperature > 0 (python float or 0-d tensor).

    Returns:
      soft_codewords: (B, N, V) — E_P[codebook[s_t]]
      log_Z:          (B,)
      marginals_P:    (B, N, S) — P(s_t = s | obs, T)  [only if return_internals]
      log_alpha:      (B, N, S)                       [only if return_internals]
      log_beta:       (B, N, S)                       [only if return_internals]
    """
    assert sequences.dim() == 3 and sequences.shape[-1] == V
    assert codebook.shape == (1 << L, V)
    device = sequences.device
    if preds is None or succs is None:
        preds, succs = build_pred_succ_tables(L=L, k=k, V=V, device=device)

    if not torch.is_tensor(T_temp):
        T_temp = torch.tensor(float(T_temp), dtype=sequences.dtype, device=device)

    log_local = _local_log_prob(sequences, codebook, T_temp)     # (B, N, S)
    log_alpha = _forward_log_alpha(log_local, preds)             # (B, N, S)
    log_beta  = _backward_log_beta(log_local, succs)             # (B, N, S)

    # log_Z = logsumexp_s log_alpha_{N-1}[s]  (and beta_{N-1}=0 drops out)
    log_Z = torch.logsumexp(log_alpha[:, -1], dim=-1)            # (B,)

    # log_marginals[b, t, s] = log_alpha[b, t, s] + log_beta[b, t, s] - log_Z[b]
    log_P = log_alpha + log_beta - log_Z.view(-1, 1, 1)          # (B, N, S)
    P = log_P.exp()                                              # (B, N, S)

    # soft codeword: expected emission under P
    # W_soft[b, t, v] = sum_s P[b, t, s] * codebook[s, v]
    soft = torch.einsum("bns,sv->bnv", P, codebook)              # (B, N, V)

    if return_internals:
        return soft, log_Z, P, log_alpha, log_beta
    return soft, log_Z
