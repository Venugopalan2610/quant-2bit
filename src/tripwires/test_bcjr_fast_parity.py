"""Numerical parity: bcjr_forward_backward (reference) vs _fast version.

Runs both on the same synthetic (sequences, codebook) and checks that:
  - soft codewords match within fp32 tolerance
  - backward gradients on sequences match within fp32 tolerance

Kept tiny (B=4, N=16) so it runs in ~5 s on a 4080.
"""
import sys

import torch

from src.bcjr.forward_backward import (
    L_BITS, V_DIM, build_pred_succ_tables, bcjr_forward_backward,
)
from src.bcjr.forward_backward_fast import bcjr_forward_backward_fast


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    torch.manual_seed(0)
    B, N, V = 4, 16, V_DIM
    S = 1 << L_BITS

    # Synthetic "sequences" in the range the quantizer would see (normalized).
    seq_ref = torch.randn(B, N, V, device="cuda", requires_grad=True)
    seq_fst = seq_ref.detach().clone().requires_grad_(True)

    codebook = torch.randn(S, V, device="cuda")
    T = 1.0
    preds, succs = build_pred_succ_tables(device="cuda")

    # ---- forward parity ----
    soft_ref, _ = bcjr_forward_backward(seq_ref, codebook, T,
                                        preds=preds, succs=succs)
    soft_fst, _ = bcjr_forward_backward_fast(seq_fst, codebook, T,
                                             preds=preds, succs=succs)
    fwd_rel = (soft_ref - soft_fst).abs().max() / soft_ref.abs().max().clamp(min=1e-30)
    print(f"forward max abs diff: {(soft_ref - soft_fst).abs().max().item():.2e}  "
          f"rel: {fwd_rel.item():.2e}")
    assert fwd_rel < 1e-3, f"forward parity failed: {fwd_rel.item():.2e}"

    # ---- backward parity ----
    # Project both to a scalar and backprop; compare gradients on seq.
    g_ref = soft_ref.sum()
    g_fst = soft_fst.sum()
    g_ref.backward()
    g_fst.backward()

    bwd_rel = ((seq_ref.grad - seq_fst.grad).abs().max()
               / seq_ref.grad.abs().max().clamp(min=1e-30))
    print(f"backward max abs diff: "
          f"{(seq_ref.grad - seq_fst.grad).abs().max().item():.2e}  "
          f"rel: {bwd_rel.item():.2e}")
    assert bwd_rel < 1e-3, f"backward parity failed: {bwd_rel.item():.2e}"

    # ---- timing spot-check (second call to let compile cache warm up) ----
    import time
    _ = bcjr_forward_backward_fast(seq_fst.detach(), codebook, T,
                                   preds=preds, succs=succs)  # warm
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = bcjr_forward_backward(seq_ref.detach(), codebook, T,
                                  preds=preds, succs=succs)
    torch.cuda.synchronize()
    dt_ref = (time.time() - t0) / 5

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = bcjr_forward_backward_fast(seq_fst.detach(), codebook, T,
                                       preds=preds, succs=succs)
    torch.cuda.synchronize()
    dt_fst = (time.time() - t0) / 5
    print(f"reference avg: {dt_ref*1000:.1f} ms  fast avg: {dt_fst*1000:.1f} ms  "
          f"speedup: {dt_ref/dt_fst:.2f}×")

    print("\nPASS: numerical parity + timing OK")


if __name__ == "__main__":
    main()
