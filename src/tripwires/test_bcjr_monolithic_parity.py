"""Parity test: monolithic BCJR (single autograd node) vs reference.

Verifies forward + backward match. Also times both for a real chunk size.
"""
import os
import sys
import time

import torch

os.environ["BCJR_TRITON"] = "1"  # reference uses Triton kernels

from src.bcjr.forward_backward import (
    L_BITS, V_DIM, build_pred_succ_tables, bcjr_forward_backward,
)
from src.bcjr.bcjr_monolithic import bcjr_forward_backward_monolithic


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    torch.manual_seed(0)
    # Parity on a tiny input first
    B, N, V = 2, 8, V_DIM
    S = 1 << L_BITS
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    preds, succs = build_pred_succ_tables(device="cuda")

    seq_ref = torch.randn(B, N, V, device="cuda", dtype=torch.float32,
                          requires_grad=True)
    seq_mono = seq_ref.detach().clone().requires_grad_(True)

    soft_ref, _ = bcjr_forward_backward(seq_ref, codebook, T_temp,
                                        preds=preds, succs=succs)
    soft_mono, _ = bcjr_forward_backward_monolithic(seq_mono, codebook, T_temp,
                                                    preds=preds, succs=succs)

    fwd_diff = (soft_ref - soft_mono).abs().max().item()
    fwd_rel = fwd_diff / soft_ref.abs().max().clamp(min=1e-30).item()
    print(f"forward  diff={fwd_diff:.2e}  rel={fwd_rel:.2e}", flush=True)
    assert fwd_rel < 1e-3, f"forward parity failed: {fwd_rel:.2e}"

    # Backward: compare d_sequences
    grad_out = torch.randn_like(soft_ref)
    soft_ref.backward(grad_out)
    soft_mono.backward(grad_out)

    bwd_diff = (seq_ref.grad - seq_mono.grad).abs().max().item()
    bwd_rel = bwd_diff / seq_ref.grad.abs().max().clamp(min=1e-30).item()
    print(f"backward diff={bwd_diff:.2e}  rel={bwd_rel:.2e}", flush=True)
    assert bwd_rel < 1e-3, f"backward parity failed: {bwd_rel:.2e}"

    # Timing at real chunk size
    print(f"\nTiming at B=16, N=128:")
    B_real, N_real = 16, 128
    sequences = torch.randn(B_real, N_real, V, device="cuda", dtype=torch.float32)

    def time_fn(fn, label, warmup=3, iters=3):
        for _ in range(warmup):
            seq = sequences.clone().requires_grad_(True)
            out, _ = fn(seq, codebook, T_temp, preds=preds, succs=succs)
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            seq = sequences.clone().requires_grad_(True)
            out, _ = fn(seq, codebook, T_temp, preds=preds, succs=succs)
            out.sum().backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters * 1000
        print(f"  {label:<25} {dt:.0f} ms", flush=True)
        return dt

    dt_ref = time_fn(bcjr_forward_backward, "reference (Triton steps)")
    dt_mono = time_fn(bcjr_forward_backward_monolithic, "monolithic (one autograd)")
    print(f"  speedup: {dt_ref/dt_mono:.2f}×")

    print("\nPASS: monolithic parity + timing OK")


if __name__ == "__main__":
    main()
