"""Parity + timing for CUDA-graph-captured monolithic BCJR.

Compares:
  (A) reference bcjr_forward_backward (Triton step kernels, eager loop)
  (B) monolithic (one autograd node, no graph)
  (C) monolithic + graph capture

Expects math parity across all three within fp32 tolerance.
"""
import os
import sys
import time

import torch

os.environ["BCJR_TRITON"] = "1"

from src.bcjr.forward_backward import (
    L_BITS, V_DIM, build_pred_succ_tables, bcjr_forward_backward,
)
from src.bcjr.bcjr_monolithic import bcjr_forward_backward_monolithic


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    torch.manual_seed(0)
    B, N, V = 2, 8, V_DIM
    S = 1 << L_BITS
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    preds, succs = build_pred_succ_tables(device="cuda")

    seq_ref = torch.randn(B, N, V, device="cuda", dtype=torch.float32,
                          requires_grad=True)
    seq_mono = seq_ref.detach().clone().requires_grad_(True)
    seq_graph = seq_ref.detach().clone().requires_grad_(True)

    # (A) reference
    soft_ref, _ = bcjr_forward_backward(seq_ref, codebook, T_temp,
                                        preds=preds, succs=succs)

    # (B) monolithic
    soft_mono, _ = bcjr_forward_backward_monolithic(seq_mono, codebook, T_temp,
                                                    preds=preds, succs=succs)

    # (C) graphed monolithic
    os.environ["BCJR_MONO_GRAPH"] = "1"
    from src.bcjr.bcjr_monolithic_graph import (
        bcjr_forward_backward_monolithic_graphed, reset_mono_graph_cache,
    )
    reset_mono_graph_cache()
    soft_graph, _ = bcjr_forward_backward_monolithic_graphed(
        seq_graph, codebook, T_temp, preds=preds, succs=succs,
    )

    # Parity
    def check_parity(name, ref, new):
        diff = (ref - new).abs().max().item()
        rel = diff / ref.abs().max().clamp(min=1e-30).item()
        print(f"{name:<25} forward  diff={diff:.2e}  rel={rel:.2e}", flush=True)
        assert rel < 1e-3, f"{name} forward parity fail: {rel:.2e}"

    check_parity("monolithic vs ref", soft_ref, soft_mono)
    check_parity("graphed vs ref",    soft_ref, soft_graph)

    # Backward parity
    grad_out = torch.randn_like(soft_ref)
    soft_ref.backward(grad_out)
    soft_mono.backward(grad_out)
    soft_graph.backward(grad_out)

    def check_bwd(name, ref_grad, new_grad):
        diff = (ref_grad - new_grad).abs().max().item()
        rel = diff / ref_grad.abs().max().clamp(min=1e-30).item()
        print(f"{name:<25} backward diff={diff:.2e}  rel={rel:.2e}", flush=True)
        assert rel < 1e-3, f"{name} backward parity fail: {rel:.2e}"

    check_bwd("monolithic vs ref", seq_ref.grad, seq_mono.grad)
    check_bwd("graphed vs ref",    seq_ref.grad, seq_graph.grad)

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
        print(f"  {label:<30} {dt:.0f} ms", flush=True)
        return dt

    reset_mono_graph_cache()

    dt_ref = time_fn(bcjr_forward_backward, "reference (eager Triton loop)")
    dt_mono = time_fn(bcjr_forward_backward_monolithic, "monolithic (no graph)")

    # Reset cache before timing graphed — the first call captures, subsequent replay
    reset_mono_graph_cache()
    dt_graph = time_fn(
        bcjr_forward_backward_monolithic_graphed, "monolithic + graph capture"
    )

    print(f"\n  speedup mono vs ref:  {dt_ref/dt_mono:.2f}×")
    print(f"  speedup graph vs ref: {dt_ref/dt_graph:.2f}×")
    print(f"  speedup graph vs mono:{dt_mono/dt_graph:.2f}×")

    print("\nPASS: parity + timing OK")


if __name__ == "__main__":
    main()
