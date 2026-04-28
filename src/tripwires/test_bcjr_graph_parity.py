"""Parity + timing for CUDA graph BCJR vs reference.

Tiny input (B=4, N=16, V=2) to run fast. Verifies:
  1. Graph-captured forward matches reference forward (fp32 tolerance)
  2. Graph-captured backward matches reference backward
  3. Replay speedup vs reference on this input size

If parity passes, BCJR_GRAPH=1 is safe to enable.
"""
import os
import sys
import time

import torch

# Make sure graph path is OFF for the reference run, ON for the graphed run
from src.bcjr.forward_backward import (
    L_BITS, V_DIM, build_pred_succ_tables, bcjr_forward_backward,
)


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    torch.manual_seed(0)
    B, N, V = 4, 16, V_DIM
    S = 1 << L_BITS

    seq_ref = torch.randn(B, N, V, device="cuda", requires_grad=True)
    seq_graph = seq_ref.detach().clone().requires_grad_(True)

    codebook = torch.randn(S, V, device="cuda")
    T_temp = torch.tensor(1.0, device="cuda")
    preds, succs = build_pred_succ_tables(device="cuda")

    # --- Reference forward+backward ---
    soft_ref, _ = bcjr_forward_backward(seq_ref, codebook, T_temp,
                                        preds=preds, succs=succs)
    soft_ref.sum().backward()

    # --- Graphed forward+backward ---
    # Import AFTER setting env so the module sees BCJR_GRAPH=1
    os.environ["BCJR_GRAPH"] = "1"
    from src.bcjr.bcjr_graph import graphed_bcjr_chunk_forward, reset_graph_cache
    reset_graph_cache()

    soft_graph = graphed_bcjr_chunk_forward(seq_graph, codebook, T_temp,
                                             preds, succs)
    soft_graph.sum().backward()

    # --- Parity check ---
    fwd_diff = (soft_ref - soft_graph).abs().max().item()
    fwd_rel = fwd_diff / soft_ref.abs().max().clamp(min=1e-30).item()
    print(f"forward max abs diff: {fwd_diff:.2e}  rel: {fwd_rel:.2e}",
          flush=True)

    bwd_diff = (seq_ref.grad - seq_graph.grad).abs().max().item()
    bwd_rel = bwd_diff / seq_ref.grad.abs().max().clamp(min=1e-30).item()
    print(f"backward max abs diff: {bwd_diff:.2e}  rel: {bwd_rel:.2e}",
          flush=True)

    # CUDA graphs can have slightly different reduction ordering → allow 1e-3.
    assert fwd_rel < 1e-3, f"forward parity failed: {fwd_rel:.2e}"
    assert bwd_rel < 1e-3, f"backward parity failed: {bwd_rel:.2e}"

    # --- Timing: reference vs graphed ---
    # Warm up both paths.
    for _ in range(3):
        _ = bcjr_forward_backward(seq_ref.detach(), codebook, T_temp,
                                  preds=preds, succs=succs)
        _ = graphed_bcjr_chunk_forward(seq_graph.detach(), codebook, T_temp,
                                        preds, succs)
    torch.cuda.synchronize()

    n_iter = 20

    t0 = time.time()
    for _ in range(n_iter):
        _ = bcjr_forward_backward(seq_ref.detach(), codebook, T_temp,
                                  preds=preds, succs=succs)
    torch.cuda.synchronize()
    dt_ref = (time.time() - t0) / n_iter * 1000

    t0 = time.time()
    for _ in range(n_iter):
        _ = graphed_bcjr_chunk_forward(seq_graph.detach(), codebook, T_temp,
                                        preds, succs)
    torch.cuda.synchronize()
    dt_graph = (time.time() - t0) / n_iter * 1000

    print(f"\nforward-only timing ({n_iter} iters averaged):")
    print(f"  reference: {dt_ref:.2f} ms")
    print(f"  graphed:   {dt_graph:.2f} ms")
    print(f"  speedup:   {dt_ref/dt_graph:.2f}×")

    print("\nPASS: CUDA graph parity + timing OK")


if __name__ == "__main__":
    main()
