"""Time ONE BCJR chunk forward+backward at real chunk size.

B=16, N=128 (one chunk of real shape, matching bcjr_quant.py default).
Runs reference and graphed paths, reports ms per call and speedup.

Completes in seconds, not minutes. No matrix iteration, no training.
"""
import os
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    B, N, V = 16, 128, 2      # one real BCJR chunk
    S = 1 << 16

    torch.manual_seed(0)
    # BCJR internals are fp32 (codebook is fp32). In the real training path,
    # bf16 W_latent is upcast before entering BCJR.
    sequences = torch.randn(B, N, V, device="cuda", dtype=torch.float32)
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    # Reference path (no graph)
    os.environ["BCJR_GRAPH"] = "0"
    from src.bcjr.forward_backward import (
        build_pred_succ_tables, bcjr_forward_backward,
    )
    preds, succs = build_pred_succ_tables(device="cuda")

    # Warm up reference
    for _ in range(3):
        seq = sequences.clone().requires_grad_(True)
        soft, _ = bcjr_forward_backward(seq, codebook, T_temp,
                                        preds=preds, succs=succs)
        soft.sum().backward()
    torch.cuda.synchronize()

    # Time reference forward+backward
    n_iter = 5
    t0 = time.time()
    for _ in range(n_iter):
        seq = sequences.clone().requires_grad_(True)
        soft, _ = bcjr_forward_backward(seq, codebook, T_temp,
                                        preds=preds, succs=succs)
        soft.sum().backward()
    torch.cuda.synchronize()
    dt_ref = (time.time() - t0) / n_iter * 1000
    print(f"Reference (no graph):  {dt_ref:.0f} ms per BCJR fwd+bwd")

    # Graphed path
    os.environ["BCJR_GRAPH"] = "1"
    from src.bcjr.bcjr_graph import graphed_bcjr_chunk_forward, reset_graph_cache
    reset_graph_cache()

    # Warm up graph (first call captures)
    print("  ... capturing CUDA graph (first call, slow) ...", flush=True)
    for _ in range(3):
        seq = sequences.clone().requires_grad_(True)
        soft = graphed_bcjr_chunk_forward(seq, codebook, T_temp, preds, succs)
        soft.sum().backward()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        seq = sequences.clone().requires_grad_(True)
        soft = graphed_bcjr_chunk_forward(seq, codebook, T_temp, preds, succs)
        soft.sum().backward()
    torch.cuda.synchronize()
    dt_graph = (time.time() - t0) / n_iter * 1000
    print(f"Graphed:               {dt_graph:.0f} ms per BCJR fwd+bwd")
    print(f"Speedup:               {dt_ref/dt_graph:.2f}×")

    # Extrapolate: how many chunks in a full Llama-1B step?
    #   q/o proj (2048x2048): 128×128 = 16384 tiles → 16384/16 = 1024 chunks
    #   k/v proj (512x2048):   128×32  = 4096 tiles  → 256 chunks
    #   mlp gate/up (8192x2048): 512×128 = 65536 tiles → 4096 chunks
    #   mlp down (2048x8192):  same as gate/up = 4096 chunks
    chunks_per_layer = 2 * 1024 + 2 * 256 + 2 * 4096 + 4096  # 14848 per layer
    chunks_per_step = 16 * chunks_per_layer                   # 237568 per step

    print(f"\nLlama-3.2-1B: {chunks_per_step:,} BCJR chunks per training step "
          f"(reencode step only)")
    print(f"  extrapolated reference: "
          f"{chunks_per_step * dt_ref / 1000 / 60:.1f} min per BCJR step")
    print(f"  extrapolated graphed:   "
          f"{chunks_per_step * dt_graph / 1000 / 60:.1f} min per BCJR step")


if __name__ == "__main__":
    main()
