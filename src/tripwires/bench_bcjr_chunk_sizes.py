"""Sweep bcjr_chunk size to find the optimal chunk for Llama-1B on the host GPU.

With Triton kernels, launch overhead dominates — bigger chunks should amortize
the per-chunk fixed cost (log_local allocation, Python dispatch) over more work.
Memory scales linearly with chunk, so on 4080 we're limited; on cloud we have
more headroom.
"""
import os
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    N, V = 128, 2
    S = 1 << 16

    torch.manual_seed(0)
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    os.environ["BCJR_TRITON"] = "1"
    from src.bcjr.forward_backward import (
        build_pred_succ_tables, bcjr_forward_backward,
    )
    preds, succs = build_pred_succ_tables(device="cuda")

    chunk_sizes = [16, 32, 64, 128]

    print(f"{'chunk':<8}{'per-chunk ms':<15}{'peak VRAM GB':<15}"
          f"{'extrap. step-0 min':<20}")
    print("-" * 60)

    for chunk in chunk_sizes:
        sequences = torch.randn(chunk, N, V, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            seq = sequences.clone().requires_grad_(True)
            try:
                soft, _ = bcjr_forward_backward(
                    seq, codebook, T_temp, preds=preds, succs=succs,
                )
                soft.sum().backward()
            except torch.cuda.OutOfMemoryError:
                print(f"{chunk:<8}OOM")
                torch.cuda.empty_cache()
                break
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Time
        n_iter = 3
        t0 = time.time()
        for _ in range(n_iter):
            seq = sequences.clone().requires_grad_(True)
            soft, _ = bcjr_forward_backward(
                seq, codebook, T_temp, preds=preds, succs=succs,
            )
            soft.sum().backward()
        torch.cuda.synchronize()
        dt_ms = (time.time() - t0) / n_iter * 1000
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        # Extrapolate: 237568 TOTAL tiles in Llama step, so chunks_used = 237568 / chunk
        total_tiles = 237_568
        chunks_needed = total_tiles // chunk
        extrap_min = chunks_needed * dt_ms / 1000 / 60
        print(f"{chunk:<8}{dt_ms:<15.1f}{peak_gb:<15.2f}{extrap_min:<20.1f}",
              flush=True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
