"""Time BCJR forward+backward at real chunk size, eager vs Triton.

B=16, N=128 matches bcjr_quant.py default chunk. Reports per-chunk time
and extrapolates step-0 time on Llama-3.2-1B on the host GPU.
"""
import os
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    B, N, V = 16, 128, 2
    S = 1 << 16

    torch.manual_seed(0)
    sequences = torch.randn(B, N, V, device="cuda", dtype=torch.float32)
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    def run(path_label, enable_triton, n_iter=5):
        os.environ["BCJR_TRITON"] = "1" if enable_triton else "0"
        # Re-import to pick up env change (Python caches the module but
        # the dispatcher reads env each call).
        from src.bcjr.forward_backward import (
            build_pred_succ_tables, bcjr_forward_backward,
        )
        preds, succs = build_pred_succ_tables(device="cuda")

        # Warmup
        for _ in range(3):
            seq = sequences.clone().requires_grad_(True)
            soft, _ = bcjr_forward_backward(
                seq, codebook, T_temp, preds=preds, succs=succs,
            )
            soft.sum().backward()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_iter):
            seq = sequences.clone().requires_grad_(True)
            soft, _ = bcjr_forward_backward(
                seq, codebook, T_temp, preds=preds, succs=succs,
            )
            soft.sum().backward()
        torch.cuda.synchronize()
        dt_ms = (time.time() - t0) / n_iter * 1000
        print(f"{path_label:<20} {dt_ms:8.1f} ms per fwd+bwd", flush=True)
        return dt_ms

    print(f"BCJR chunk: B={B}  N={N}  S={S}\n")
    dt_eager = run("eager (BCJR_TRITON=0)", enable_triton=False)
    dt_triton = run("triton (BCJR_TRITON=1)", enable_triton=True)

    print(f"\nspeedup: {dt_eager / dt_triton:.2f}×")

    # Extrapolate step-0 on Llama-3.2-1B:
    chunks_per_step = 237_568
    print(f"\nLlama-3.2-1B step 0 (all 237,568 chunks):")
    print(f"  eager:  {chunks_per_step * dt_eager / 1000 / 60:.1f} min")
    print(f"  triton: {chunks_per_step * dt_triton / 1000 / 60:.1f} min")


if __name__ == "__main__":
    main()
