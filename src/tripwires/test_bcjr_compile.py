"""Test torch.compile(mode='reduce-overhead') on the monolithic BCJR.

'reduce-overhead' uses CUDA graphs internally. If it works on the current
torch version, we get graph capture for free without manual management.

Parity check: compiled vs reference. Timing: compiled vs monolithic (no graph).
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
from src.bcjr.bcjr_monolithic_graph import _bcjr_pure_forward


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)
    print(f"torch.__version__ = {torch.__version__}", flush=True)

    torch.manual_seed(0)
    B, N, V = 2, 8, V_DIM
    S = 1 << L_BITS
    codebook = torch.randn(S, V, device="cuda", dtype=torch.float32)
    T_temp = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    preds, succs = build_pred_succ_tables(device="cuda")

    # Build compiled version. This may take a while on first call (JIT compile).
    # backend='cudagraphs' bypasses Inductor (which has a typing bug on
    # Python 3.12 in some torch builds) and just wraps the function in a
    # CUDA graph. Exactly what we want for capturing the time loop.
    print("Compiling _bcjr_pure_forward with torch.compile(backend='cudagraphs')...",
          flush=True)
    t0 = time.time()
    try:
        compiled_fn = torch.compile(
            _bcjr_pure_forward, backend="cudagraphs", dynamic=False,
        )
    except Exception as e:
        print(f"torch.compile construction failed: {e}", flush=True)
        sys.exit(1)
    print(f"  compile-wrapper built in {time.time()-t0:.2f}s", flush=True)

    # Parity run (small input)
    seq_ref = torch.randn(B, N, V, device="cuda", dtype=torch.float32,
                          requires_grad=True)
    seq_cmp = seq_ref.detach().clone().requires_grad_(True)

    print("\nRunning reference...", flush=True)
    soft_ref, _ = bcjr_forward_backward(seq_ref, codebook, T_temp,
                                        preds=preds, succs=succs)

    print("Running compiled (first call = JIT trace)...", flush=True)
    t0 = time.time()
    try:
        soft_cmp = compiled_fn(seq_cmp, codebook, T_temp, preds, succs)
    except Exception as e:
        print(f"FAIL on first compiled call: {e}", flush=True)
        raise
    torch.cuda.synchronize()
    print(f"  first compiled call: {time.time()-t0:.2f}s "
          f"(includes JIT compilation)", flush=True)

    # Forward parity
    diff = (soft_ref - soft_cmp).abs().max().item()
    rel = diff / soft_ref.abs().max().clamp(min=1e-30).item()
    print(f"forward  diff={diff:.2e}  rel={rel:.2e}", flush=True)
    assert rel < 1e-3, f"forward parity fail: {rel:.2e}"

    # Backward parity
    grad_out = torch.randn_like(soft_ref)
    soft_ref.backward(grad_out)
    soft_cmp.backward(grad_out)
    diff = (seq_ref.grad - seq_cmp.grad).abs().max().item()
    rel = diff / seq_ref.grad.abs().max().clamp(min=1e-30).item()
    print(f"backward diff={diff:.2e}  rel={rel:.2e}", flush=True)
    assert rel < 1e-3, f"backward parity fail: {rel:.2e}"

    # Timing at real chunk size
    print(f"\nTiming at B=16, N=128:")
    B_real, N_real = 16, 128
    sequences = torch.randn(B_real, N_real, V, device="cuda", dtype=torch.float32)

    def time_fn(fn, label, warmup, iters):
        for _ in range(warmup):
            seq = sequences.clone().requires_grad_(True)
            out = fn(seq, codebook, T_temp, preds, succs)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            seq = sequences.clone().requires_grad_(True)
            out = fn(seq, codebook, T_temp, preds, succs)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    # Eager monolithic wrapper (returns tuple)
    def mono_wrap(seq, cb, T, p, s):
        return bcjr_forward_backward_monolithic(seq, cb, T, preds=p, succs=s)[0]

    dt_mono = time_fn(mono_wrap, "monolithic eager", warmup=3, iters=3)
    print(f"  monolithic eager:     {dt_mono:.0f} ms")

    dt_compile = time_fn(compiled_fn, "compiled", warmup=5, iters=5)
    print(f"  compiled:             {dt_compile:.0f} ms")
    print(f"  speedup compile/mono: {dt_mono/dt_compile:.2f}×")

    print("\nPASS: torch.compile works + parity holds")


if __name__ == "__main__":
    main()
