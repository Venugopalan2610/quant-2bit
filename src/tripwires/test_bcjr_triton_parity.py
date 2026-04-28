"""Parity: Triton alpha/beta step kernels vs eager reference.

Runs on tiny input (B=2, S=65536) so first-call compile finishes fast.
Checks fwd + bwd parity within fp32 tolerance.

If it passes, we can wire the kernels into bcjr_forward_backward.
"""
import sys
import time

import torch

from src.bcjr.forward_backward import (
    L_BITS, build_pred_succ_tables,
    _alpha_step as alpha_step_eager,
    _beta_step as beta_step_eager,
)
from src.bcjr.bcjr_triton import alpha_step_triton, beta_step_triton


def check_parity(name, eager_fn, triton_fn, inputs_eager, inputs_triton):
    out_eager = eager_fn(*inputs_eager)
    out_triton = triton_fn(*inputs_triton)

    fwd_diff = (out_eager - out_triton).abs().max().item()
    fwd_rel = fwd_diff / out_eager.abs().max().clamp(min=1e-30).item()
    print(f"{name} forward  diff={fwd_diff:.2e}  rel={fwd_rel:.2e}", flush=True)
    assert fwd_rel < 1e-4, f"{name} forward parity fail: rel={fwd_rel:.2e}"

    # backward
    grad_out = torch.randn_like(out_eager)
    # Eager backward
    g_eager = torch.autograd.grad(
        outputs=out_eager, inputs=[t for t in inputs_eager if t.requires_grad],
        grad_outputs=grad_out, retain_graph=False,
    )
    g_triton = torch.autograd.grad(
        outputs=out_triton, inputs=[t for t in inputs_triton if t.requires_grad],
        grad_outputs=grad_out, retain_graph=False,
    )
    for i, (ge, gt) in enumerate(zip(g_eager, g_triton)):
        bwd_diff = (ge - gt).abs().max().item()
        bwd_rel = bwd_diff / ge.abs().max().clamp(min=1e-30).item()
        print(f"{name} backward grad[{i}]  diff={bwd_diff:.2e}  rel={bwd_rel:.2e}",
              flush=True)
        assert bwd_rel < 1e-4, f"{name} backward[{i}] parity fail: rel={bwd_rel:.2e}"


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    torch.manual_seed(0)
    B, V = 2, 2
    S = 1 << L_BITS
    preds, succs = build_pred_succ_tables(device="cuda")

    # ----- alpha step parity -----
    la_prev_e = torch.randn(B, S, device="cuda", requires_grad=True)
    la_prev_t = la_prev_e.detach().clone().requires_grad_(True)
    local_e = torch.randn(B, S, device="cuda", requires_grad=True)
    local_t = local_e.detach().clone().requires_grad_(True)
    check_parity(
        "alpha_step",
        alpha_step_eager, alpha_step_triton,
        [la_prev_e, local_e, preds],
        [la_prev_t, local_t, preds],
    )

    # ----- beta step parity -----
    lb_next_e = torch.randn(B, S, device="cuda", requires_grad=True)
    lb_next_t = lb_next_e.detach().clone().requires_grad_(True)
    local_next_e = torch.randn(B, S, device="cuda", requires_grad=True)
    local_next_t = local_next_e.detach().clone().requires_grad_(True)
    check_parity(
        "beta_step",
        beta_step_eager, beta_step_triton,
        [lb_next_e, local_next_e, succs],
        [lb_next_t, local_next_t, succs],
    )

    # ----- quick timing: single step -----
    def time_step(fn, *args, warmup=3, iters=10):
        for _ in range(warmup):
            _ = fn(*args)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = fn(*args)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    B_real = 16
    la_prev = torch.randn(B_real, S, device="cuda", requires_grad=False)
    local = torch.randn(B_real, S, device="cuda", requires_grad=False)

    dt_eager = time_step(alpha_step_eager, la_prev, local, preds)
    dt_triton = time_step(alpha_step_triton, la_prev, local, preds)
    print(f"\nalpha step (B=16 forward-only timing, 10 iters avg):")
    print(f"  eager:  {dt_eager:.3f} ms")
    print(f"  triton: {dt_triton:.3f} ms")
    print(f"  speedup: {dt_eager/dt_triton:.2f}×")

    print("\nPASS: Triton alpha/beta parity + timing OK")


if __name__ == "__main__":
    main()
