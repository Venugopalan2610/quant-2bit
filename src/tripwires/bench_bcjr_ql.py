"""Measure one BCJR forward+backward on a single real-sized Llama QL.

No estimates. Just timing. Runs in <2 min, uses <5 GB VRAM.

Tests each Llama-3.2-1B matrix shape:
  attn q/o:   (2048, 2048)
  attn k/v:   (512, 2048)
  mlp gate/up: (8192, 2048)
  mlp down:   (2048, 8192)

For each, runs ONE BCJR forward+backward and reports the ms cost.
Multiply by 7 per layer × 16 layers to get layer-total; then by 2 (fwd+bwd)
to get step-0 estimate — grounded in measurement, not guesswork.
"""
import sys
import time

import torch

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.quantized_linear import QuantizedLinear, _get_bcjr_tables
from src.bcjr.bcjr_quant import bcjr_quantize_weight


def bench_shape(out_f, in_f, codebook, label, dtype, chunk):
    ql = QuantizedLinear(
        in_features=in_f, out_features=out_f, codebook_gpu=codebook,
        seed=0, reencode_every_n_steps=1, quant_mode="bcjr",
        T_init=1.0, bcjr_chunk=chunk, device="cuda", dtype=dtype,
    )
    preds, succs = _get_bcjr_tables("cuda")

    # Forward
    torch.cuda.synchronize(); t0 = time.time()
    W_q = bcjr_quantize_weight(
        ql.W_latent, ql.sign_l, ql.sign_r, ql.codebook_gpu, ql.T_temp,
        preds=preds, succs=succs, chunk_size=chunk,
    )
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000

    # Backward (scalar loss on W_q, measures grad-to-W_latent cost)
    loss = W_q.pow(2).mean()
    torch.cuda.synchronize(); t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t0) * 1000

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    params_m = (out_f * in_f) / 1e6
    print(f"  {label:<20} {tuple(W_q.shape)!s:<18} "
          f"{params_m:6.1f}M params  "
          f"fwd {fwd_ms:7.0f} ms  bwd {bwd_ms:7.0f} ms  "
          f"peak {mem_gb:.2f} GB",
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    del ql, W_q, loss
    torch.cuda.empty_cache()
    return fwd_ms, bwd_ms


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    print("Building codebook (once)...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=0)
    codebook = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # Touch BCJR tables once so first-call setup doesn't pollute timings.
    _ = _get_bcjr_tables("cuda")

    shapes = [
        ("attn_q_proj",  2048, 2048),
        ("attn_k_proj",   512, 2048),
        ("attn_v_proj",   512, 2048),
        ("attn_o_proj",  2048, 2048),
        ("mlp_gate_proj", 8192, 2048),
        ("mlp_up_proj",   8192, 2048),
        ("mlp_down_proj", 2048, 8192),
    ]

    for dtype_label, dtype, chunk in [("bf16", torch.bfloat16, 16),
                                      ("bf16", torch.bfloat16, 32)]:
        print(f"\n=== dtype={dtype_label}  bcjr_chunk={chunk} ===", flush=True)
        total_fwd = 0.0
        total_bwd = 0.0
        for label, out_f, in_f in shapes:
            f, b = bench_shape(out_f, in_f, codebook, label, dtype, chunk)
            total_fwd += f
            total_bwd += b
        print(f"\n  PER-LAYER  (7 QLs):  fwd {total_fwd:7.0f} ms  "
              f"bwd {total_bwd:7.0f} ms  total {total_fwd+total_bwd:7.0f} ms",
              flush=True)
        print(f"  16-LAYER   (step 0): fwd {16*total_fwd/1000:6.1f} s  "
              f"bwd {16*total_bwd/1000:6.1f} s  "
              f"total {16*(total_fwd+total_bwd)/1000:6.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
