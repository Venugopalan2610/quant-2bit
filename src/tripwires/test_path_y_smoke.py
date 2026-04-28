"""Path Y smoke test for 4080 12GB.

Validates the full pipeline with a truncated model (1 decoder layer only)
before we commit to H200 rental. Tests:
  1. build_bcjr_student wraps layers in place without breaking forward.
  2. v2 snapshot installs into W_latent.
  3. BCJR mode enabled, T broadcasting works.
  4. Teacher (fp16) and student (fp32) forward both produce logits.
  5. KL loss backward produces non-zero gradients on W_latent.
  6. One optimizer step runs without OOM.

Run:
    python3 -m src.tripwires.test_path_y_smoke
"""
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.e2e_student import (
    build_bcjr_student,
    flip_all_to_bcjr,
    set_global_temperature,
    student_trainable_parameters,
    freeze_non_qat,
    count_trainable,
)
from src.qat.train_e2e_kl import kl_loss_full_vocab


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
V2_DIR = "cache/qat_bcjr_full_v2"


def _truncate_to_n_layers(model, n):
    """Destructively truncate an OlmoeForCausalLM to its first `n` decoder
    layers. Updates model.config.num_hidden_layers so forward terminates
    after n layers."""
    import torch.nn as nn
    assert 1 <= n <= len(model.model.layers)
    model.model.layers = nn.ModuleList(list(model.model.layers[:n]))
    model.config.num_hidden_layers = n


def main():
    if not torch.cuda.is_available():
        print("FAIL: need CUDA"); sys.exit(1)

    N_LAYERS = 1
    SEQ_LEN = 128
    BATCH = 1

    print("=" * 60)
    print(f"Path Y smoke test on 4080")
    print(f"  N_LAYERS={N_LAYERS}  SEQ_LEN={SEQ_LEN}  BATCH={BATCH}")
    print("=" * 60)

    # ---- codebook ----
    print("\n[1/6] Codebook...", flush=True)
    t0 = time.time()
    lut = init_hyb_lut(Q=9, n_samples=50_000, seed=0)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)
    print(f"  built in {time.time() - t0:.1f}s  shape={tuple(cb.shape)}", flush=True)

    # ---- teacher (fp16, truncated) ----
    # Load on CPU first, truncate, then move — full model won't fit on 12 GB GPU.
    print(f"\n[2/6] Teacher fp16 (truncated to {N_LAYERS} layers)...", flush=True)
    t0 = time.time()
    teacher = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cpu",
    )
    _truncate_to_n_layers(teacher, N_LAYERS)
    teacher.to("cuda")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher: {torch.cuda.memory_allocated()/1e9:.2f} GB  "
          f"{time.time() - t0:.1f}s", flush=True)

    # ---- student (bf16 base, fp32 kept layers, truncated) ----
    # Same CPU-load-then-truncate trick. Keep embeddings/lm_head in bf16
    # (frozen), promote kept decoder layers to fp32 so QuantizedLinear wraps
    # them as fp32 trainable latents.
    print(f"\n[3/6] Student (bf16 base, fp32 kept layers, {N_LAYERS} layers) + v2 install...",
          flush=True)
    t0 = time.time()
    student = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    _truncate_to_n_layers(student, N_LAYERS)
    student.to("cuda")
    # Promote kept decoder layers to fp32 for trainable QAT
    for i in range(N_LAYERS):
        student.model.layers[i] = student.model.layers[i].float()
    student, _ = build_bcjr_student(
        student, codebook_gpu=cb, v2_snap_dir=V2_DIR,
        seed=0, reencode_every_n_steps=1, bcjr_chunk=8,
        measure_T0=False, prime=False, verbose=False,
    )
    freeze_non_qat(student)
    flip_all_to_bcjr(student, T_init=1.0, bcjr_chunk=8)
    n_train, n_total = count_trainable(student)
    print(f"  wrapped in {time.time() - t0:.1f}s  "
          f"trainable={n_train/1e6:.1f}M  total={n_total/1e6:.1f}M", flush=True)
    print(f"  post-wrap GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # ---- tiny random batch ----
    print(f"\n[4/6] Building random batch ({BATCH}x{SEQ_LEN})...", flush=True)
    vocab = teacher.config.vocab_size
    batch = torch.randint(0, vocab, (BATCH, SEQ_LEN), device="cuda", dtype=torch.long)

    # ---- teacher + student forward ----
    print("\n[5/6] Forward + KL loss...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        t_out = teacher(input_ids=batch, use_cache=False).logits
    s_out = student(input_ids=batch, use_cache=False).logits
    print(f"  teacher logits: {tuple(t_out.shape)}  dtype={t_out.dtype}", flush=True)
    print(f"  student logits: {tuple(s_out.shape)}  dtype={s_out.dtype}", flush=True)
    assert t_out.shape == s_out.shape, "teacher/student logit shape mismatch"

    loss = kl_loss_full_vocab(s_out, t_out)
    print(f"  KL loss: {loss.item():.4e}", flush=True)
    assert torch.isfinite(loss), f"loss not finite: {loss}"

    # ---- backward + opt step ----
    print("\n[6/6] Backward + AdamW step...", flush=True)
    trainable = list(student_trainable_parameters(student))
    opt = torch.optim.AdamW(trainable, lr=1e-5)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # Verify gradients reached W_latent
    n_nonzero_grads = 0
    total_grad_norm = 0.0
    for p in trainable:
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_nonzero_grads += 1
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  trainable tensors: {len(trainable)}", flush=True)
    print(f"  tensors with non-zero grad: {n_nonzero_grads}", flush=True)
    print(f"  total grad norm: {total_grad_norm:.4e}", flush=True)
    assert n_nonzero_grads > 0, "ZERO gradients reached W_latent — plumbing broken"

    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    opt.step()
    print(f"  opt.step() OK  "
          f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # ---- second step with different T ----
    print("\n[bonus] second step at T=0.1 (test anneal plumbing)...", flush=True)
    set_global_temperature(student, 0.1)
    s_out = student(input_ids=batch, use_cache=False).logits
    loss2 = kl_loss_full_vocab(s_out, t_out)
    opt.zero_grad(set_to_none=True)
    loss2.backward()
    opt.step()
    print(f"  T=0.1  loss={loss2.item():.4e}  (was {loss.item():.4e})",
          flush=True)

    print(f"\n{'='*60}")
    print(f"PASS: Path Y pipeline plumbing works end-to-end.")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
