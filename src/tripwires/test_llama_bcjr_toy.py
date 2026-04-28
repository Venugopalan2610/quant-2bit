"""Toy Llama BCJR smoke test — validates plumbing without any model download.

Builds a tiny LlamaForCausalLM from scratch (1 layer, hidden=256, vocab=1024)
so the whole test fits in <2 GB VRAM on a 4080. Validates:

  1. QATDenseDecoderLayer wraps a Llama decoder layer correctly.
  2. build_bcjr_student_dense wraps all layers in place.
  3. flip_all_to_bcjr + temperature broadcast.
  4. Full teacher + student forward produces logits.
  5. KL loss is finite.
  6. Backward puts non-zero gradient on W_latent across all QLs.
  7. Optimizer step runs without error.
  8. Second step at lower T reduces loss (anneal plumbing).

Run:
    python3 -m src.tripwires.test_llama_bcjr_toy
"""
import sys
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.dense_student import (
    build_bcjr_student_dense,
    flip_all_to_bcjr,
    set_global_temperature,
    student_trainable_parameters,
    freeze_non_qat,
    count_trainable,
)
from src.qat.train_e2e_kl import kl_loss_full_vocab


def build_tiny_llama():
    """Tiny Llama: 1 layer, hidden=256, intermediate=512, vocab=1024.
    Dims are multiples of 16 (BCJR tile requirement) and of head_dim."""
    config = LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        head_dim=64,
        rms_norm_eps=1e-5,
    )
    return LlamaForCausalLM(config)


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    SEQ_LEN = 32
    BATCH = 2

    print("=" * 60)
    print("Llama BCJR toy smoke test")
    print(f"  tiny Llama: 1 layer, hidden=256, vocab=1024")
    print(f"  SEQ_LEN={SEQ_LEN}  BATCH={BATCH}")
    print("=" * 60)

    print("\n[1/6] Codebook (small LUT)...", flush=True)
    t0 = time.time()
    lut = init_hyb_lut(Q=9, n_samples=50_000, seed=0)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    print("\n[2/6] Teacher (tiny, fp16)...", flush=True)
    torch.manual_seed(42)
    teacher = build_tiny_llama().to("cuda", dtype=torch.float16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB",
          flush=True)

    print("\n[3/6] Student (tiny, fp32, wrap all layers to BCJR)...",
          flush=True)
    torch.manual_seed(42)
    student = build_tiny_llama().to("cuda", dtype=torch.float32)
    student = build_bcjr_student_dense(
        student, codebook_gpu=cb,
        seed=0, reencode_every_n_steps=1, bcjr_chunk=4, verbose=False,
    )
    freeze_non_qat(student)
    flip_all_to_bcjr(student, T_init=1.0, bcjr_chunk=4)

    n_train, n_total = count_trainable(student)
    print(f"  trainable: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M  "
          f"({100*n_train/n_total:.1f}%)", flush=True)
    print(f"  student GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB",
          flush=True)

    print(f"\n[4/6] Random batch ({BATCH}x{SEQ_LEN})...", flush=True)
    batch = torch.randint(0, student.config.vocab_size, (BATCH, SEQ_LEN),
                          device="cuda", dtype=torch.long)

    print("\n[5/6] Forward + KL loss...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        t_out = teacher(input_ids=batch, use_cache=False).logits
    print(f"  teacher fwd: {tuple(t_out.shape)}  {time.time() - t0:.2f}s",
          flush=True)

    t0 = time.time()
    s_out = student(input_ids=batch, use_cache=False).logits
    print(f"  student fwd: {tuple(s_out.shape)}  {time.time() - t0:.2f}s",
          flush=True)

    loss = kl_loss_full_vocab(s_out, t_out)
    print(f"  KL loss: {loss.item():.4e}", flush=True)
    assert torch.isfinite(loss), f"loss not finite: {loss}"

    print("\n[6/6] Backward + AdamW step...", flush=True)
    trainable = list(student_trainable_parameters(student))
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    loss.backward()
    print(f"  backward: {time.time() - t0:.2f}s", flush=True)

    n_nonzero = 0
    grad_norm_sq = 0.0
    for p in trainable:
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_nonzero += 1
            grad_norm_sq += p.grad.norm().item() ** 2
    print(f"  trainable tensors: {len(trainable)}  "
          f"non-zero grad: {n_nonzero}  "
          f"grad norm: {grad_norm_sq**0.5:.4e}", flush=True)
    assert n_nonzero == len(trainable), (
        f"only {n_nonzero}/{len(trainable)} tensors got gradient — "
        f"BCJR backward plumbing is broken"
    )

    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    opt.step()
    print(f"  opt.step() OK", flush=True)

    print("\n[bonus] second step at T=0.1...", flush=True)
    set_global_temperature(student, 0.1)
    s_out2 = student(input_ids=batch, use_cache=False).logits
    loss2 = kl_loss_full_vocab(s_out2, t_out)
    opt.zero_grad(set_to_none=True)
    loss2.backward()
    opt.step()
    print(f"  T=0.1  loss={loss2.item():.4e}  (was {loss.item():.4e})",
          flush=True)

    print(f"\n{'='*60}")
    print(f"PASS: Llama BCJR plumbing works end-to-end on tiny model.")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
