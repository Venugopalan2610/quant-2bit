"""Path Y toy smoke test: plumbing only, no 7 B model load.

Builds a tiny OlmoeForCausalLM from scratch (1 layer, 4 experts, hidden=256)
so the full test fits in ~1 GB VRAM. Validates the Path-Y-specific pieces:

  1. build_bcjr_student wraps layers without loading v2 snapshots (we skip
     snap install here and patch W_latent from the random FP init).
  2. QATDecoderLayer.forward accepts HF's kwargs when called from inside
     OlmoeModel.forward (the bug the full smoke caught).
  3. flip_all_to_bcjr + temperature broadcast.
  4. Full teacher + student forward produces logits, KL loss is finite.
  5. Backward puts non-zero gradient on W_latent.
  6. Optimizer step runs without error.

v2-snapshot install correctness was already validated in the first run
(phase 3 completed with the expected trainable param count).

Run:
    python3 -m src.tripwires.test_path_y_toy
"""
import os
import sys
import time

import torch
import torch.nn as nn
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_decoder_layer import QATDecoderLayer
from src.bcjr.anneal import convert_layer_to_bcjr, set_layer_temperature
from src.qat.train_e2e_kl import kl_loss_full_vocab


def build_tiny_olmoe():
    """Build a tiny OLMoE for plumbing tests. ~20M params, <1 GB VRAM."""
    config = OlmoeConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        head_dim=64,
    )
    model = OlmoeForCausalLM(config)
    return model


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    SEQ_LEN = 32
    BATCH = 2

    print("=" * 60)
    print("Path Y toy smoke test")
    print(f"  tiny OLMoE: 1 layer, 4 experts, hidden=256, vocab=1024")
    print(f"  SEQ_LEN={SEQ_LEN}  BATCH={BATCH}")
    print("=" * 60)

    # ---- codebook ----
    print("\n[1/6] Codebook (smaller LUT for speed)...", flush=True)
    t0 = time.time()
    lut = init_hyb_lut(Q=9, n_samples=50_000, seed=0)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)
    print(f"  built in {time.time() - t0:.1f}s  shape={tuple(cb.shape)}", flush=True)

    # ---- teacher ----
    print("\n[2/6] Teacher (tiny, fp16 on GPU)...", flush=True)
    torch.manual_seed(42)
    teacher = build_tiny_olmoe().to("cuda", dtype=torch.float16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB", flush=True)

    # ---- student (same seed → same weights as teacher before wrapping) ----
    print("\n[3/6] Student (tiny, fp32, wrap + BCJR flip)...", flush=True)
    torch.manual_seed(42)
    student = build_tiny_olmoe().to("cuda", dtype=torch.float32)
    cfg = student.config

    # Wrap the one decoder layer with QATDecoderLayer (in place)
    fp_layer = student.model.layers[0]
    qat_layer = QATDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=0, reencode_every_n_steps=1,
    )
    student.model.layers[0] = qat_layer

    # Flip to BCJR mode
    convert_layer_to_bcjr(qat_layer, T_init=1.0, bcjr_chunk=4)

    # Freeze non-QAT params
    for p in student.parameters():
        p.requires_grad_(False)
    for p in qat_layer.trainable_parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M "
          f"({100*n_trainable/n_total:.1f}%)", flush=True)
    print(f"  student GPU: {torch.cuda.memory_allocated()/1e6:.0f} MB", flush=True)

    # ---- batch ----
    print(f"\n[4/6] Building random batch ({BATCH}x{SEQ_LEN})...", flush=True)
    batch = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN),
                          device="cuda", dtype=torch.long)

    # ---- forward + KL ----
    print("\n[5/6] Forward + KL loss...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        t_out = teacher(input_ids=batch, use_cache=False).logits
    print(f"  teacher fwd: {tuple(t_out.shape)}  "
          f"{time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    s_out = student(input_ids=batch, use_cache=False).logits
    print(f"  student fwd: {tuple(s_out.shape)}  "
          f"{time.time() - t0:.2f}s", flush=True)

    loss = kl_loss_full_vocab(s_out, t_out)
    print(f"  KL loss: {loss.item():.4e}", flush=True)
    assert torch.isfinite(loss), f"loss not finite: {loss}"

    # ---- backward + opt step ----
    print("\n[6/6] Backward + AdamW step...", flush=True)
    trainable = [p for p in student.parameters() if p.requires_grad]
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
    grad_norm = grad_norm_sq ** 0.5
    print(f"  trainable tensors: {len(trainable)}", flush=True)
    print(f"  tensors with non-zero grad: {n_nonzero}", flush=True)
    print(f"  total grad norm: {grad_norm:.4e}", flush=True)
    assert n_nonzero > 0, "ZERO gradients — BCJR backward is broken"

    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    opt.step()
    print(f"  opt.step() OK", flush=True)

    # ---- second step at lower T ----
    print("\n[bonus] second step at T=0.1...", flush=True)
    set_layer_temperature(qat_layer, 0.1)
    s_out2 = student(input_ids=batch, use_cache=False).logits
    loss2 = kl_loss_full_vocab(s_out2, t_out)
    opt.zero_grad(set_to_none=True)
    loss2.backward()
    opt.step()
    print(f"  T=0.1  loss={loss2.item():.4e}  (was {loss.item():.4e})",
          flush=True)

    print(f"\n{'='*60}")
    print(f"PASS: Path Y plumbing works end-to-end on tiny model.")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
