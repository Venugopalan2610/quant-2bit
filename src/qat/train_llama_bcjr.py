"""Llama BCJR-QAT end-to-end KL distillation.

Wraps every decoder layer of a dense LM (Llama / Mistral / Qwen2 / Phi-3)
with QATDenseDecoderLayer, flips all to BCJR mode, trains W_latent via
full-vocab KL to a frozen FP teacher.

No v2 PTQ dependency — W_latent initializes from FP weights directly.
The training run IS the quantization: BCJR soft codewords in forward,
gradient through BCJR autograd in backward, T annealed toward zero.

Run:
    python -m src.qat.train_llama_bcjr \\
        --model meta-llama/Llama-3.2-1B \\
        --total-steps 500 --use-adam8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.dense_student import (
    build_bcjr_student_dense,
    flip_all_to_bcjr,
    flip_all_to_ste,
    set_global_temperature,
    prime_all,
    student_trainable_parameters,
    freeze_non_qat,
    count_trainable,
)
from src.qat.train_e2e_kl import kl_loss_full_vocab, exp_temperature_schedule


def _load_or_synth_calibration(calib_path, tokenizer, n_seqs, seq_len, seed):
    """Load tokens.npy if available, else synthesize random-token calibration.

    Random-token calibration is fine for plumbing tests and for early
    training iterations where the student is still near the FP teacher —
    the gradient signal comes from KL(teacher, student), not from the
    token distribution being meaningful.
    """
    if calib_path and os.path.exists(calib_path):
        tokens = np.load(calib_path)
        print(f"  loaded calibration from {calib_path}: {tokens.shape}",
              flush=True)
        return tokens
    vocab_size = getattr(tokenizer, "vocab_size", 128256)  # Llama default
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=(n_seqs, seq_len), dtype=np.int64)
    print(f"  synthesized random-token calibration: {tokens.shape}  "
          f"vocab={vocab_size}", flush=True)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--out-dir", default="cache/llama_bcjr_qat")
    parser.add_argument("--calib", default="",
                        help="Optional tokens.npy path; if missing, synthesize "
                             "random tokens (OK for plumbing tests).")
    parser.add_argument("--n-seqs", type=int, default=512,
                        help="Used only when synthesizing calibration.")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--batch-seqs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--T-init", type=float, default=1.0)
    parser.add_argument("--T-min", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bcjr-chunk", type=int, default=32)
    parser.add_argument("--reencode-every", type=int, default=10)
    parser.add_argument("--student-dtype", choices=["fp32", "bf16"],
                        default="fp32")
    parser.add_argument("--teacher-dtype", choices=["fp16", "bf16"],
                        default="fp16")
    parser.add_argument("--use-adam8", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Llama BCJR-QAT: {args.model}")
    print(f"  total_steps={args.total_steps}  lr={args.lr}  batch_seqs={args.batch_seqs}")
    print(f"  T: {args.T_init} -> {args.T_min} (exponential)")
    print(f"  reencode_every={args.reencode_every}  bcjr_chunk={args.bcjr_chunk}")
    print(f"  student={args.student_dtype}  teacher={args.teacher_dtype}  "
          f"adam8={args.use_adam8}")
    print("=" * 70, flush=True)

    # ---- tokenizer (for vocab size and optional calibration gen) ----
    print(f"\nLoading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- calibration ----
    tokens = _load_or_synth_calibration(
        args.calib, tokenizer, args.n_seqs, args.seq_len, args.seed,
    )
    tokens_t = torch.from_numpy(tokens.astype(np.int64))
    n_seqs = tokens.shape[0]

    # ---- codebook ----
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # ---- teacher (frozen) ----
    teacher_dtype = torch.float16 if args.teacher_dtype == "fp16" else torch.bfloat16
    print(f"\nLoading teacher ({args.teacher_dtype})...", flush=True)
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=teacher_dtype, device_map="cuda",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher loaded in {time.time() - t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # ---- student (trainable) ----
    student_dtype = torch.float32 if args.student_dtype == "fp32" else torch.bfloat16
    print(f"\nLoading student ({args.student_dtype})...", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=student_dtype, device_map="cuda",
    )
    print(f"  student loaded in {time.time() - t0:.0f}s", flush=True)

    # ---- wrap all layers with BCJR-QAT ----
    print(f"\nWrapping {student.config.num_hidden_layers} layers with BCJR-QAT...",
          flush=True)
    t0 = time.time()
    student = build_bcjr_student_dense(
        student, codebook_gpu=cb,
        seed=args.seed, reencode_every_n_steps=args.reencode_every,
        bcjr_chunk=args.bcjr_chunk, verbose=True,
    )
    freeze_non_qat(student)
    flip_all_to_bcjr(student, T_init=args.T_init, bcjr_chunk=args.bcjr_chunk)
    n_train, n_total = count_trainable(student)
    print(f"  wrapped in {time.time() - t0:.0f}s  "
          f"trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.1f}%)", flush=True)
    print(f"  student GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # ---- optimizer ----
    trainable = list(student_trainable_parameters(student))
    if args.use_adam8:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
            print(f"  optimizer: AdamW8bit ({len(trainable)} tensors)", flush=True)
        except ImportError:
            opt = torch.optim.AdamW(trainable, lr=args.lr)
            print("  bitsandbytes missing, using fp32 AdamW", flush=True)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"  peak VRAM after opt init: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ---- training loop ----
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70, flush=True)

    t_train0 = time.time()
    losses = []

    for step in range(args.total_steps):
        T_now = exp_temperature_schedule(step, args.total_steps,
                                         args.T_init, args.T_min)
        set_global_temperature(student, T_now)

        idx = torch.randperm(n_seqs)[:args.batch_seqs]
        batch = tokens_t[idx].to("cuda", non_blocking=True)

        t_step = time.time()

        with torch.no_grad():
            t_out = teacher(input_ids=batch, use_cache=False).logits

        s_out = student(input_ids=batch, use_cache=False).logits

        loss = kl_loss_full_vocab(s_out, t_out)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        opt.step()

        step_dt = time.time() - t_step
        losses.append(float(loss.item()))

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t_train0
            rate = (step + 1) / elapsed
            eta = (args.total_steps - step - 1) / rate
            print(f"  step {step+1}/{args.total_steps}  "
                  f"kl={loss.item():.4e}  T={T_now:.3e}  "
                  f"dt={step_dt:.1f}s  "
                  f"ETA {eta/60:.1f}min  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        if (step + 1) % args.ckpt_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"step{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
            for layer_idx, layer in enumerate(student.model.layers):
                if isinstance(layer, QATDenseDecoderLayer):
                    snap = layer.snapshot_W_q_cpu()
                    torch.save(snap, os.path.join(
                        ckpt_dir, f"layer_{layer_idx:02d}_wq.pt"
                    ))
            print(f"  [ckpt] saved {ckpt_dir}", flush=True)

    # ---- finalize: flip to STE, prime, save ----
    print("\n" + "=" * 70)
    print("Finalizing: flip to STE + prime cache + save...")
    print("=" * 70, flush=True)
    flip_all_to_ste(student)
    prime_all(student)

    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
    for layer_idx, layer in enumerate(student.model.layers):
        if isinstance(layer, QATDenseDecoderLayer):
            snap = layer.snapshot_W_q_cpu()
            torch.save(snap, os.path.join(final_dir, f"layer_{layer_idx:02d}_wq.pt"))
    print(f"  saved final snapshots to {final_dir}", flush=True)

    print(f"\n  train loss trajectory: {losses[0]:.4e} -> {losses[-1]:.4e}",
          flush=True)
    print(f"  wall time: {(time.time() - t_train0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
