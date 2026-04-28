"""Path Y: end-to-end KL distillation for BCJR-QAT.

Setup:
  - FP16 teacher: OlmoeForCausalLM, frozen, no_grad forward.
  - Student: same model, every decoder layer wrapped as QATDecoderLayer in
    BCJR mode, W_latent fp32 (trainable), initialized from v2 snapshots.
  - Loss: full-vocabulary KL(student || teacher) per position.
  - Optimizer: bitsandbytes AdamW 8-bit on student W_latent.
  - Temperature schedule: exponential 1.0 -> 0.02 across total steps.

Memory budget on H200 140 GB (rough):
  Teacher FP16                     14 GB
  Student W_latent fp32            28 GB
  Student grads fp32               28 GB
  8-bit Adam state                 14 GB
  Student W_q cache bf16           14 GB
  Teacher activations (no_grad)     5 GB
  Student activations (ckpt)        8 GB
  BCJR workspace                    3 GB
  --------------------------------
  ~114 GB

Usage:
    python -m src.qat.train_e2e_kl \\
        --v2-dir cache/qat_bcjr_full_v2 \\
        --out-dir cache/qat_bcjr_e2e_kl \\
        --calib cache/calibration/tokens.npy \\
        --total-steps 4000 \\
        --batch-seqs 2 \\
        --lr 1e-5
"""
import os
import sys
import math
import time
import argparse
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.e2e_student import (
    build_bcjr_student,
    flip_all_to_bcjr,
    flip_all_to_ste,
    set_global_temperature,
    prime_all,
    student_trainable_parameters,
    freeze_non_qat,
    count_trainable,
)


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"


def exp_temperature_schedule(step, total_steps, T_0, T_min):
    """Exponential decay T_0 -> T_min over total_steps."""
    if total_steps <= 1:
        return T_min
    s = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return T_0 * (T_min / T_0) ** s


def kl_loss_full_vocab(student_logits, teacher_logits, mask=None):
    """KL(teacher || student) at every position, averaged.

    student_logits, teacher_logits: (B, T, V)
    mask: optional (B, T) float; default all ones.

    Uses teacher as the target distribution: standard forward-KL
    distillation (Hinton). This is what LLM-QAT uses.
    """
    V = student_logits.shape[-1]
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    p = log_p.exp()
    # kl_per_pos = sum_v p * (log_p - log_q)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp_min(1.0)
    return kl.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="cache/qat_bcjr_full_v2")
    parser.add_argument("--out-dir", default="cache/qat_bcjr_e2e_kl")
    parser.add_argument("--calib", default="cache/calibration/tokens.npy")
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--batch-seqs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--T-init", type=float, default=1.0)
    parser.add_argument("--T-min", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bcjr-chunk", type=int, default=16)
    parser.add_argument("--reencode-every", type=int, default=10,
                        help="BCJR reencode cadence. 1=every step (slow, soft-gradient each step). "
                             ">1 runs BCJR every N steps and uses cached W_q with STE gradient on the "
                             "intervening steps. Trades some BCJR fidelity for ~N× per-step speedup.")
    parser.add_argument("--mode", choices=["bcjr", "ste"], default="bcjr",
                        help="ste = hard Viterbi forward + STE gradient (fast, no BCJR autograd). "
                             "bcjr = soft codeword forward at temperature T with full BCJR autograd.")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable model.gradient_checkpointing_enable() on student")
    parser.add_argument("--use-adam8", action="store_true",
                        help="Use bitsandbytes AdamW 8-bit (memory-efficient)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Path Y: end-to-end KL distillation")
    print(f"  v2_dir: {args.v2_dir}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  calib: {args.calib}")
    print(f"  total_steps={args.total_steps}  lr={args.lr}  batch_seqs={args.batch_seqs}")
    print(f"  T: {args.T_init} -> {args.T_min} (exponential)")
    print(f"  grad_ckpt={args.grad_ckpt}  adam8={args.use_adam8}  "
          f"reencode_every={args.reencode_every}  bcjr_chunk={args.bcjr_chunk}")
    print("=" * 70)

    # ---------------- calibration tokens ----------------
    tokens = np.load(args.calib)
    print(f"\nCalibration: {tokens.shape} {tokens.dtype}")
    n_seqs, seq_len = tokens.shape
    tokens_t = torch.from_numpy(tokens.astype(np.int64))

    # ---------------- codebook ----------------
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # ---------------- teacher (FP16, frozen) ----------------
    print("\nLoading FP16 teacher...", flush=True)
    t0 = time.time()
    teacher = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher loaded in {time.time() - t0:.0f}s", flush=True)
    print(f"  teacher GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ---------------- student (fp32 trainable latent) ----------------
    print("\nLoading student base (fp32 on CUDA)...", flush=True)
    t0 = time.time()
    student = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cuda",
    )
    print(f"  student loaded in {time.time() - t0:.0f}s", flush=True)

    print("\nWrapping student with QAT layers + installing v2 codes...", flush=True)
    t0 = time.time()
    student, _ = build_bcjr_student(
        student, codebook_gpu=cb, v2_snap_dir=args.v2_dir,
        seed=args.seed, reencode_every_n_steps=args.reencode_every,
        bcjr_chunk=args.bcjr_chunk,
        measure_T0=False, prime=False, verbose=True,
    )
    print(f"  wrapped in {time.time() - t0:.0f}s", flush=True)
    print(f"  student GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Freeze non-QAT params; flip to requested mode
    freeze_non_qat(student)
    if args.mode == "bcjr":
        flip_all_to_bcjr(student, T_init=args.T_init, bcjr_chunk=args.bcjr_chunk)
        print(f"  mode=bcjr  T_init={args.T_init}", flush=True)
    else:
        flip_all_to_ste(student)
        print(f"  mode=ste  (T schedule inactive)", flush=True)
    n_train, n_total = count_trainable(student)
    print(f"  trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)",
          flush=True)

    if args.grad_ckpt:
        student.gradient_checkpointing_enable()
        print("  gradient checkpointing enabled", flush=True)

    # ---------------- optimizer ----------------
    trainable = list(student_trainable_parameters(student))
    if args.use_adam8:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
            print(f"  optimizer: AdamW8bit  ({len(trainable)} param tensors)",
                  flush=True)
        except ImportError:
            print("  bitsandbytes not available, falling back to AdamW fp32",
                  flush=True)
            opt = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"  peak VRAM after opt init: "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    # ---------------- training loop ----------------
    print("\n" + "=" * 70, flush=True)
    print("Starting training", flush=True)
    print("=" * 70, flush=True)

    stats = {"train_kl": [], "T": [], "val_kl": []}
    t_train0 = time.time()

    def _dump_mem_on_oom(label):
        print(f"\n*** OOM at {label} — memory summary ***", flush=True)
        print(torch.cuda.memory_summary(abbreviated=False), flush=True)
        print(f"*** max_memory_allocated = "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    # One-shot heartbeat: first forward through each layer prints enter/exit.
    # Hooks auto-unregister after step 0 so we don't spam the log forever.
    _heartbeat_handles = []
    _heartbeat_t0 = [None]  # list so nested fn can mutate
    def _make_pre(idx, tag):
        def _h(module, *_a, **_kw):
            if _heartbeat_t0[0] is None:
                _heartbeat_t0[0] = time.time()
            dt = time.time() - _heartbeat_t0[0]
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  [hb {dt:5.1f}s  mem={mem:5.1f}GB] enter {tag}{idx}", flush=True)
        return _h
    def _make_post(idx, tag):
        def _h(module, *_a, **_kw):
            dt = time.time() - _heartbeat_t0[0]
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  [hb {dt:5.1f}s  mem={mem:5.1f}GB] exit  {tag}{idx}", flush=True)
        return _h
    for i, layer in enumerate(student.model.layers):
        _heartbeat_handles.append(layer.register_forward_pre_hook(_make_pre(i, "L")))
        _heartbeat_handles.append(layer.register_forward_hook(_make_post(i, "L")))

    for step in range(args.total_steps):
        # temperature
        T_now = exp_temperature_schedule(step, args.total_steps,
                                         args.T_init, args.T_min)
        set_global_temperature(student, T_now)

        # batch
        idx = torch.randperm(n_seqs)[:args.batch_seqs]
        batch = tokens_t[idx].to("cuda", non_blocking=True)  # (B, T) int64

        # Phase timers (step 0 especially — we want to see exactly where time goes)
        t_phase = time.time()
        if step <= 2:
            print(f"\n--- step {step} start ---", flush=True)

        # teacher forward (no grad)
        try:
            with torch.no_grad():
                teacher_out = teacher(input_ids=batch, use_cache=False)
                teacher_logits = teacher_out.logits  # (B, T, V)
        except torch.cuda.OutOfMemoryError:
            _dump_mem_on_oom(f"step={step} teacher forward")
            raise
        if step <= 2:
            print(f"  teacher forward: {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        # student forward (grad)
        try:
            student_out = student(input_ids=batch, use_cache=False)
            student_logits = student_out.logits
        except torch.cuda.OutOfMemoryError:
            _dump_mem_on_oom(f"step={step} student forward")
            raise
        if step <= 2:
            print(f"  student forward: {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        loss = kl_loss_full_vocab(student_logits, teacher_logits)

        try:
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            _dump_mem_on_oom(f"step={step} backward")
            raise
        if step <= 2:
            print(f"  backward:        {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        opt.step()
        if step <= 2:
            print(f"  opt.step:        {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB  "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

        # After step 0, remove heartbeat hooks to stop log spam.
        if step == 0:
            for h in _heartbeat_handles:
                h.remove()
            _heartbeat_handles.clear()

        stats["train_kl"].append(float(loss.item()))
        stats["T"].append(float(T_now))

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t_train0
            rate = (step + 1) / elapsed
            eta = (args.total_steps - step - 1) / rate
            print(f"  step {step+1}/{args.total_steps}  "
                  f"kl={loss.item():.4e}  T={T_now:.3e}  "
                  f"{rate:.2f} step/s  ETA {eta/60:.0f} min", flush=True)

        if (step + 1) % args.eval_every == 0:
            # light eval on held-out chunk of calib
            val_kls = []
            with torch.no_grad():
                prime_all(student)  # refresh W_q cache
                for vi in range(4):
                    vb = tokens_t[-(vi+1)*args.batch_seqs:
                                  -vi*args.batch_seqs or None].to("cuda")
                    t_out = teacher(input_ids=vb, use_cache=False).logits
                    s_out = student(input_ids=vb, use_cache=False).logits
                    val_kls.append(float(kl_loss_full_vocab(s_out, t_out).item()))
            val_kl = sum(val_kls) / len(val_kls)
            stats["val_kl"].append((step + 1, val_kl))
            print(f"    [eval] val_kl={val_kl:.4e}", flush=True)

        if (step + 1) % args.ckpt_every == 0:
            save_snapshot(student, args.out_dir, tag=f"step{step+1}",
                          config_num_experts=teacher.config.num_experts,
                          intermediate_size=teacher.config.intermediate_size)
            torch.save(stats, os.path.join(args.out_dir, "train_stats.pt"))

    # ---------------- finalize ----------------
    print("\nFinalizing: flip to STE + prime + save...", flush=True)
    flip_all_to_ste(student)
    prime_all(student)
    save_snapshot(student, args.out_dir, tag="final",
                  config_num_experts=teacher.config.num_experts,
                  intermediate_size=teacher.config.intermediate_size)
    torch.save(stats, os.path.join(args.out_dir, "train_stats.pt"))

    total_sec = time.time() - t_train0
    print(f"\nDone.  total {total_sec/60:.1f} min  "
          f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)


def save_snapshot(student, out_dir, tag, config_num_experts, intermediate_size):
    """Save per-layer W_q snapshots in the same format as cache/qat_bcjr_full_v2/
    so install_bcjr + run_ppl pipelines work unchanged.
    """
    from src.qat.qat_decoder_layer import QATDecoderLayer
    ckpt_dir = os.path.join(out_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    for layer_idx, layer in enumerate(student.model.layers):
        if not isinstance(layer, QATDecoderLayer):
            continue
        snap = layer.snapshot_W_q_cpu()  # CPU fp32 dict
        path = os.path.join(ckpt_dir, f"layer_{layer_idx:02d}_wq.pt")
        torch.save(snap, path)
    print(f"    snapshot saved to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
