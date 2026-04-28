"""Llama BCJR-QAT: single-layer end-to-end KL distillation.

Full 16-layer Llama-3.2-1B student, but ONLY layer --target-layer is wrapped
with QATDenseDecoderLayer in BCJR mode. Other layers stay at original FP
(frozen). Loss is full-vocab KL to a frozen FP16 teacher, backward flows
through all 16 layers but only layer K's W_latent receives gradient.

Demonstrates the BCJR-QAT mechanism at real Llama-1B scale without
requiring v2 PTQ snapshots (we don't have those for Llama). 4080-viable
overnight: ~60-90 min per step × 3 steps = 3-5 hours.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.bcjr.anneal import convert_layer_to_bcjr, convert_layer_to_ste, set_layer_temperature
from src.qat.train_e2e_kl import kl_loss_full_vocab, exp_temperature_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--calib", default="cache/calibration/tokens_llama.npy")
    parser.add_argument("--out-dir", default="cache/llama_bcjr_single")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=3)
    parser.add_argument("--batch-seqs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--T-init", type=float, default=1.0)
    parser.add_argument("--T-min", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bcjr-chunk", type=int, default=16)
    parser.add_argument("--reencode-every", type=int, default=1,
                        help="BCJR every N steps. 1=every step (pure Path Y).")
    parser.add_argument("--init-from-ptq", action="store_true",
                        help="Initialize W_latent to hard-Viterbi PTQ codes "
                             "(instead of FP weights). Cleaner comparison: any "
                             "deviation from PTQ baseline = pure BCJR training effect.")
    parser.add_argument("--resume-from", default=None,
                        help="Resume training from a per-step W_latent ckpt. "
                             "Loads W_latent, sets start_step from ckpt['step'], "
                             "and continues the T schedule for the remaining steps. "
                             "Mutually exclusive with --init-from-ptq.")
    parser.add_argument("--use-adam8", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="Save W_latent snapshot every N steps (0 = off, "
                             "save only at end). Each ckpt is ~230 MB.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.resume_from and args.init_from_ptq:
        print("FAIL: --resume-from and --init-from-ptq are mutually exclusive")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Force monolithic BCJR (the 6.57× fast path with single autograd node)
    os.environ["BCJR_TRITON"] = "1"
    os.environ["BCJR_MONOLITHIC"] = "1"

    print("=" * 70)
    print(f"Llama BCJR-QAT single-layer: {args.model}")
    print(f"  target_layer={args.target_layer}  total_steps={args.total_steps}")
    print(f"  seq_len={args.seq_len}  batch_seqs={args.batch_seqs}  lr={args.lr}")
    print(f"  T: {args.T_init} -> {args.T_min}  reencode_every={args.reencode_every}")
    print(f"  bcjr_chunk={args.bcjr_chunk}  adam8={args.use_adam8}")
    print("=" * 70, flush=True)

    # Calibration (must be tokenized with Llama's tokenizer)
    tokens = np.load(args.calib)
    print(f"\nCalibration: {tokens.shape} {tokens.dtype}", flush=True)
    n_seqs, seq_len_calib = tokens.shape
    if args.seq_len > seq_len_calib:
        print(f"  capping seq_len to calib width: {seq_len_calib}")
        args.seq_len = seq_len_calib
    tokens_t = torch.from_numpy(tokens[:, :args.seq_len].astype(np.int64))

    # Codebook
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # Teacher (fp16 to save VRAM)
    print("\nLoading teacher (fp16)...", flush=True)
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher loaded in {time.time()-t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Student (bf16 so fp32 W_latent for target layer fits in 12 GB with AdamW)
    print("\nLoading student (bf16)...", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    print(f"  student loaded in {time.time()-t0:.0f}s", flush=True)

    # Wrap ONLY the target layer with QATDenseDecoderLayer in BCJR mode
    cfg = student.config
    print(f"\nWrapping layer {args.target_layer} with BCJR-QAT...", flush=True)
    t0 = time.time()
    # Convert just the target layer to fp32 so W_latent is fp32 trainable
    fp_layer = student.model.layers[args.target_layer].float()
    qat_layer = QATDenseDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=args.seed + args.target_layer * 997,
        reencode_every_n_steps=args.reencode_every,
    )
    student.model.layers[args.target_layer] = qat_layer

    start_step = 0
    if args.init_from_ptq:
        # Replace W_latent with hard-Viterbi PTQ codes so training starts
        # at the PTQ baseline. Cleaner attribution: any final PPL change
        # vs PTQ baseline is pure BCJR training effect, not init noise.
        print(f"\nInitializing W_latent from hard-Viterbi PTQ codes...",
              flush=True)
        with torch.no_grad():
            for name, ql in qat_layer._all_qls():
                # TrellisQuantSTE: RHT → tile → Viterbi → untile → invRHT
                # Returns W_q in original (pre-IP) basis, fp32.
                W_q = TrellisQuantSTE.apply(
                    ql.W_latent.float(), ql.sign_l, ql.sign_r, ql.codebook_gpu,
                )
                ql.W_latent.copy_(W_q.to(ql.W_latent.dtype))
                ql._W_q_cache.copy_(W_q.to(ql._W_q_cache.dtype))
                print(f"  {name}: ||W_fp - W_PTQ|| = "
                      f"{(ql.W_latent.float() - W_q.float()).norm().item():.4e}",
                      flush=True)
    elif args.resume_from:
        # Load W_latent from a per-step ckpt and continue. Optimizer state is
        # NOT restored (we use a fresh AdamW8bit) — beta1/2 moments converge
        # in 1-2 steps so the warmup cost is small.
        print(f"\nResuming from {args.resume_from}...", flush=True)
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        start_step = int(ckpt["step"])
        print(f"  ckpt step={start_step}  T_at_save={ckpt['T']:.4e}  "
              f"kl_at_save={ckpt['kl']:.4e}", flush=True)
        if start_step >= args.total_steps:
            print(f"FAIL: ckpt step {start_step} >= total_steps {args.total_steps}; "
                  f"nothing to resume.")
            sys.exit(1)
        w_latent_dict = ckpt["W_latent"]
        with torch.no_grad():
            for name, ql in qat_layer._all_qls():
                if name not in w_latent_dict:
                    raise KeyError(
                        f"resume ckpt missing W_latent for {name!r}. "
                        f"Got: {sorted(w_latent_dict.keys())}"
                    )
                W = w_latent_dict[name].to(ql.W_latent.device, dtype=ql.W_latent.dtype)
                ql.W_latent.copy_(W)
        print(f"  resumed: will run steps {start_step}..{args.total_steps-1} "
              f"({args.total_steps - start_step} steps remaining)", flush=True)

    # Freeze everything except target layer's W_latent
    for _, p in student.named_parameters():
        p.requires_grad_(False)
    for p in qat_layer.trainable_parameters():
        p.requires_grad_(True)

    convert_layer_to_bcjr(qat_layer, T_init=args.T_init,
                          bcjr_chunk=args.bcjr_chunk)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  wrapped in {time.time()-t0:.0f}s  "
          f"trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}%)", flush=True)
    print(f"  GPU after wrap: {torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    # Optimizer
    trainable = list(qat_layer.trainable_parameters())
    if args.use_adam8:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
            print(f"  optimizer: AdamW8bit ({len(trainable)} tensors)",
                  flush=True)
        except ImportError:
            opt = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"  peak VRAM after opt init: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70, flush=True)

    t_train0 = time.time()
    losses = []

    for step in range(start_step, args.total_steps):
        T_now = exp_temperature_schedule(step, args.total_steps,
                                         args.T_init, args.T_min)
        set_layer_temperature(qat_layer, T_now)

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
        elapsed = time.time() - t_train0
        print(f"  step {step+1}/{args.total_steps}  "
              f"kl={loss.item():.4e}  T={T_now:.3e}  "
              f"dt={step_dt/60:.1f}min  "
              f"elapsed={elapsed/60:.1f}min  "
              f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.1f}GB",
              flush=True)

        # Per-step W_latent snapshot for trajectory analysis + interruption
        # insurance. We save ONLY W_latent (no opt state, no mode flips), so
        # the training state is untouched by the save. To use a ckpt: load
        # W_latent into a fresh model, run TrellisQuantSTE.apply to harden
        # to Viterbi codes, install via the existing eval pipeline.
        if args.ckpt_every > 0 and ((step + 1) % args.ckpt_every == 0):
            ckpt = {
                "step": step + 1,
                "T": float(T_now),
                "kl": float(loss.item()),
                "W_latent": {
                    name: ql.W_latent.detach().cpu().clone()
                    for name, ql in qat_layer._all_qls()
                },
            }
            ckpt_suffix = "_ptqinit" if args.init_from_ptq else ""
            ckpt_path = os.path.join(
                args.out_dir,
                f"layer_{args.target_layer:02d}_step{step+1:02d}{ckpt_suffix}_latent.pt",
            )
            torch.save(ckpt, ckpt_path)
            print(f"    ckpt: {ckpt_path}", flush=True)

    # Finalize: flip to STE + save
    print("\n" + "=" * 70)
    print("Finalizing + saving snapshot")
    print("=" * 70, flush=True)
    convert_layer_to_ste(qat_layer)
    qat_layer.prime_cache()
    snap = qat_layer.snapshot_W_q_cpu()
    suffix = "_ptqinit" if args.init_from_ptq else ""
    out_path = os.path.join(args.out_dir,
                            f"layer_{args.target_layer:02d}_wq{suffix}.pt")
    torch.save(snap, out_path)
    print(f"  saved to {out_path}", flush=True)

    print(f"\n  losses: {losses}")
    print(f"  loss trajectory: {losses[0]:.4e} -> {losses[-1]:.4e}")
    print(f"  total wall time: {(time.time()-t_train0)/60:.1f} min")


if __name__ == "__main__":
    main()
