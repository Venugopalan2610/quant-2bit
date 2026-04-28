"""Path Y single-layer: BCJR + E2E KL distillation on ONE decoder layer.

Approach:
  - Teacher: FP16 OLMoE, frozen.
  - Student: full 16-layer OLMoE. Layers 0..15 except --target-layer get v2
    PTQ weights (frozen fp32 nn.Linear). The target layer is wrapped with
    QATDecoderLayer in BCJR mode; its W_latent is the only trainable tensor.
  - Loss: full-vocab KL(teacher, student) at the LM head — end-to-end.
  - Backward flows through all 16 student layers; only target layer's W_latent
    receives gradient (other layers have no trainable params).

Preserves Path Y's two novel claims at real-model scale:
  1. BCJR soft codewords with full forward-backward autograd during training
  2. Global LM-distillation (KL over full vocab), not per-layer MSE

Scaling: demonstrated on one decoder layer. Full-model scaling is engineering
future work — current BCJR implementation is too slow across all 2112 QLs.

Run:
    python -m src.qat.train_single_layer_path_y \\
        --target-layer 8 --total-steps 100 --use-adam8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import OlmoeForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.qat_decoder_layer import QATDecoderLayer
from src.bcjr.anneal import convert_layer_to_bcjr, set_layer_temperature
from src.eval.install_bcjr import install_bcjr_weights
from src.qat.e2e_student import _install_snapshot_into_qat_layer, _load_v2_snapshot
from src.qat.train_e2e_kl import kl_loss_full_vocab, exp_temperature_schedule


MODEL_DIR = "cache/model/olmoe-1b-7b-0125"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="cache/qat_bcjr_full_v2")
    parser.add_argument("--out-dir", default="cache/qat_bcjr_path_y_single")
    parser.add_argument("--calib", default="cache/calibration/tokens.npy")
    parser.add_argument("--target-layer", type=int, required=True,
                        help="Which decoder layer (0..15) to train with BCJR.")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--batch-seqs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--T-init", type=float, default=1.0)
    parser.add_argument("--T-min", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bcjr-chunk", type=int, default=32)
    parser.add_argument("--reencode-every", type=int, default=10)
    parser.add_argument("--use-adam8", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Path Y single-layer: target_layer={args.target_layer}")
    print(f"  total_steps={args.total_steps}  lr={args.lr}  batch_seqs={args.batch_seqs}")
    print(f"  T: {args.T_init} -> {args.T_min} (exponential)")
    print(f"  reencode_every={args.reencode_every}  bcjr_chunk={args.bcjr_chunk}")
    print("=" * 70, flush=True)

    # ---------------- calibration ----------------
    tokens = np.load(args.calib)
    print(f"\nCalibration: {tokens.shape} {tokens.dtype}")
    n_seqs, seq_len = tokens.shape
    tokens_t = torch.from_numpy(tokens.astype(np.int64))

    # ---------------- codebook ----------------
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # ---------------- teacher ----------------
    print("\nLoading FP16 teacher...", flush=True)
    t0 = time.time()
    teacher = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  teacher loaded in {time.time() - t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # ---------------- student ----------------
    print("\nLoading student base (fp32)...", flush=True)
    t0 = time.time()
    student = OlmoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cuda",
    )
    print(f"  student loaded in {time.time() - t0:.0f}s", flush=True)

    # ---------------- install v2 PTQ into all layers ----------------
    # This installs v2 weights into the raw nn.Linear modules of every layer.
    # For the target layer, these values get overwritten below when we wrap
    # with QATDecoderLayer (which replaces Linear with QuantizedLinear and
    # installs v2 into W_latent instead).
    print(f"\nInstalling v2 PTQ weights into all 16 layers...", flush=True)
    t0 = time.time()
    install_bcjr_weights(student, args.v2_dir, verbose=False)
    print(f"  installed in {time.time() - t0:.0f}s", flush=True)

    # ---------------- wrap target layer ----------------
    print(f"\nWrapping layer {args.target_layer} with QATDecoderLayer + BCJR...",
          flush=True)
    t0 = time.time()
    cfg = student.config
    fp_layer = student.model.layers[args.target_layer]
    qat_layer = QATDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=args.seed + args.target_layer * 997,
        reencode_every_n_steps=args.reencode_every,
    )
    snap = _load_v2_snapshot(args.v2_dir, args.target_layer)
    _install_snapshot_into_qat_layer(qat_layer, snap,
                                     intermediate_size=cfg.intermediate_size)
    del snap
    student.model.layers[args.target_layer] = qat_layer

    # ---------------- freeze everything except target layer's W_latent ----------------
    for _, p in student.named_parameters():
        p.requires_grad_(False)
    for p in qat_layer.trainable_parameters():
        p.requires_grad_(True)

    # ---------------- flip target layer to BCJR ----------------
    convert_layer_to_bcjr(qat_layer, T_init=args.T_init, bcjr_chunk=args.bcjr_chunk)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  wrapped in {time.time() - t0:.0f}s  "
          f"trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}%)", flush=True)
    print(f"  student GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # ---------------- optimizer ----------------
    trainable = [p for p in student.parameters() if p.requires_grad]
    if args.use_adam8:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
            print(f"  optimizer: AdamW8bit  ({len(trainable)} tensors)", flush=True)
        except ImportError:
            print("  bitsandbytes not available, falling back to fp32 AdamW",
                  flush=True)
            opt = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"  peak VRAM after opt init: "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    # ---------------- one-shot heartbeat for first step ----------------
    _heartbeat_handles = []
    _heartbeat_t0 = [None]
    def _make_pre(idx):
        def _h(module, *_a, **_kw):
            if _heartbeat_t0[0] is None:
                _heartbeat_t0[0] = time.time()
            dt = time.time() - _heartbeat_t0[0]
            mem = torch.cuda.memory_allocated() / 1e9
            tag = "L*" if idx == args.target_layer else "L "
            print(f"  [hb {dt:5.1f}s  mem={mem:5.1f}GB] enter {tag}{idx}",
                  flush=True)
        return _h
    def _make_post(idx):
        def _h(module, *_a, **_kw):
            dt = time.time() - _heartbeat_t0[0]
            mem = torch.cuda.memory_allocated() / 1e9
            tag = "L*" if idx == args.target_layer else "L "
            print(f"  [hb {dt:5.1f}s  mem={mem:5.1f}GB] exit  {tag}{idx}",
                  flush=True)
        return _h
    for i, layer in enumerate(student.model.layers):
        _heartbeat_handles.append(layer.register_forward_pre_hook(_make_pre(i)))
        _heartbeat_handles.append(layer.register_forward_hook(_make_post(i)))

    # ---------------- training loop ----------------
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70, flush=True)

    t_train0 = time.time()
    losses = []

    for step in range(args.total_steps):
        T_now = exp_temperature_schedule(step, args.total_steps,
                                         args.T_init, args.T_min)
        set_layer_temperature(qat_layer, T_now)

        idx = torch.randperm(n_seqs)[:args.batch_seqs]
        batch = tokens_t[idx].to("cuda", non_blocking=True)

        t_step = time.time()
        if step <= 2:
            print(f"\n--- step {step} start ---", flush=True)

        with torch.no_grad():
            t_out = teacher(input_ids=batch, use_cache=False).logits
        if step <= 2:
            print(f"  teacher forward: {time.time()-t_step:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        s_out = student(input_ids=batch, use_cache=False).logits
        if step <= 2:
            print(f"  student forward: {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        loss = kl_loss_full_vocab(s_out, t_out)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step <= 2:
            print(f"  backward:        {time.time()-t_phase:.1f}s  "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
            t_phase = time.time()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        opt.step()
        if step <= 2:
            print(f"  opt.step:        {time.time()-t_phase:.1f}s  "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                  flush=True)

        step_dt = time.time() - t_step
        losses.append(float(loss.item()))

        if (step + 1) % args.log_every == 0:
            elapsed_total = time.time() - t_train0
            rate = (step + 1) / elapsed_total
            eta_min = (args.total_steps - step - 1) / rate / 60
            print(f"  step {step+1}/{args.total_steps}  "
                  f"kl={loss.item():.4e}  T={T_now:.3e}  "
                  f"dt={step_dt:.1f}s  ETA {eta_min:.0f}min", flush=True)

        # Remove heartbeat hooks after step 0 to stop spam
        if step == 0:
            for h in _heartbeat_handles:
                h.remove()
            _heartbeat_handles.clear()

        # Save checkpoint
        if (step + 1) % args.ckpt_every == 0:
            ckpt_snap = qat_layer.snapshot_W_q_cpu()
            ckpt_path = os.path.join(args.out_dir,
                                     f"layer_{args.target_layer:02d}_wq_step{step+1}.pt")
            torch.save(ckpt_snap, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

    # ---------------- final save ----------------
    print("\n" + "=" * 70)
    print("Finalizing: flip target layer to STE + prime + save...")
    print("=" * 70, flush=True)
    from src.bcjr.anneal import convert_layer_to_ste
    convert_layer_to_ste(qat_layer)
    qat_layer.prime_cache()
    final_snap = qat_layer.snapshot_W_q_cpu()
    final_path = os.path.join(args.out_dir,
                              f"layer_{args.target_layer:02d}_wq_final.pt")
    torch.save(final_snap, final_path)
    print(f"  saved final W_q to {final_path}", flush=True)
    print(f"\n  train loss trajectory (first/last): "
          f"{losses[0]:.4e} -> {losses[-1]:.4e}", flush=True)
    print(f"  wall time: {(time.time() - t_train0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
