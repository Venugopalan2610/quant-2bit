"""Llama-3.2-1B full-model BCJR-QAT via per-layer greedy sequential assembly.

Rung 1 of the zero-budget full-model experiment.

Trains each layer in order 0→15. After each layer is trained, its W_q is
installed back into the student as a frozen bf16 nn.Linear — so the NEXT
layer trains with all prior layers already at their quantized activations.
Only ONE layer has fp32 W_latent + optimizer state at any moment.

Memory budget at batch=1, seq=1024 on RTX 4080 (12 GB):
  Teacher FP16:                    ~2.5 GB  (resident throughout)
  Student BF16 frozen layers:      ~2.0 GB
  One fp32 QAT layer (W_latent):   ~0.25 GB
  AdamW8bit state:                 ~0.25 GB
  Activations (16-layer backward): ~0.5 GB
  BCJR Triton workspace:           ~0.1 GB
  ─────────────────────────────────────────
  Peak estimate: ~5.6 GB — fits on 4080.

  NOTE: If you hit OOM, try --use-adam8 (reduces optimizer state) or
  reduce --seq-len to 512.

Output: cache/llama_bcjr_full_greedy/
  layer_NN_wq.pt        CPU fp32 W_q dict (7 projections per layer)
  layer_NN_stats.pt     loss trajectory + timing
  aggregate_stats.pt    all layers combined
  full_model_eval.json  WikiText-2 PPL (written after training)
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.qat.qat_dense_decoder_layer import (
    QATDenseDecoderLayer, ATTN_PROJ_NAMES, MLP_PROJ_NAMES,
)
from src.bcjr.anneal import convert_layer_to_bcjr, convert_layer_to_ste, set_layer_temperature
from src.qat.train_e2e_kl import kl_loss_full_vocab, exp_temperature_schedule
from src.qat.eval_llama_layer import eval_ppl_wikitext, install_layer_weights


NUM_LAYERS = 16


def _install_snap_as_frozen_layer(student, layer_idx, snap, dtype=torch.bfloat16):
    """Replace trained QuantizedLinear projections with frozen bf16 nn.Linear.

    After training layer `layer_idx`, student.model.layers[layer_idx] is a
    QATDenseDecoderLayer whose inner projections are QuantizedLinear (fp32).
    This function:
      1. Replaces each QuantizedLinear with a new frozen nn.Linear (bf16, W_q).
      2. Casts remaining fp32 tensors (layernorms) back to bf16.
      3. Reinstalls the inner LlamaDecoderLayer directly, freeing the wrapper.

    Postcondition: student.model.layers[layer_idx] is a plain LlamaDecoderLayer
    with bf16 frozen quantized weights. The QATDenseDecoderLayer and its fp32
    buffers are freed on the next gc.collect() / cuda.empty_cache().
    """
    qat_wrap = student.model.layers[layer_idx]
    assert isinstance(qat_wrap, QATDenseDecoderLayer), (
        f"Expected QATDenseDecoderLayer at layer {layer_idx}, got {type(qat_wrap)}"
    )
    inner = qat_wrap.layer  # the fp32 LlamaDecoderLayer

    for name in ATTN_PROJ_NAMES:
        W_q = snap[f"attn_{name}"]      # CPU fp32, shape (out, in)
        out_f, in_f = W_q.shape
        new_lin = torch.nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=dtype)
        new_lin.weight.data.copy_(W_q.to(dtype=dtype))
        new_lin.weight.requires_grad_(False)
        setattr(inner.self_attn, name, new_lin)

    for name in MLP_PROJ_NAMES:
        W_q = snap[f"mlp_{name}"]
        out_f, in_f = W_q.shape
        new_lin = torch.nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=dtype)
        new_lin.weight.data.copy_(W_q.to(dtype=dtype))
        new_lin.weight.requires_grad_(False)
        setattr(inner.mlp, name, new_lin)

    # Cast layernorms and any remaining fp32 params back to bf16
    inner.bfloat16()

    # Reinstall inner layer; the QATDenseDecoderLayer wrapper (and its fp32
    # buffers) will be freed on the next gc.collect() + cuda.empty_cache().
    student.model.layers[layer_idx] = inner
    del qat_wrap


def train_one_layer(student, teacher, tokens_t, cb, layer_idx, args):
    """Wrap, train, finalize, and snapshot one decoder layer.

    Returns (snap, stats):
      snap  — CPU fp32 dict of W_q tensors (7 keys, one per projection)
      stats — dict with loss trajectory and wall-clock timing
    """
    cfg = student.config
    n_seqs = tokens_t.shape[0]

    print(f"\n{'='*70}", flush=True)
    print(f"Layer {layer_idx:02d}/{NUM_LAYERS-1}  "
          f"GPU before wrap: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Cast this layer to fp32 and wrap with QATDenseDecoderLayer.
    # .float() is in-place and returns self, so fp_layer IS the layer object.
    fp_layer = student.model.layers[layer_idx].float()
    seed_layer = args.seed + layer_idx * 997
    qat_layer = QATDenseDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=seed_layer,
        reencode_every_n_steps=args.reencode_every,
    )
    student.model.layers[layer_idx] = qat_layer

    # Optionally anchor W_latent to hard-Viterbi PTQ codes so training starts
    # exactly at the PTQ baseline. Any final PPL delta = pure BCJR effect.
    if args.init_from_ptq:
        with torch.no_grad():
            for _, ql in qat_layer._all_qls():
                W_q = TrellisQuantSTE.apply(
                    ql.W_latent.float(), ql.sign_l, ql.sign_r, ql.codebook_gpu,
                )
                ql.W_latent.copy_(W_q.to(ql.W_latent.dtype))
                ql._W_q_cache.copy_(W_q.to(ql._W_q_cache.dtype))

    # Freeze all params, then unfreeze only this layer's W_latents.
    for _, p in student.named_parameters():
        p.requires_grad_(False)
    for p in qat_layer.trainable_parameters():
        p.requires_grad_(True)

    convert_layer_to_bcjr(qat_layer, T_init=args.T_init, bcjr_chunk=args.bcjr_chunk)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  trainable: {n_train/1e6:.2f}M  "
          f"GPU after wrap: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    trainable = list(qat_layer.trainable_parameters())
    if args.use_adam8:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
            print(f"  optimizer: AdamW8bit  lr={args.lr}", flush=True)
        except ImportError:
            opt = torch.optim.AdamW(trainable, lr=args.lr)
            print(f"  optimizer: AdamW (bitsandbytes missing)  lr={args.lr}", flush=True)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
        print(f"  optimizer: AdamW  lr={args.lr}", flush=True)

    print(f"  peak VRAM after opt init: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    losses = []
    t_layer = time.time()

    for step in range(args.total_steps):
        T_now = exp_temperature_schedule(step, args.total_steps, args.T_init, args.T_min)
        set_layer_temperature(qat_layer, T_now)

        idx = torch.randperm(n_seqs)[:args.batch_seqs]
        batch = tokens_t[idx].to("cuda", non_blocking=True)

        t_step = time.time()

        # Teacher forward: no_grad, produces only final logits (small tensor).
        # Teacher model stays in VRAM (~2.5 GB) throughout — fine for 4080.
        with torch.no_grad():
            t_out = teacher(input_ids=batch, use_cache=False).logits

        # Student forward with grad flowing through all 16 layers.
        # Only qat_layer's W_latent has requires_grad=True.
        s_out = student(input_ids=batch, use_cache=False).logits
        loss = kl_loss_full_vocab(s_out, t_out)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        opt.step()

        step_dt = time.time() - t_step
        losses.append(float(loss.item()))
        print(f"  step {step+1}/{args.total_steps}  "
              f"kl={loss.item():.4e}  T={T_now:.3e}  "
              f"dt={step_dt/60:.1f}min  "
              f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.1f} GB",
              flush=True)

    # Finalize: hard-Viterbi encode from final W_latent and snapshot.
    convert_layer_to_ste(qat_layer)
    qat_layer.prime_cache()
    snap = qat_layer.snapshot_W_q_cpu()   # CPU fp32 {attn_q_proj: ..., ...}

    # Free optimizer and trainable list before next layer's init.
    del opt, trainable
    gc.collect()
    torch.cuda.empty_cache()

    layer_time = time.time() - t_layer
    print(f"  Layer {layer_idx:02d} done: {layer_time/60:.1f} min  "
          f"loss {losses[0]:.4e} → {losses[-1]:.4e}", flush=True)

    return snap, {
        "layer_idx": layer_idx,
        "losses": losses,
        "seconds": layer_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/model/Llama-3.2-1B",
                        help="HF model path or Hub ID for Llama-3.2-1B")
    parser.add_argument("--calib", default="cache/calibration/tokens_llama.npy")
    parser.add_argument("--out-dir", default="cache/llama_bcjr_full_greedy")
    parser.add_argument("--start-layer", type=int, default=0,
                        help="Resume from this layer. Layers 0..(start-1) must "
                             "have layer_NN_wq.pt in --out-dir already.")
    parser.add_argument("--end-layer", type=int, default=NUM_LAYERS,
                        help="Exclusive upper bound. Default 16 = all layers.")
    parser.add_argument("--total-steps", type=int, default=3,
                        help="BCJR-QAT steps per layer. 3 matches the paper's "
                             "single-layer result; 10 is the skip-high-T run. "
                             "Check timing after layer 0 before committing.")
    parser.add_argument("--batch-seqs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T-init", type=float, default=0.3,
                        help="Skip-high-T starting temperature (paper finding: "
                             "T0=0.3 avoids early overshoot). Do not raise to 1.0.")
    parser.add_argument("--T-min", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bcjr-chunk", type=int, default=16)
    parser.add_argument("--reencode-every", type=int, default=1,
                        help="Re-run BCJR every N steps. 1 = every step (pure BCJR gradient).")
    parser.add_argument("--init-from-ptq", action="store_true",
                        help="Anchor W_latent to hard-Viterbi PTQ at each layer's start "
                             "so ΔPPL vs PTQ = pure BCJR training effect.")
    parser.add_argument("--use-adam8", action="store_true",
                        help="Use bitsandbytes AdamW8bit (~saves 0.25 GB per layer).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seq-len", type=int, default=2048,
                        help="Sequence length for final WikiText-2 eval.")
    parser.add_argument("--eval-max-windows", type=int, default=None,
                        help="Cap eval windows for faster approximate eval.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip final WikiText-2 PPL eval. Run separately "
                             "with scripts/rung3_comparison.sh later.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ["BCJR_TRITON"] = "1"
    os.environ["BCJR_MONOLITHIC"] = "1"

    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Llama-3.2-1B  full-model BCJR-QAT  (per-layer greedy sequential)")
    print(f"  model:        {args.model}")
    print(f"  layers:       {args.start_layer} → {args.end_layer - 1}")
    print(f"  steps/layer:  {args.total_steps}  lr={args.lr}")
    print(f"  T schedule:   {args.T_init} → {args.T_min}  (skip-high-T)")
    print(f"  init_from_ptq={args.init_from_ptq}  adam8={args.use_adam8}")
    print(f"  out_dir:      {args.out_dir}")
    print("=" * 70, flush=True)

    print("\nHonest label: 'per-layer greedy assembly', NOT 'joint end-to-end'.")
    print("Super-additivity at scale is tested by Rung 2 (joint, H100+, DEFERRED).")
    print(flush=True)

    # Calibration tokens (C4, tokenized for Llama)
    tokens = np.load(args.calib)
    print(f"Calibration: {tokens.shape} {tokens.dtype}", flush=True)
    n_seqs, seq_len_calib = tokens.shape
    if args.seq_len > seq_len_calib:
        print(f"  capping seq_len {args.seq_len} → {seq_len_calib}")
        args.seq_len = seq_len_calib
    tokens_t = torch.from_numpy(tokens[:, :args.seq_len].astype(np.int64))

    # Codebook — shared across all layers (same codebook, different sign vectors)
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # Teacher: frozen FP16, stays resident throughout all 16 layers.
    # Llama-3.2-1B at FP16 ≈ 2.5 GB — comfortably coexists with student on 4080.
    print("\nLoading teacher (fp16)...", flush=True)
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.time()-t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Student: BF16, modified layer-by-layer in the training loop.
    print("\nLoading student (bf16)...", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    print(f"  loaded in {time.time()-t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Resume: install already-trained W_q for layers before start_layer.
    # install_layer_weights copies into existing nn.Linear.weight.data (bf16 student).
    if args.start_layer > 0:
        print(f"\nResuming from layer {args.start_layer}: "
              f"installing snapshots for layers 0..{args.start_layer-1}", flush=True)
        for layer_idx in range(args.start_layer):
            snap_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_wq.pt")
            if not os.path.exists(snap_path):
                print(f"FAIL: snapshot missing: {snap_path}")
                sys.exit(1)
            snap = torch.load(snap_path, map_location="cpu", weights_only=True)
            install_layer_weights(student, layer_idx, snap, dtype=torch.bfloat16)
            print(f"  layer {layer_idx:02d}: installed from {snap_path}", flush=True)

    # Main training loop: one layer at a time.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_stats = []
    grand_start = time.time()

    for layer_idx in range(args.start_layer, args.end_layer):
        snap, stats = train_one_layer(student, teacher, tokens_t, cb, layer_idx, args)

        # Save W_q snapshot to disk before installing (so we can resume if interrupted).
        snap_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_wq.pt")
        torch.save(snap, snap_path)
        stats_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_stats.pt")
        torch.save(stats, stats_path)
        print(f"  saved: {snap_path}", flush=True)

        # Install trained W_q back into the student as frozen bf16. This is the
        # "greedy sequential" mechanism: each subsequent layer trains in the
        # context of all prior layers already at their quantized activations.
        _install_snap_as_frozen_layer(student, layer_idx, snap)
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  GPU after install: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB  "
              f"(peak so far: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)",
              flush=True)

        all_stats.append(stats)

        elapsed = time.time() - grand_start
        done = layer_idx - args.start_layer + 1
        remaining = args.end_layer - layer_idx - 1
        if remaining > 0:
            eta_min = (elapsed / done) * remaining / 60
            print(f"  ETA: {eta_min:.0f} min ({remaining} layers left)", flush=True)

    # Aggregate stats
    torch.save({
        "layer_stats": all_stats,
        "args": vars(args),
        "total_train_seconds": time.time() - grand_start,
    }, os.path.join(args.out_dir, "aggregate_stats.pt"))

    train_time = time.time() - grand_start
    print(f"\n{'='*70}")
    print(f"Training done: {args.end_layer - args.start_layer} layers in "
          f"{train_time/60:.1f} min")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*70}", flush=True)

    if args.skip_eval:
        print("\n--skip-eval: run comparison table with scripts/rung3_comparison.sh")
        return

    # Final WikiText-2 PPL on the fully-quantized student
    print("\nFinal WikiText-2 PPL eval (fully quantized student)...", flush=True)
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()

    ppl = eval_ppl_wikitext(
        student, tokenizer,
        seq_len=args.eval_seq_len,
        max_windows=args.eval_max_windows,
    )

    result = {
        "model": args.model,
        "method": "bcjr_qat_greedy_sequential",
        "bits": 2,
        "layers_trained": list(range(args.start_layer, args.end_layer)),
        "wikitext2_ppl": ppl,
        "eval_seq_len": args.eval_seq_len,
        "args": vars(args),
        "total_train_seconds": train_time,
    }
    result_path = os.path.join(args.out_dir, "full_model_eval.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"FULL-MODEL 2-bit BCJR-QAT (greedy sequential)  WikiText-2 PPL: {ppl:.4f}")
    print(f"Saved: {result_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
