"""Matched-budget QAT comparison arms — the controlled trellis-vs-scalar test.

Same harness, same data, same steps/LR/seed/grad-clip as the BCJR single-layer
runner (src/qat/train_llama_single_layer.py); the ONLY thing that changes is
the quantizer arm. This is the experiment that decides whether the trellis
*representation* is worth anything — the comparison the BCJR paper never ran
(it only beat PTQ, i.e. trellis-trained > trellis-untrained, never trellis vs
scalar). See PREREGISTRATION.md.

Arms (--arm):
  ste_trellis      : trellis codebook, identity-STE training (no soft relax).
  scalar_ctrl      : uniform n-bit in the SAME RHT basis + global scale, only
                     the quantizer differs from ste_trellis. Exactly n_bits/w.
                     Near-Lloyd-Max optimal scalar → no-excuses baseline.
  scalar_faithful  : ParetoQ-style native-basis per-group LEARNED clip. Costs
                     scale-overhead bits (reported honestly; ~2.125b @group128).

Init (--init): fp (default — all arms start from identical FP weights, the
clean isolation) or ptq (start from trellis hard-Viterbi codes, to init-match
the existing BCJR-from-PTQ result).

No temperature: scalar/STE arms have no soft relaxation, so there is no T
schedule — "matched budget" means matched data, steps, LR, seed, grad-clip.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.bcjr.anneal import convert_layer_to_ste, convert_layer_to_scalar
from src.qat.scalar_quant import effective_bits_faithful
from src.qat.train_e2e_kl import kl_loss_full_vocab


ARMS = ("ste_trellis", "scalar_ctrl", "scalar_faithful")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cache/model/Llama-3.2-1B")
    p.add_argument("--calib", default="cache/calibration/tokens_llama.npy")
    p.add_argument("--out-dir", default="cache/llama_matched")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--init", choices=("fp", "ptq"), default="fp")
    p.add_argument("--target-layer", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=10)
    p.add_argument("--batch-seqs", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-bits", type=int, default=2, help="sub-2-bit pivot: set 1")
    p.add_argument("--group-size", type=int, default=128, help="scalar_faithful only")
    p.add_argument("--use-adam8", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    os.environ["BCJR_TRITON"] = "1"
    os.environ["BCJR_MONOLITHIC"] = "1"

    print("=" * 70)
    print(f"Matched-arm QAT: {args.arm}  init={args.init}  L{args.target_layer}")
    print(f"  steps={args.total_steps} seq_len={args.seq_len} lr={args.lr} "
          f"seed={args.seed} n_bits={args.n_bits}")
    if args.arm == "scalar_faithful":
        # bit accounting depends on the wrapped layer's shapes; print q_proj as
        # a representative once we know dims (below). For now flag the group.
        print(f"  group_size={args.group_size} (effective bits printed after wrap)")
    print("=" * 70, flush=True)

    tokens = np.load(args.calib)
    n_seqs, seq_len_calib = tokens.shape
    if args.seq_len > seq_len_calib:
        args.seq_len = seq_len_calib
    tokens_t = torch.from_numpy(tokens[:, :args.seq_len].astype(np.int64))
    print(f"Calibration: {tokens.shape} {tokens.dtype}", flush=True)

    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    cb = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    print("\nLoading teacher (fp16)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda")
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print("Loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")

    cfg = student.config
    print(f"\nWrapping layer {args.target_layer} (seed-matched to BCJR run)...",
          flush=True)
    fp_layer = student.model.layers[args.target_layer].float()
    # IDENTICAL seed derivation to train_llama_single_layer.py → IDENTICAL RHT
    # sign vectors → the scalar_ctrl arm quantizes in the exact same basis as
    # the trellis arm (pre-mortem fix #9: matched basis).
    qat_layer = QATDenseDecoderLayer(
        fp_layer, codebook_gpu=cb, config=cfg,
        seed=args.seed + args.target_layer * 997,
        reencode_every_n_steps=1,
    )
    student.model.layers[args.target_layer] = qat_layer

    # ---- init ----
    if args.init == "ptq":
        print("Init W_latent from hard-Viterbi PTQ codes...", flush=True)
        with torch.no_grad():
            for name, ql in qat_layer._all_qls():
                W_q = TrellisQuantSTE.apply(
                    ql.W_latent.float(), ql.sign_l, ql.sign_r, ql.codebook_gpu)
                ql.W_latent.copy_(W_q.to(ql.W_latent.dtype))
    # else: FP init — W_latent already holds the FP weights from the wrap.

    # ---- select arm ----
    if args.arm == "ste_trellis":
        convert_layer_to_ste(qat_layer)              # trellis + identity-STE
    else:
        convert_layer_to_scalar(qat_layer, args.arm,
                                group_size=args.group_size, n_bits=args.n_bits)
        if args.arm == "scalar_faithful":
            # Honest effective bits/weight (counts per-group clip overhead),
            # reported per projection next to the trellis arm's clean n_bits.
            for name, ql in qat_layer._all_qls():
                eb = effective_bits_faithful(ql.out_features, ql.in_features,
                                             args.group_size, n_bits=args.n_bits)
                print(f"  {name}: effective bits = {eb:.4f} "
                      f"(trellis = {args.n_bits}.0000)", flush=True)

    # freeze all but the target layer's trainable params (W_latent [+ clip])
    for _, prm in student.named_parameters():
        prm.requires_grad_(False)
    trainable = qat_layer.trainable_parameters()
    for prm in trainable:
        prm.requires_grad_(True)
    n_train = sum(t.numel() for t in trainable)
    print(f"\ntrainable tensors={len(trainable)}  params={n_train/1e6:.1f}M",
          flush=True)

    if args.use_adam8:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)

    print("\n" + "=" * 70 + "\nStarting training\n" + "=" * 70, flush=True)
    t_train0 = time.time()
    for step in range(args.total_steps):
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
        dt = (time.time() - t_step) / 60.0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  step {step+1}/{args.total_steps}  kl={loss.item():.4e}  "
              f"dt={dt:.2f}min  peak_vram={peak:.1f}GB", flush=True)

    # ---- harden + snapshot (same format the eval harness consumes) ----
    print("\nHardening + saving snapshot...", flush=True)
    qat_layer.prime_cache()
    snap = qat_layer.snapshot_W_q_cpu()
    pad = f"{args.target_layer:02d}"
    out = os.path.join(args.out_dir,
                       f"layer_{pad}_{args.arm}_{args.init}_seed{args.seed}.pt")
    torch.save(snap, out)
    print(f"  saved {out}", flush=True)
    print(f"  total train time {(time.time()-t_train0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
