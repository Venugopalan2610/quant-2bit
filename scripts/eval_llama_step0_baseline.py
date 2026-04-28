"""Measure the BCJR pipeline's step-0 (untrained) PPL to test whether it
matches the production PTQ baseline.

If step-0 PPL == production PTQ PPL → the BCJR pipeline reproduces PTQ
exactly at initialization, and the 0.04 PPL gap observed at step-2 is
training drift (not RHT-seed variance). This is the bit-identical test.

Outputs:
  - PPL with target layer wrapped in QATDenseDecoderLayer + --init-from-ptq
    applied + hardened immediately (no training)
  - Compared to PTQ-via-eval_llama_layer for the same layer
  - Compared to FP baseline

Runs on the user's 4080 in ~5-10 min total. No cloud compute required.

Usage:
  python -m scripts.eval_llama_step0_baseline --target-layer 4
  python -m scripts.eval_llama_step0_baseline --target-layers 4,8
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.qat.eval_llama_layer import (
    eval_ppl_wikitext, install_layer_weights, hard_viterbi_quantize,
    ATTN_PROJ_NAMES, MLP_PROJ_NAMES,
)


def stash_layer(layer):
    return {
        **{f"attn_{n}": getattr(layer.self_attn, n).weight.data.detach().clone()
           for n in ATTN_PROJ_NAMES},
        **{f"mlp_{n}": getattr(layer.mlp, n).weight.data.detach().clone()
           for n in MLP_PROJ_NAMES},
    }


def restore_layer(layer, stash):
    for n in ATTN_PROJ_NAMES:
        getattr(layer.self_attn, n).weight.data.copy_(stash[f"attn_{n}"])
    for n in MLP_PROJ_NAMES:
        getattr(layer.mlp, n).weight.data.copy_(stash[f"mlp_{n}"])


def make_seed_offsets(layer_idx, seed):
    base_seed = seed + layer_idx * 997
    return {
        ("attn", "q_proj"): base_seed + 1_000_000 + 0 * 17,
        ("attn", "k_proj"): base_seed + 1_000_000 + 1 * 17,
        ("attn", "v_proj"): base_seed + 1_000_000 + 2 * 17,
        ("attn", "o_proj"): base_seed + 1_000_000 + 3 * 17,
        ("mlp",  "gate_proj"): base_seed + 2_000_000 + 0 * 31,
        ("mlp",  "up_proj"):   base_seed + 2_000_000 + 1 * 31,
        ("mlp",  "down_proj"): base_seed + 2_000_000 + 2 * 31,
    }


def step0_w_q_via_qat_pipeline(target_layer, codebook, layer_idx, cfg, seed):
    """Build the BCJR pipeline's step-0 W_q snap via fresh
    QATDenseDecoderLayer + --init-from-ptq logic. This mirrors what training
    does at iteration 0 BEFORE any opt.step() happens."""
    fp_layer_for_fixture = copy.deepcopy(target_layer).float()
    qat_fixture = QATDenseDecoderLayer(
        fp_layer_for_fixture, codebook_gpu=codebook, config=cfg,
        seed=seed + layer_idx * 997, reencode_every_n_steps=1,
    ).cuda()

    snap = {}
    with torch.no_grad():
        for name, ql in qat_fixture._all_qls():
            # ql.W_latent is currently W_FP (from_linear loaded it).
            # Apply the --init-from-ptq operation: hard Viterbi.
            W_q = TrellisQuantSTE.apply(
                ql.W_latent.float(), ql.sign_l, ql.sign_r, ql.codebook_gpu,
            )
            # Map name to snap key (handles both "attn_X" and "X" conventions)
            if name.startswith("attn_") or name.startswith("mlp_"):
                snap[name] = W_q.detach().cpu()
            elif name in ATTN_PROJ_NAMES:
                snap[f"attn_{name}"] = W_q.detach().cpu()
            elif name in MLP_PROJ_NAMES:
                snap[f"mlp_{name}"] = W_q.detach().cpu()
            else:
                raise ValueError(f"can't map QL name {name!r} to snap key")
    return snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/model/Llama-3.2-1B")
    parser.add_argument("--target-layers", default="4,8",
                        help="Comma-separated layer indices to test")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Cap windows for faster eval (e.g. 30 for ~1 min/eval)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppl-fp-cached", type=float, default=None)
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    print("=" * 70)
    print(f"Step-0 baseline test: BCJR pipeline at iteration 0 (no training)")
    print(f"  target_layers: {target_layers}")
    print(f"  seq_len: {args.seq_len}  max_windows: {args.max_windows}")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    codebook = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    print("\nLoading FP model (fp16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    cfg = model.config

    # FP baseline
    if args.ppl_fp_cached is not None:
        ppl_fp = args.ppl_fp_cached
        print(f"\n[a] FP baseline (cached): {ppl_fp:.4f}")
    else:
        print("\n[a] FP baseline (no quantization)", flush=True)
        ppl_fp = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        print(f"  PPL = {ppl_fp:.4f}")

    # For each target layer, compare:
    #   (b) Production PTQ via eval_llama_layer.hard_viterbi_quantize
    #   (c) Step-0 BCJR pipeline (fresh fixture + --init-from-ptq, no training)
    summary = []
    for L in target_layers:
        print(f"\n{'='*70}\nLayer {L}\n{'='*70}", flush=True)
        target = model.model.layers[L]
        fp_stash = stash_layer(target)

        # (b) Production PTQ baseline (eval_llama_layer pipeline)
        print(f"\n[b] Production PTQ at layer {L} (eval_llama_layer pipeline)", flush=True)
        seed_offsets = make_seed_offsets(L, args.seed)
        def hv_fn(kind, name, W_fp):
            return hard_viterbi_quantize(W_fp, codebook, seed=seed_offsets[(kind, name)])
        install_layer_weights(model, L, hv_fn, dtype=torch.float16)
        ppl_prod = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        print(f"  PPL_PTQ_production = {ppl_prod:.4f}")
        restore_layer(target, fp_stash)

        # (c) Step-0 BCJR pipeline (fresh QATDenseDecoderLayer + init_from_ptq, hardened)
        print(f"\n[c] Step-0 BCJR pipeline at layer {L} (fresh fixture, --init-from-ptq, hardened)", flush=True)
        snap_step0 = step0_w_q_via_qat_pipeline(target, codebook, L, cfg, args.seed)
        install_layer_weights(model, L, snap_step0, dtype=torch.float16)
        ppl_step0 = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        print(f"  PPL_BCJR_step0 = {ppl_step0:.4f}")
        restore_layer(target, fp_stash)

        delta = ppl_step0 - ppl_prod
        summary.append((L, ppl_prod, ppl_step0, delta))
        print(f"\n  Δ (BCJR_step0 − Production_PTQ) = {delta:+.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print(f"Summary  (FP baseline = {ppl_fp:.4f})")
    print("-" * 70)
    print(f"  {'layer':>5} {'Production PTQ':>15} {'BCJR step-0':>13} {'Δ':>10}")
    for L, p_prod, p_step0, d in summary:
        print(f"  {L:>5} {p_prod:>15.4f} {p_step0:>13.4f} {d:>+10.4f}")
    print()
    avg_delta = sum(d for _, _, _, d in summary) / len(summary)
    print(f"  Average |Δ|: {abs(avg_delta):.4f}")
    if abs(avg_delta) < 0.005:
        print(f"  ✓ Pipelines are bit-equivalent at step 0.")
        print(f"    The step-2 PPL gap is BCJR training drift, not pipeline difference.")
        print(f"    Therefore step-10 BCJR comparisons should be made vs Production PTQ,")
        print(f"    NOT vs step-2 in-pipeline.")
    else:
        print(f"  ✗ Pipelines differ at step 0 by ~{abs(avg_delta):.4f} PPL.")
        print(f"    The within-pipeline (step-2 vs step-10) comparison is valid.")
        print(f"    BCJR-trained gain over the BCJR pipeline's PTQ-equivalent")
        print(f"    is the method's true effect; cross-pipeline comparison is confounded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
