"""Eval Llama-3.2-1B with BCJR snapshots at MULTIPLE target layers.

Tests whether per-layer BCJR-QAT wins compound when multiple layers are
trained independently and installed simultaneously.

Compares four configurations (loads model ONCE, ~$1 of compute):
  (a) FP baseline (no quantization)
  (b) Multi-layer hard-Viterbi PTQ at all target layers
  (c) Multi-layer BCJR-trained snapshots at all target layers
  (d) Per-layer single-layer BCJR (for breakdown)

Usage:
  python -m scripts.eval_llama_multilayer \\
      --target-layers 4,8 \\
      --snapshots cache/llama_bcjr_lr10x_L4/layer_04_wq_ptqinit.pt,cache/llama_bcjr_lr10x/layer_08_wq_ptqinit.pt \\
      --output results/llama_multilayer_compounding.json \\
      --ppl-fp-cached 9.70

Decision rules:
  PPL_multi_bcjr < PPL_multi_ptq → COMPOUNDING (paper claim validated)
  PPL_multi_bcjr ≈ PPL_multi_ptq → no compounding (interference cancels gains)
  PPL_multi_bcjr > PPL_multi_ptq → ANTI-COMPOUNDING (per-layer wins interfere)
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
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
    """Same seed scheme as eval_llama_layer.py and train_llama_single_layer.py."""
    base_seed = seed + layer_idx * 997
    return {
        ("attn", "q_proj"):    base_seed + 1_000_000 + 0 * 17,
        ("attn", "k_proj"):    base_seed + 1_000_000 + 1 * 17,
        ("attn", "v_proj"):    base_seed + 1_000_000 + 2 * 17,
        ("attn", "o_proj"):    base_seed + 1_000_000 + 3 * 17,
        ("mlp",  "gate_proj"): base_seed + 2_000_000 + 0 * 31,
        ("mlp",  "up_proj"):   base_seed + 2_000_000 + 1 * 31,
        ("mlp",  "down_proj"): base_seed + 2_000_000 + 2 * 31,
    }


def install_ptq_at_layer(model, layer_idx, codebook, seed):
    seed_offsets = make_seed_offsets(layer_idx, seed)
    def hv_fn(kind, name, W_fp):
        return hard_viterbi_quantize(W_fp, codebook,
                                     seed=seed_offsets[(kind, name)])
    install_layer_weights(model, layer_idx, hv_fn, dtype=torch.float16)


def install_bcjr_snap_at_layer(model, layer_idx, snap_path):
    snap = torch.load(snap_path, map_location="cuda", weights_only=True)
    install_layer_weights(model, layer_idx, snap, dtype=torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/model/Llama-3.2-1B")
    parser.add_argument("--target-layers", required=True,
                        help="Comma-separated layer indices, e.g. '4,8'")
    parser.add_argument("--snapshots", required=True,
                        help="Comma-separated snap paths, in same order as --target-layers")
    parser.add_argument("--output", default="results/llama_multilayer_compounding.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppl-fp-cached", type=float, default=None)
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    snapshot_paths = [p.strip() for p in args.snapshots.split(",")]
    if len(target_layers) != len(snapshot_paths):
        print(f"FAIL: --target-layers ({len(target_layers)}) and --snapshots "
              f"({len(snapshot_paths)}) must have same number of entries", file=sys.stderr)
        sys.exit(1)
    for snap_path in snapshot_paths:
        if not os.path.exists(snap_path):
            print(f"FAIL: snapshot not found: {snap_path}", file=sys.stderr)
            sys.exit(1)

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    print("=" * 70)
    print(f"Multi-layer BCJR compounding test")
    print(f"  target_layers: {target_layers}")
    for L, p in zip(target_layers, snapshot_paths):
        print(f"  layer {L}: {p}")
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

    # Stash FP weights for each target layer (for restore between configs)
    stashes = {L: stash_layer(model.model.layers[L]) for L in target_layers}

    results = {
        "model": args.model,
        "target_layers": target_layers,
        "snapshot_paths": snapshot_paths,
        "evals": [],
    }

    def restore_all():
        for L in target_layers:
            restore_layer(model.model.layers[L], stashes[L])

    # --- (a) FP baseline ---
    if args.ppl_fp_cached is not None:
        print(f"\n[a] FP baseline (cached): {args.ppl_fp_cached:.4f}")
        ppl_fp = args.ppl_fp_cached
    else:
        print("\n[a] FP baseline (no quantization)")
        ppl_fp = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    results["ppl_fp"] = ppl_fp

    # --- (b) Multi-layer PTQ ---
    print(f"\n[b] Multi-layer hard-Viterbi PTQ at layers {target_layers}", flush=True)
    for L in target_layers:
        install_ptq_at_layer(model, L, codebook, args.seed)
    ppl_ptq_multi = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    results["evals"].append({"label": "multi_ptq", "layers": target_layers, "ppl": ppl_ptq_multi})
    print(f"  PPL (multi PTQ at {target_layers}) = {ppl_ptq_multi:.4f}")
    restore_all()

    # --- (c) Multi-layer BCJR ---
    print(f"\n[c] Multi-layer BCJR-trained at layers {target_layers}", flush=True)
    for L, snap_path in zip(target_layers, snapshot_paths):
        install_bcjr_snap_at_layer(model, L, snap_path)
        print(f"  installed {snap_path} at layer {L}")
    ppl_bcjr_multi = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    results["evals"].append({"label": "multi_bcjr", "layers": target_layers, "ppl": ppl_bcjr_multi})
    print(f"  PPL (multi BCJR at {target_layers}) = {ppl_bcjr_multi:.4f}")
    restore_all()

    # --- (d) Single-layer BCJR breakdown (BCJR at one layer, FP elsewhere) ---
    print(f"\n[d] Per-layer BCJR breakdown (one layer at a time)", flush=True)
    for L, snap_path in zip(target_layers, snapshot_paths):
        install_bcjr_snap_at_layer(model, L, snap_path)
        ppl = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        results["evals"].append({"label": f"bcjr_L{L:02d}_only", "layers": [L], "ppl": ppl})
        print(f"  PPL (BCJR at L{L:02d} only)  = {ppl:.4f}")
        restore_all()

    # --- (e) Single-layer PTQ breakdown (for clean apples-to-apples) ---
    print(f"\n[e] Per-layer PTQ breakdown (one layer at a time)", flush=True)
    for L in target_layers:
        install_ptq_at_layer(model, L, codebook, args.seed)
        ppl = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        results["evals"].append({"label": f"ptq_L{L:02d}_only", "layers": [L], "ppl": ppl})
        print(f"  PPL (PTQ at L{L:02d} only)  = {ppl:.4f}")
        restore_all()

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"Multi-layer compounding test  (FP baseline = {ppl_fp:.4f})")
    print("-" * 70)
    by_label = {ev["label"]: ev["ppl"] for ev in results["evals"]}
    print(f"  {'config':<32} {'PPL':>8} {'Δ vs FP':>10}")
    print(f"  {'-'*32} {'-'*8} {'-'*10}")
    print(f"  {'FP (no quant)':<32} {ppl_fp:>8.4f}  {'0.0000':>10}")
    for L in target_layers:
        k = f"ptq_L{L:02d}_only"
        if k in by_label:
            print(f"  {f'PTQ at L{L:02d} only':<32} {by_label[k]:>8.4f}  {by_label[k]-ppl_fp:>+10.4f}")
    for L in target_layers:
        k = f"bcjr_L{L:02d}_only"
        if k in by_label:
            print(f"  {f'BCJR at L{L:02d} only':<32} {by_label[k]:>8.4f}  {by_label[k]-ppl_fp:>+10.4f}")
    print(f"  {f'PTQ at {target_layers}':<32} {ppl_ptq_multi:>8.4f}  {ppl_ptq_multi-ppl_fp:>+10.4f}")
    print(f"  {f'BCJR at {target_layers}':<32} {ppl_bcjr_multi:>8.4f}  {ppl_bcjr_multi-ppl_fp:>+10.4f}")
    print()

    delta_compound = ppl_bcjr_multi - ppl_ptq_multi
    print(f"  COMPOUNDING TEST: BCJR-multi vs PTQ-multi = {delta_compound:+.4f} PPL")
    if delta_compound < -0.01:
        print(f"  ✓ COMPOUNDING — multi-layer BCJR beats multi-layer PTQ by "
              f"{-delta_compound:.4f} PPL")
    elif delta_compound < 0.01:
        print(f"  ≈ NEUTRAL — multi-layer BCJR within ±0.01 of multi-layer PTQ")
    else:
        print(f"  ✗ ANTI-COMPOUNDING — multi-layer BCJR is "
              f"{delta_compound:+.4f} PPL WORSE than multi-layer PTQ")

    # Additivity check: sum of per-layer gains vs joint multi-layer gain
    sum_per_layer_bcjr_gain = 0
    for L in target_layers:
        bcjr_only = by_label.get(f"bcjr_L{L:02d}_only")
        ptq_only = by_label.get(f"ptq_L{L:02d}_only")
        if bcjr_only is not None and ptq_only is not None:
            sum_per_layer_bcjr_gain += (ptq_only - bcjr_only)
    multi_bcjr_gain = ppl_ptq_multi - ppl_bcjr_multi
    print(f"\n  ADDITIVITY CHECK:")
    print(f"    sum of per-layer BCJR gains (vs PTQ): {sum_per_layer_bcjr_gain:+.4f} PPL")
    print(f"    joint multi-layer BCJR gain (vs PTQ): {multi_bcjr_gain:+.4f} PPL")
    if abs(sum_per_layer_bcjr_gain - multi_bcjr_gain) < 0.005:
        print(f"    → ADDITIVE (within ±0.005)")
    elif sum_per_layer_bcjr_gain < multi_bcjr_gain:
        print(f"    → SUPER-ADDITIVE (gains reinforce each other)")
    else:
        print(f"    → SUB-ADDITIVE (gains partially cancel)")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
