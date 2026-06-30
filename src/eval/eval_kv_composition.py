"""2x2 composition-quality eval: BCJR 2-bit WEIGHTS x TurboQuant low-bit KV.

Answers the cheap, load-bearing question before scaling the "couple them" pitch:
does stacking weight-quant + KV-quant degrade quality ADDITIVELY or
SUPER-additively? Four configs on Llama-3.2-1B, WikiText-2 PPL (+ optional
lm-eval 0-shot):

    fp16                : ceiling
    weights only        : BCJR 2-bit weights (Rung 1 full-model snapshots)
    kv only             : real TurboQuant KV (3-bit key / 2-bit val)
    both                : weights + KV

Composition check: Δ_both vs (Δ_weights + Δ_kv). ≈ → clean/additive (pitch
holds); ≫ → super-additive degradation (the real risk, found cheaply).

NOTE: this is the QUALITY test (1B is too small to show the cost/throughput
story — that's a separate big-model + vLLM milestone). KV path uses real
TurboQuant's torch quantizer; no Triton/vLLM needed for quality.

Usage (after Rung 1 finishes):
    python -m src.eval.eval_kv_composition \
        --weights-dir cache/llama_bcjr_full_greedy --seq-len 2048
"""
import argparse
import glob
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qat.eval_llama_layer import eval_ppl_wikitext, install_layer_weights
from src.eval.kv_quant import KVQuantHooks


def _install_all_bcjr_layers(model, weights_dir, dtype):
    """Install every layer_NN_wq.pt snapshot from a Rung 1 run into the model."""
    snaps = sorted(glob.glob(os.path.join(weights_dir, "layer_*_wq.pt")))
    n = 0
    for path in snaps:
        m = re.search(r"layer_(\d+)_wq\.pt", path)
        if not m:
            continue
        idx = int(m.group(1))
        snap = torch.load(path, map_location="cuda", weights_only=True)
        install_layer_weights(model, idx, snap, dtype=dtype)
        n += 1
    return n


def _fresh_model(model_path, dtype):
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                             device_map="cuda")
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cache/model/Llama-3.2-1B")
    p.add_argument("--weights-dir", default="cache/llama_bcjr_full_greedy")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=None, help="smoke: e.g. 20")
    p.add_argument("--key-bits", type=int, default=3)
    p.add_argument("--val-bits", type=int, default=2)
    p.add_argument("--kv-mode", choices=("turboquant", "uniform"), default="turboquant")
    p.add_argument("--out", default="results/kv_composition.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16

    def run(install_weights, kv):
        m = _fresh_model(args.model, dtype)
        if install_weights:
            n = _install_all_bcjr_layers(m, args.weights_dir, dtype)
            print(f"  installed {n} BCJR layers", flush=True)
        hooks = None
        if kv:
            hooks = KVQuantHooks(m, key_bits=args.key_bits, val_bits=args.val_bits,
                                 mode=args.kv_mode)
            print(f"  KV hooks ({args.kv_mode} k{args.key_bits}/v{args.val_bits}): "
                  f"{hooks.add()} projections", flush=True)
        ppl = eval_ppl_wikitext(m, tok, args.seq_len, args.max_windows)
        if hooks:
            hooks.remove()
        del m
        torch.cuda.empty_cache()
        return ppl

    print("\n=== [1/4] fp16 baseline ==="); fp = run(False, False)
    print("\n=== [2/4] weights only (BCJR 2-bit) ==="); w = run(True, False)
    print(f"\n=== [3/4] KV only ({args.kv_mode}) ==="); kv = run(False, True)
    print("\n=== [4/4] both ==="); both = run(True, True)

    d_w, d_kv, d_both = w - fp, kv - fp, both - fp
    additive = d_w + d_kv
    print("\n" + "=" * 60)
    print(f"  fp16            PPL = {fp:.4f}")
    print(f"  weights only    PPL = {w:.4f}   Δ {d_w:+.4f}")
    print(f"  KV only         PPL = {kv:.4f}   Δ {d_kv:+.4f}")
    print(f"  both            PPL = {both:.4f}   Δ {d_both:+.4f}")
    print(f"  additive expectation Δ_w+Δ_kv = {additive:+.4f}")
    excess = d_both - additive
    print(f"  super-additive excess = {excess:+.4f}", )
    verdict = ("CLEAN / additive — composition holds" if abs(excess) <= 0.15
               else "SUPER-ADDITIVE — stacking degrades; investigate before scaling")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    json.dump({"fp16": fp, "weights": w, "kv": kv, "both": both,
               "excess": excess, "kv_mode": args.kv_mode,
               "key_bits": args.key_bits, "val_bits": args.val_bits},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
