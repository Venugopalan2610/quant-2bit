"""Full-model 2-bit hard-Viterbi PTQ baseline for Llama-3.2-1B.

For the Rung 3 comparison table: quantize all 16 layers with hard Viterbi
(same codebook + seed scheme as the BCJR-QAT pipeline) and evaluate
WikiText-2 PPL. This is the "QTIP-PTQ" reference point.

Usage:
    python -m src.eval.run_llama_full_ptq_baseline \\
        --model cache/model/Llama-3.2-1B \\
        --output results/llama_full_ptq_ppl.json
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.eval_llama_layer import (
    eval_ppl_wikitext, hard_viterbi_quantize, install_layer_weights,
)


NUM_LAYERS = 16


def make_seed_offsets(layer_idx, seed):
    """Must match QATDenseDecoderLayer's seed scheme exactly."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/model/Llama-3.2-1B")
    parser.add_argument("--output", default="results/llama_full_ptq_ppl.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated indices to quantize (default: all 0-15)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    layers_to_quant = (
        [int(x) for x in args.layers.split(",")]
        if args.layers else list(range(NUM_LAYERS))
    )

    print("=" * 70)
    print(f"Full-model hard-Viterbi PTQ baseline: Llama-3.2-1B")
    print(f"  model:  {args.model}")
    print(f"  layers: {layers_to_quant}")
    print(f"  seed:   {args.seed}")
    print("=" * 70, flush=True)

    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    codebook = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    print("\nLoading model (fp16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  loaded in {time.time()-t0:.0f}s  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    print(f"\nQuantizing {len(layers_to_quant)} layers with hard Viterbi PTQ...", flush=True)
    t0 = time.time()
    for layer_idx in layers_to_quant:
        seed_offsets = make_seed_offsets(layer_idx, args.seed)

        def hv_fn(kind, name, W_fp, _so=seed_offsets):
            return hard_viterbi_quantize(W_fp, codebook, seed=_so[(kind, name)])

        install_layer_weights(model, layer_idx, hv_fn, dtype=torch.float16)
        if (layer_idx + 1) % 4 == 0:
            print(f"  quantized layers 0..{layer_idx}", flush=True)
    print(f"  all {len(layers_to_quant)} layers in {time.time()-t0:.0f}s", flush=True)

    print("\nEvaluating WikiText-2 PPL...", flush=True)
    ppl = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)

    result = {
        "model": args.model,
        "method": "hard_viterbi_ptq",
        "bits": 2,
        "layers_quantized": layers_to_quant,
        "wikitext2_ppl": ppl,
        "eval_seq_len": args.seq_len,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Full-model 2-bit hard-Viterbi PTQ  WikiText-2 PPL: {ppl:.4f}")
    print(f"Saved: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
