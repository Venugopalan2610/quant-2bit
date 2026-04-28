"""Run lm-eval-harness on a (possibly quantized) OLMoE model.

Usage:
    python -m src.eval.run_lm_eval --config fp16 --tasks wikitext,hellaswag,arc_challenge,arc_easy,piqa
    python -m src.eval.run_lm_eval --config per_expert --tasks wikitext,hellaswag,arc_challenge,arc_easy,piqa
    python -m src.eval.run_lm_eval --config per_layer_H --tasks wikitext,hellaswag,arc_challenge,arc_easy,piqa
"""
import os
import sys
import json
import time
import argparse
import torch
from transformers import OlmoeForCausalLM, AutoTokenizer

from src.eval.install_quantized import install_quantized_weights
from src.eval.install_bcjr import install_bcjr_weights

MODEL_DIR = "cache/model/olmoe-1b-7b-0125"
RESULTS_DIR = "results"

CONFIG_TO_QUANT_DIR = {
    "fp16":         None,
    "per_expert":   "cache/quantized",
    "per_layer_H":  "cache/quantized_per_layer_H",
    "per_layer_weighted_H": "cache/quantized_per_layer_weighted_H",
    "bcjr":         "cache/qat_bcjr_full",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIG_TO_QUANT_DIR.keys()), required=True)
    parser.add_argument("--tasks", required=True,
                        help="comma-separated list of lm-eval tasks")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="auto:4",
                        help="lm-eval batch size, e.g. 'auto:4' or '1'")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path; default auto-generated")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples per task for debug runs")
    parser.add_argument("--bcjr-dir", default="cache/qat_bcjr_full",
                        help="dir with per-layer BCJR snapshots (for --config bcjr)")
    parser.add_argument("--allow-partial-bcjr", action="store_true")
    parser.add_argument("--gpu-memory", default=None,
                        help="per-GPU budget for device_map='auto' (e.g. '10GiB'). "
                             "Offloads to CPU above this. Required when model > VRAM.")
    parser.add_argument("--cpu-memory", default="28GiB",
                        help="CPU-side budget for offloaded weights (default 28GiB)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print("=" * 60)
    print(f"lm-eval run: config={args.config}")
    print(f"  tasks: {tasks}")
    print(f"  num_fewshot: {args.num_fewshot}")
    print(f"  batch_size: {args.batch_size}")
    if args.limit:
        print(f"  limit (per task): {args.limit}")
    print("=" * 60)

    # Load model
    if args.gpu_memory:
        print(f"\n[1/4] Loading OLMoE bf16 with CPU offload "
              f"(GPU<={args.gpu_memory}, CPU<={args.cpu_memory})...")
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
            low_cpu_mem_usage=True,
        )
    else:
        print(f"\n[1/4] Loading OLMoE in bf16 on GPU...")
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(MODEL_DIR, **load_kwargs)
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Optionally install quantized weights
    if args.config == "bcjr":
        print(f"\n[2/4] Installing BCJR-trained weights from {args.bcjr_dir}...")
        t0 = time.time()
        install_stats = install_bcjr_weights(
            model, bcjr_dir=args.bcjr_dir, verbose=True,
            allow_partial=args.allow_partial_bcjr,
        )
        print(f"  install complete in {time.time() - t0:.0f}s")
    elif CONFIG_TO_QUANT_DIR[args.config] is not None:
        quant_dir = CONFIG_TO_QUANT_DIR[args.config]
        print(f"\n[2/4] Installing 2-bit quantized weights from {quant_dir}...")
        t0 = time.time()
        install_stats = install_quantized_weights(model, quant_dir=quant_dir, verbose=True)
        if install_stats["n_missing"] > 0:
            print(f"FAIL: {install_stats['n_missing']} missing targets")
            sys.exit(1)
        print(f"  install complete in {time.time() - t0:.0f}s")
    else:
        print("\n[2/4] Skipping install (fp16 baseline)")
        install_stats = None

    # Wrap in lm-eval HFLM
    print(f"\n[3/4] Wrapping model in lm-eval HFLM...")
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device="cuda:0",
    )

    # Run lm-eval
    print(f"\n[4/4] Running lm-eval on tasks: {tasks}")
    t0 = time.time()
    results = simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    elapsed = time.time() - t0
    print(f"\nlm-eval finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if "results" in results:
        for task_name, metrics in results["results"].items():
            print(f"\n{task_name}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    # Save full results JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"lm_eval_{args.config}_{ts}.json")

    # results dict can have numpy/tensor values; coerce to JSON-safe
    def _safe(o):
        if isinstance(o, (str, int, float, bool, type(None))):
            return o
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(x) for x in o]
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                return str(o)
        return str(o)

    save_dict = {
        "config": args.config,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "install_stats": install_stats,
        "results": _safe(results.get("results", {})),
        "configs": _safe(results.get("configs", {})),
        "versions": _safe(results.get("versions", {})),
    }

    with open(output_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
