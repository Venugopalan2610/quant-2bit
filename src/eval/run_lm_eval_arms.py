"""0-shot lm-eval on the controlled single-layer arms — the downstream check.

Loads Llama-3.2-1B, installs one arm's quantized weights at --layer (or leaves
FP), wraps in the lm-eval harness, runs a 0-shot suite. Addresses the PPL-only
limitation: does the trellis-vs-scalar PPL edge show up in task accuracy?

Honest caveat baked into the design: at single-layer (15/16 layers FP16) the
effect is diluted, so a null (all arms within noise) is a plausible and reportable
outcome — most likely to show on a sensitive layer (L1) where the PPL gap is large.

Arms: fp | ptq | ste_trellis | scalar_ctrl | bcjr
Usage:
  python -m src.eval.run_lm_eval_arms --arm scalar_ctrl --layer 4 \
      --tasks arc_easy,piqa,winogrande,hellaswag
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qat.eval_llama_layer import install_layer_weights, hard_viterbi_quantize
from src.eval.run_llama_full_ptq_baseline import make_seed_offsets

SNAP = {
    "ste_trellis": "cache/llama_matched/layer_{L:02d}_ste_trellis_fp_seed{s}.pt",
    "scalar_ctrl": "cache/llama_matched/layer_{L:02d}_scalar_ctrl_fp_seed{s}.pt",
    "bcjr":        "cache/llama_rung0/seed{s}/layer_{L:02d}_wq_ptqinit.pt",
}


def build_model(arm, layer, seed, model_path):
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    if arm == "fp":
        return m
    if arm == "ptq":
        from src.codes.lut_init import init_hyb_lut
        from src.qat.ste import make_hyb_codebook_gpu, L_BITS
        cb = make_hyb_codebook_gpu(init_hyb_lut(Q=9, n_samples=200_000, seed=seed),
                                   Q=9, L_bits=L_BITS)
        so = make_seed_offsets(layer, seed)
        install_layer_weights(m, layer,
            lambda kind, name, W: hard_viterbi_quantize(W, cb, seed=so[(kind, name)]),
            dtype=torch.bfloat16)
        return m
    path = SNAP[arm].format(L=layer, s=seed)
    snap = torch.load(path, map_location="cuda", weights_only=True)
    install_layer_weights(m, layer, snap, dtype=torch.bfloat16)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cache/model/Llama-3.2-1B")
    p.add_argument("--arm", required=True,
                   choices=["fp", "ptq", "ste_trellis", "scalar_ctrl", "bcjr"])
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default="arc_easy,piqa,winogrande,hellaswag")
    p.add_argument("--batch-size", default="16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"=== lm-eval 0-shot: arm={args.arm} L{args.layer} seed{args.seed} ===",
          flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = build_model(args.arm, args.layer, args.seed, args.model)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    res = simple_evaluate(model=lm, tasks=args.tasks.split(","), device="cuda")

    print(f"\n=== {args.arm} L{args.layer} 0-shot accuracy ===", flush=True)
    accs = {}
    for task, metrics in res["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        accs[task] = acc
        print(f"  {task:18s} acc = {acc:.4f}", flush=True)
    mean = sum(accs.values()) / len(accs)
    print(f"  {'MEAN':18s} acc = {mean:.4f}", flush=True)

    out = args.out or f"results/lmeval_L{args.layer}_{args.arm}_seed{args.seed}.json"
    os.makedirs("results", exist_ok=True)
    json.dump({"arm": args.arm, "layer": args.layer, "seed": args.seed,
               "acc": accs, "mean": mean}, open(out, "w"), indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
