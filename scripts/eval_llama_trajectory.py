"""Evaluate the per-step W_latent checkpoint trajectory from a 30-step
BCJR-QAT run.

For each `layer_NN_step{NN}_ptqinit_latent.pt` in the ckpt dir:
  1. Inject W_latent into a QAT layer fixture
  2. Hard-Viterbi to W_q via TrellisQuantSTE
  3. Install W_q into a fresh Llama-3.2-1B FP model's target layer
  4. Eval WikiText-2 PPL

Loads the model + codebook ONCE (~30 s), then each step's eval is just the
PPL pass (~3-4 min on H100). For 6 ckpts that's ~25 min total — much faster
than 6 separate `eval_llama_layer.py` invocations.

Run on the pod after the 30-step training finishes:
    python -m scripts.eval_llama_trajectory \
        --ckpt-dir cache/llama_bcjr_30step \
        --target-layer 8 \
        --output results/llama_30step_trajectory.json

Output JSON has: FP baseline, PTQ baseline, then [{step, T, kl, ppl}, ...].
"""
import argparse
import copy
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.bcjr.anneal import convert_layer_to_bcjr, convert_layer_to_ste
from src.qat.eval_llama_layer import (
    eval_ppl_wikitext, install_layer_weights,
    ATTN_PROJ_NAMES, MLP_PROJ_NAMES,
    hard_viterbi_quantize,
)


def stash_layer(layer):
    """Snapshot the FP weights of a Llama decoder layer for restore."""
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


def w_latent_ckpt_to_w_q_snap(ckpt, qat_fixture, codebook_gpu):
    """Convert a per-step W_latent ckpt into the W_q snap dict format that
    install_layer_weights consumes.

    Strategy: use the existing QAT layer fixture (already wraps a Llama layer),
    inject the ckpt's W_latent into each QL, then run hard-Viterbi via
    TrellisQuantSTE on each — same code path the trainer uses at end-of-run.
    """
    w_latent_dict = ckpt["W_latent"]
    snap = {}
    with torch.no_grad():
        for name, ql in qat_fixture._all_qls():
            if name not in w_latent_dict:
                raise KeyError(
                    f"ckpt missing W_latent for {name!r}. Got: {sorted(w_latent_dict.keys())}"
                )
            W_latent_cpu = w_latent_dict[name]
            W_latent = W_latent_cpu.to(ql.W_latent.device, dtype=ql.W_latent.dtype)
            ql.W_latent.copy_(W_latent)
            W_q = TrellisQuantSTE.apply(
                ql.W_latent.float(), ql.sign_l, ql.sign_r, ql.codebook_gpu,
            )
            # Map QL name → snap key.
            # Observed convention from QATDenseDecoderLayer._all_qls():
            # names are already in "attn_q_proj" / "mlp_gate_proj" format,
            # which matches install_layer_weights's expected snap keys exactly.
            # Handle the dotted "attn.q_proj" form too defensively.
            if name.startswith("attn_") or name.startswith("mlp_"):
                snap_key = name
            elif "." in name:
                kind, proj = name.split(".", 1)
                snap_key = f"{kind}_{proj}"
            elif name in ATTN_PROJ_NAMES:
                snap_key = f"attn_{name}"
            elif name in MLP_PROJ_NAMES:
                snap_key = f"mlp_{name}"
            else:
                raise ValueError(f"can't map QL name {name!r} to snap key")
            snap[snap_key] = W_q.detach().cpu()
    return snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cache/model/Llama-3.2-1B")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Directory with layer_NN_step{NN}_*_latent.pt files")
    parser.add_argument("--ckpt-glob",
                        default="layer_{LAYER:02d}_step*_latent.pt",
                        help="Glob pattern; {LAYER} expands to --target-layer")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--output", default="results/llama_trajectory.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip FP and PTQ baselines (use cached values)")
    parser.add_argument("--ppl-fp-cached", type=float, default=None)
    parser.add_argument("--ppl-ptq-cached", type=float, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    os.environ.setdefault("BCJR_TRITON", "1")
    os.environ.setdefault("BCJR_MONOLITHIC", "1")

    # --- Discover ckpts and sort by step ---
    pattern = os.path.join(args.ckpt_dir,
                           args.ckpt_glob.format(LAYER=args.target_layer))
    ckpt_paths = sorted(glob.glob(pattern))
    if not ckpt_paths:
        print(f"FAIL: no ckpts matched {pattern}"); sys.exit(1)
    print(f"Found {len(ckpt_paths)} W_latent ckpts:")
    for p in ckpt_paths:
        print(f"  {p}")

    print("\nLoading tokenizer + codebook...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    codebook = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    print("\nLoading FP model (fp16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    target = model.model.layers[args.target_layer]
    fp_stash = stash_layer(target)

    results = {
        "model": args.model,
        "target_layer": args.target_layer,
        "seq_len": args.seq_len,
        "max_windows": args.max_windows,
        "ckpts": [],
    }

    # --- Baselines ---
    if args.skip_baselines and args.ppl_fp_cached is not None:
        print(f"\n[a] FP baseline (cached): {args.ppl_fp_cached:.4f}")
        ppl_fp = args.ppl_fp_cached
    else:
        print("\n[a] FP baseline (no quantization)")
        ppl_fp = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    results["ppl_fp"] = ppl_fp

    if args.skip_baselines and args.ppl_ptq_cached is not None:
        print(f"\n[b] Hard Viterbi PTQ on layer {args.target_layer} (cached): "
              f"{args.ppl_ptq_cached:.4f}")
        ppl_ptq = args.ppl_ptq_cached
    else:
        print(f"\n[b] Hard Viterbi PTQ on layer {args.target_layer} only")
        base_seed = args.seed + args.target_layer * 997
        seed_offsets = {
            ("attn", "q_proj"): base_seed + 1_000_000 + 0 * 17,
            ("attn", "k_proj"): base_seed + 1_000_000 + 1 * 17,
            ("attn", "v_proj"): base_seed + 1_000_000 + 2 * 17,
            ("attn", "o_proj"): base_seed + 1_000_000 + 3 * 17,
            ("mlp",  "gate_proj"): base_seed + 2_000_000 + 0 * 31,
            ("mlp",  "up_proj"):   base_seed + 2_000_000 + 1 * 31,
            ("mlp",  "down_proj"): base_seed + 2_000_000 + 2 * 31,
        }

        def hv_fn(kind, name, W_fp):
            return hard_viterbi_quantize(W_fp, codebook,
                                         seed=seed_offsets[(kind, name)])
        install_layer_weights(model, args.target_layer, hv_fn, dtype=torch.float16)
        ppl_ptq = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        restore_layer(target, fp_stash)
    results["ppl_ptq"] = ppl_ptq

    # --- QAT fixture for converting W_latent → W_q ---
    # IMPORTANT: deepcopy the target before wrapping. QATDenseDecoderLayer
    # mutates its input layer's children in place (replaces nn.Linear with
    # QuantizedLinear). Without the deepcopy, model.model.layers[target]
    # gets corrupted and we can't install/restore weights for eval anymore.
    print(f"\nBuilding QAT layer fixture (deepcopy of target, for W_latent → W_q conversion)...",
          flush=True)
    cfg = model.config
    fp_layer_for_fixture = copy.deepcopy(target).float()
    qat_fixture = QATDenseDecoderLayer(
        fp_layer_for_fixture, codebook_gpu=codebook, config=cfg,
        seed=args.seed + args.target_layer * 997,
        reencode_every_n_steps=1,
    )
    # Move to GPU; we won't train it.
    qat_fixture = qat_fixture.cuda()
    # Sanity check: target should still be an FP nn.Linear-based layer.
    assert hasattr(target.self_attn.q_proj, "weight"), \
        "QAT fixture mutated the FP model's target layer despite deepcopy"

    # --- Per-ckpt eval loop ---
    print("\n" + "=" * 70)
    print("Per-step BCJR trajectory")
    print("=" * 70, flush=True)
    step_re = re.compile(r"step(\d+)")
    for path in ckpt_paths:
        m = step_re.search(os.path.basename(path))
        step_num = int(m.group(1)) if m else -1
        print(f"\n[step {step_num}] {path}", flush=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        snap = w_latent_ckpt_to_w_q_snap(ckpt, qat_fixture, codebook)
        install_layer_weights(model, args.target_layer, snap, dtype=torch.float16)
        ppl = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
        restore_layer(target, fp_stash)
        rec = {
            "step": step_num,
            "T": ckpt.get("T"),
            "kl": ckpt.get("kl"),
            "ppl": ppl,
            "ckpt_path": path,
        }
        results["ckpts"].append(rec)
        print(f"  step {step_num}: T={rec['T']}  kl={rec['kl']:.4e}  PPL={ppl:.4f}",
              flush=True)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"Trajectory summary  (FP={ppl_fp:.4f}, PTQ={ppl_ptq:.4f})")
    print("-" * 70)
    print(f"  {'step':>5}  {'T':>10}  {'kl':>12}  {'PPL':>8}  {'Δ vs PTQ':>10}")
    for rec in results["ckpts"]:
        delta = rec["ppl"] - ppl_ptq
        print(f"  {rec['step']:>5}  {rec['T']:>10.4e}  {rec['kl']:>12.4e}  "
              f"{rec['ppl']:>8.4f}  {delta:>+10.4f}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved trajectory to {args.output}")


if __name__ == "__main__":
    main()
