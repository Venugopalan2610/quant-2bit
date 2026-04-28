"""Evaluate per-layer quantization quality on Llama-3.2-1B.

For a given target layer, compares wikitext PPL across:
  (a) FP baseline — no quantization
  (b) Hard Viterbi PTQ on target layer only — no training
  (c) BCJR-trained snapshot for target layer — your overnight result

Other layers stay FP throughout. Isolates the layer's quantization quality
without confounding from the rest of the model.

Run:
    python -m src.qat.eval_llama_layer \\
        --target-layer 8 \\
        --snapshot cache/llama_bcjr_single/layer_08_wq.pt
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS, TrellisQuantSTE
from src.rht.transform import make_sign_vector
import numpy as np


ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def eval_ppl_wikitext(model, tokenizer, seq_len=2048, max_windows=None,
                      device="cuda"):
    """Compute PPL on wikitext-2 test (or wikitext-103 if you want).
    Sliding window with non-overlapping stride for speed.
    """
    print("  loading wikitext-2 raw test...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(device)
    n_total = ids.size(1)
    n_windows = n_total // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    print(f"  {n_total} tokens, {n_windows} non-overlapping seq_len={seq_len} windows",
          flush=True)

    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    t0 = time.time()
    for i in range(n_windows):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(input_ids=chunk, labels=chunk, use_cache=False)
        # HF returns mean cross-entropy over (seq_len-1) shift positions
        loss = out.loss.item()
        nll_sum += loss * (seq_len - 1)
        n_tokens += (seq_len - 1)
        if (i + 1) % 10 == 0:
            print(f"    window {i+1}/{n_windows}  "
                  f"running PPL={np.exp(nll_sum/n_tokens):.3f}  "
                  f"{time.time()-t0:.0f}s",
                  flush=True)
    ppl = float(np.exp(nll_sum / n_tokens))
    print(f"  PPL = {ppl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return ppl


@torch.no_grad()
def hard_viterbi_quantize(W_fp, codebook, seed):
    """Apply hard Viterbi quantization to a single weight matrix.
    Mirrors what _W_q_cache stores after prime_cache() in STE mode.
    Returns W_q in original (un-rotated) basis, fp32.
    """
    out_f, in_f = W_fp.shape
    sign_l_np = make_sign_vector(out_f, seed=seed)
    sign_r_np = make_sign_vector(in_f, seed=seed + 1)
    sign_l = torch.from_numpy(sign_l_np.astype(np.float32)).to(W_fp.device)
    sign_r = torch.from_numpy(sign_r_np.astype(np.float32)).to(W_fp.device)
    # TrellisQuantSTE.apply does the full pipeline:
    #   apply RHT → tile → Viterbi → untile → inverse RHT.
    W_q = TrellisQuantSTE.apply(W_fp.float(), sign_l, sign_r, codebook)
    return W_q


def install_layer_weights(model, layer_idx, snap_or_fn, dtype):
    """Install a per-layer weight dict into model.model.layers[layer_idx].
    `snap_or_fn` is either a dict {name: W_q tensor} or a callable that takes
    (proj_kind, name, W_fp) and returns W_q.
    """
    layer = model.model.layers[layer_idx]
    for name in ATTN_PROJ_NAMES:
        lin = getattr(layer.self_attn, name)
        key = f"attn_{name}"
        if isinstance(snap_or_fn, dict):
            W_q = snap_or_fn[key]
        else:
            W_q = snap_or_fn("attn", name, lin.weight.data)
        lin.weight.data.copy_(W_q.to(device=lin.weight.device, dtype=dtype))
    for name in MLP_PROJ_NAMES:
        lin = getattr(layer.mlp, name)
        key = f"mlp_{name}"
        if isinstance(snap_or_fn, dict):
            W_q = snap_or_fn[key]
        else:
            W_q = snap_or_fn("mlp", name, lin.weight.data)
        lin.weight.data.copy_(W_q.to(device=lin.weight.device, dtype=dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--snapshot", default="cache/llama_bcjr_single/layer_08_wq.pt")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA required"); sys.exit(1)

    print("=" * 70)
    print(f"Eval Llama-3.2-1B layer {args.target_layer} quantization")
    print(f"  snapshot: {args.snapshot}")
    print(f"  seq_len={args.seq_len}  max_windows={args.max_windows}")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("\nBuilding codebook...", flush=True)
    lut = init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed)
    codebook = make_hyb_codebook_gpu(lut, Q=9, L_bits=L_BITS)

    # ---------------- (a) FP baseline ----------------
    print("\n[a] FP baseline (no quantization)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    ppl_fp = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    print(f"\n  PPL_FP = {ppl_fp:.4f}", flush=True)

    # Stash original layer weights for reuse across (b) and (c)
    original_weights = {}
    layer = model.model.layers[args.target_layer]
    for name in ATTN_PROJ_NAMES:
        original_weights[f"attn_{name}"] = (
            getattr(layer.self_attn, name).weight.data.detach().clone()
        )
    for name in MLP_PROJ_NAMES:
        original_weights[f"mlp_{name}"] = (
            getattr(layer.mlp, name).weight.data.detach().clone()
        )

    # ---------------- (b) Hard Viterbi PTQ on target layer only ----------------
    print(f"\n[b] Hard Viterbi PTQ on layer {args.target_layer} only",
          flush=True)
    # Per-QL seed must match what QATDenseDecoderLayer uses, so the sign
    # vectors are identical to what BCJR training saw — fair comparison.
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
    ppl_hv = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    print(f"\n  PPL_HardViterbi = {ppl_hv:.4f}", flush=True)

    # Restore originals before (c) so (c) is independent of (b)
    for name in ATTN_PROJ_NAMES:
        getattr(layer.self_attn, name).weight.data.copy_(
            original_weights[f"attn_{name}"]
        )
    for name in MLP_PROJ_NAMES:
        getattr(layer.mlp, name).weight.data.copy_(
            original_weights[f"mlp_{name}"]
        )

    # ---------------- (c) BCJR-trained snapshot ----------------
    print(f"\n[c] BCJR-trained snapshot for layer {args.target_layer}",
          flush=True)
    if not os.path.exists(args.snapshot):
        print(f"  ERROR: snapshot not found at {args.snapshot}")
        print(f"  Skipping (c). Got PPL_FP={ppl_fp:.4f} PPL_HV={ppl_hv:.4f}")
        return
    snap = torch.load(args.snapshot, map_location="cuda", weights_only=True)
    print(f"  loaded snapshot keys: {sorted(snap.keys())}", flush=True)
    install_layer_weights(model, args.target_layer, snap, dtype=torch.float16)
    ppl_bcjr = eval_ppl_wikitext(model, tokenizer, args.seq_len, args.max_windows)
    print(f"\n  PPL_BCJR = {ppl_bcjr:.4f}", flush=True)

    # ---------------- summary ----------------
    print("\n" + "=" * 70)
    print(f"Llama-3.2-1B  layer {args.target_layer}  wikitext-2 PPL")
    print("-" * 70)
    print(f"  (a) FP baseline:          {ppl_fp:.4f}")
    print(f"  (b) Hard Viterbi PTQ:     {ppl_hv:.4f}  "
          f"(Δ vs FP: +{ppl_hv-ppl_fp:.4f})")
    print(f"  (c) BCJR-trained:         {ppl_bcjr:.4f}  "
          f"(Δ vs FP: +{ppl_bcjr-ppl_fp:.4f})")
    if ppl_bcjr < ppl_hv:
        print(f"\n  BCJR training BEATS hard Viterbi PTQ by "
              f"{ppl_hv-ppl_bcjr:.4f} PPL")
    else:
        print(f"\n  BCJR training did NOT beat hard Viterbi PTQ "
              f"(Δ = {ppl_bcjr-ppl_hv:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
