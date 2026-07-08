"""Drift-budget analysis: does BCJR basin-escape magnitude predict where greedy
BCJR-QAT breaks (the back-half reversals)?

Grounded in the BCJR-QAT paper's drift-budget theory (arXiv 2605.10655):
W_latent must drift past the Voronoi radius r ≈ σ_w/√(2πS) to escape the
QTIP-PTQ basin. Here we measure, per layer, how far each layer ACTUALLY escaped
its PTQ basin during Rung 1 (greedy), and correlate that escape magnitude with
the per-layer training-loss reversal. If a systematic relationship exists, the
drift-budget geometry predicts coupling-sensitivity → training effort can be
allocated principledly (joint only where the theory says greedy won't hold).

$0, no training — reads Rung 1 snapshots + recomputes PTQ (matched RHT seeds).

Per layer (mean over its 7 projections):
  escape_frac      = fraction of weights whose hardened codeword changed vs PTQ
  rel_drift        = ‖W_q_final − W_q_ptq‖ / ‖W_q_ptq‖
  escape_over_sigma= RMS(W_q_final − W_q_ptq) / σ_w   (∝ drift / r; the constant
                     √(2πS) is layer-independent so the RANKING is what matters)
vs Δloss% = (final_kl − init_kl)/init_kl from the Rung 1 logs (>0 = reversed).

Usage (after Rung 1): python -m src.eval.drift_budget_analysis
"""
import argparse
import glob
import math
import os
import re

import torch
from transformers import AutoModelForCausalLM

from src.codes.lut_init import init_hyb_lut
from src.qat.ste import make_hyb_codebook_gpu, L_BITS
from src.qat.eval_llama_layer import hard_viterbi_quantize
from src.eval.run_llama_full_ptq_baseline import make_seed_offsets

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def parse_layer_losses(log_paths):
    """layer_idx -> Δloss% from 'Layer NN done: ... loss X → Y' lines."""
    out = {}
    for p in log_paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            m = re.search(r"Layer (\d+) done:.*loss ([\d.eE+-]+)\s*(?:->|→)\s*([\d.eE+-]+)", line)
            if m:
                idx, x, y = int(m.group(1)), float(m.group(2)), float(m.group(3))
                out[idx] = 100.0 * (y - x) / (x + 1e-12)
    return out


def layer_escape_metrics(model, layer_idx, snap, codebook, seed=0):
    so = make_seed_offsets(layer_idx, seed)
    layer = model.model.layers[layer_idx]
    fracs, rels, ovs = [], [], []
    for kind, names, mod in (("attn", ATTN, layer.self_attn), ("mlp", MLP, layer.mlp)):
        for name in names:
            W_fp = getattr(mod, name).weight.data.float()
            W_ptq = hard_viterbi_quantize(W_fp, codebook, seed=so[(kind, name)]).float()
            W_fin = snap[f"{kind}_{name}"].to(W_ptq.device).float()
            diff = W_fin - W_ptq
            sigma_w = W_fp.std().item()
            fracs.append((diff.abs() > 1e-6).float().mean().item())
            rels.append((diff.norm() / (W_ptq.norm() + 1e-12)).item())
            ovs.append(diff.pow(2).mean().sqrt().item() / (sigma_w + 1e-12))
    n = len(fracs)
    return (sum(fracs) / n, sum(rels) / n, sum(ovs) / n)


def spearman(xs, ys):
    """Rank correlation without importing scipy hard-dep issues."""
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(xs, ys)
        return r, p
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cache/model/Llama-3.2-1B")
    ap.add_argument("--weights-dir", default="cache/llama_bcjr_full_greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/drift_budget.json")
    args = ap.parse_args()

    losses = parse_layer_losses(["logs/rung1_full_greedy.log", "logs/rung1_resume2_l5.log"])
    print("Building codebook + loading model...", flush=True)
    cb = make_hyb_codebook_gpu(init_hyb_lut(Q=9, n_samples=200_000, seed=args.seed),
                               Q=9, L_bits=L_BITS)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 device_map="cuda").eval()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.weights_dir, "layer_*_wq.pt"))):
        m = re.search(r"layer_(\d+)_wq\.pt", path)
        L = int(m.group(1))
        snap = torch.load(path, map_location="cuda", weights_only=True)
        frac, rel, ov = layer_escape_metrics(model, L, snap, cb, args.seed)
        rows.append((L, frac, rel, ov, losses.get(L, float("nan"))))
        print(f"  L{L:02d}: escape_frac={frac:.3f} rel_drift={rel:.4f} "
              f"esc/σ={ov:.4f}  Δloss={losses.get(L, float('nan')):+.1f}%", flush=True)

    print(f"\n{'='*70}\nDrift-budget vs greedy reversal (n={len(rows)} layers)\n{'='*70}")
    print(f"{'layer':>5} {'escape_frac':>12} {'rel_drift':>10} {'esc/σ':>9} {'Δloss%':>9} {'':>4}")
    for L, frac, rel, ov, dl in rows:
        flag = "  ⚠" if (dl == dl and dl > 0) else ""
        print(f"{L:>5} {frac:>12.3f} {rel:>10.4f} {ov:>9.4f} {dl:>+9.1f}{flag}")

    valid = [(f, r, o, d) for _, f, r, o, d in rows if d == d]
    if len(valid) >= 4:
        dl = [v[3] for v in valid]
        print(f"\nSpearman rank-correlation with Δloss% (reversal):")
        for label, xs in (("escape_frac", [v[0] for v in valid]),
                          ("rel_drift", [v[1] for v in valid]),
                          ("escape/σ (∝ drift/r)", [v[2] for v in valid])):
            r, p = spearman(xs, dl)
            print(f"  {label:22s}: ρ={r:+.3f}  p={p:.3f}")
        print("\nInterpretation:")
        print("  ρ significantly ≠ 0  → escape geometry PREDICTS where greedy reverses")
        print("      → drift-budget → principled joint-vs-greedy layer allocation.")
        print("  ρ ≈ 0                → reversals are unrelated to basin-escape magnitude")
        print("      → a different mechanism; the theory-predictor claim does NOT hold.")
    else:
        print("\n(need ≥4 layers with both metrics + logged loss for correlation)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import json
    json.dump([{"layer": L, "escape_frac": f, "rel_drift": r, "escape_over_sigma": o,
                "dloss_pct": d} for L, f, r, o, d in rows], open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
