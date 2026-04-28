"""Bootstrap noise estimate on PPL from per-window NLLs.

Wikitext-2 PPL is point-deterministic: same model + same tokens → identical
PPL across runs. To get a noise floor, we resample windows with replacement
(standard nonparametric bootstrap) and recompute PPL on each resample.

This estimates EVAL noise — the variance you'd see if you'd evaluated on a
slightly different sample of test windows. It's a LOWER bound on total noise
(it doesn't capture training-seed variance), but it's enough to answer:

    "Is the win larger than the noise floor of the eval procedure itself?"

If 0.21 PPL > 2σ_bootstrap, the win is at least eval-significant.
If 0.21 PPL < σ_bootstrap, the win is within eval noise alone.

Run:
    python -m src.eval.bootstrap_ppl results/perwin_*.npz
"""
import argparse
import os
import sys

import numpy as np


def bootstrap_ppl(nll, tokens, n_boot=10_000, seed=0):
    """Bootstrap-resample windows with replacement; return mean, std, ci."""
    rng = np.random.default_rng(seed)
    n = len(nll)
    ppl_samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_nll = nll[idx].sum()
        s_tok = tokens[idx].sum()
        ppl_samples[b] = float(np.exp(s_nll / s_tok))
    return {
        "ppl_point": float(np.exp(nll.sum() / tokens.sum())),
        "ppl_boot_mean": float(ppl_samples.mean()),
        "ppl_boot_std": float(ppl_samples.std(ddof=1)),
        "ppl_ci95_lo": float(np.percentile(ppl_samples, 2.5)),
        "ppl_ci95_hi": float(np.percentile(ppl_samples, 97.5)),
        "n_windows": n,
        "n_boot": n_boot,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_files", nargs="+",
                        help="One or more .npz files saved by run_ppl --save-per-window")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"{'config':<40} {'point':>8} {'mean':>8} {'σ':>6} "
          f"{'95% CI':>20}")
    print("-" * 92)
    results = {}
    for path in args.npz_files:
        data = np.load(path)
        nll = data["nll"]
        tokens = data["tokens"]
        stats = bootstrap_ppl(nll, tokens, n_boot=args.n_boot, seed=args.seed)
        label = os.path.basename(path).replace("perwin_", "").replace(".npz", "")
        results[label] = stats
        ci = f"[{stats['ppl_ci95_lo']:.3f}, {stats['ppl_ci95_hi']:.3f}]"
        print(f"{label:<40} {stats['ppl_point']:>8.4f} "
              f"{stats['ppl_boot_mean']:>8.4f} {stats['ppl_boot_std']:>6.4f} "
              f"{ci:>20}")

    # Pairwise significance vs the FIRST file (treated as baseline)
    if len(args.npz_files) >= 2:
        baseline_label = list(results.keys())[0]
        baseline = results[baseline_label]
        print(f"\nPairwise vs {baseline_label} (baseline):")
        print(f"{'compared config':<40} {'Δ PPL':>8} {'σ_pooled':>10} "
              f"{'Δ/σ':>8} {'verdict':>14}")
        print("-" * 92)
        for label, stats in results.items():
            if label == baseline_label:
                continue
            delta = baseline["ppl_point"] - stats["ppl_point"]   # positive = compared is better
            # Pooled std (independent samples assumption — conservative for shared windows)
            sigma = float(np.sqrt(baseline["ppl_boot_std"] ** 2
                                  + stats["ppl_boot_std"] ** 2))
            z = delta / sigma if sigma > 0 else float("inf")
            if abs(z) >= 2.0:
                verdict = "SIGNIFICANT" if z > 0 else "WORSE (sig)"
            elif abs(z) >= 1.0:
                verdict = "marginal"
            else:
                verdict = "noise"
            print(f"{label:<40} {delta:>+8.4f} {sigma:>10.4f} "
                  f"{z:>+8.2f}σ {verdict:>14}")


if __name__ == "__main__":
    main()
