"""Parse Rung 0 eval logs and print the 3-seed summary table.

Reads logs/rung0_eval_seed{0,1,2}_L{layer}.log produced by rung0_multiseed.sh
and extracts PPL_FP, PPL_HardViterbi, PPL_BCJR from each.

Usage:
    python3 -m scripts.rung0_summary --layer 4
"""
import argparse
import os
import re
import sys


def parse_eval_log(path):
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return None

    r = {}
    m = re.search(r'PPL_FP\s*=\s*([\d.]+)', text)
    if m:
        r["ppl_fp"] = float(m.group(1))
    m = re.search(r'PPL_HardViterbi\s*=\s*([\d.]+)', text)
    if m:
        r["ppl_ptq"] = float(m.group(1))
    m = re.search(r'PPL_BCJR\s*=\s*([\d.]+)', text)
    if m:
        r["ppl_bcjr"] = float(m.group(1))

    if len(r) != 3:
        return None
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    L = args.layer
    print(f"\nRung 0 — L{L} single-layer BCJR-QAT: 3-seed reproducibility")
    print("=" * 72)
    print(f"  {'Seed':>4}  {'FP PPL':>8}  {'PTQ PPL':>8}  {'BCJR PPL':>9}  "
          f"{'ΔPPL (PTQ-BCJR)':>16}  {'Pass?':>6}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*16}  {'-'*6}")

    deltas = []
    missing = []
    for seed in [0, 1, 2]:
        log_path = os.path.join(args.log_dir, f"rung0_eval_seed{seed}_L{L}.log")
        r = parse_eval_log(log_path)
        if r is None:
            missing.append(seed)
            print(f"  {seed:>4}  (not found or incomplete: {log_path})")
            continue
        delta = r["ppl_ptq"] - r["ppl_bcjr"]
        deltas.append(delta)
        ok = "✓" if delta > 0.04 else ("≈" if delta > 0 else "✗")
        print(f"  {seed:>4}  {r['ppl_fp']:>8.4f}  {r['ppl_ptq']:>8.4f}  "
              f"{r['ppl_bcjr']:>9.4f}  {delta:>+16.4f}  {ok:>6}")

    if missing:
        print(f"\n  WARNING: seeds {missing} logs missing or incomplete.")

    if not deltas:
        print("\n  No data to summarize.")
        sys.exit(1)

    import statistics
    mean_d = sum(deltas) / len(deltas)
    std_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    print(f"\n  Seeds with data: {len(deltas)}/3")
    print(f"  mean ΔPPL:    {mean_d:+.4f}")
    print(f"  std ΔPPL:     {std_d:.4f}")
    print(f"  mean - 1σ:    {mean_d - std_d:+.4f}")
    print()

    if mean_d - std_d > 0.04:
        verdict = "PASS — BCJR robustly beats PTQ. Proceed to Rung 1."
        code = 0
    elif mean_d > 0:
        verdict = ("MARGINAL — positive mean but high variance or small gain.\n"
                   "  Consider more steps or debugging before Rung 1.")
        code = 1
    else:
        verdict = ("FAIL — no consistent win. Stop and diagnose before scaling.\n"
                   "  Check: quantizer bypass bug? non-monotone loss? PTQ init drift?")
        code = 2

    print(f"  Verdict: {verdict}")
    print("=" * 72)
    sys.exit(code)


if __name__ == "__main__":
    main()
