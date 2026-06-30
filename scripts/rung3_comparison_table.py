"""Print the Rung 3 comparison table.

Reads BCJR-QAT result (from Rung 1) and full-model PTQ baseline (from
run_llama_full_ptq_baseline) and prints the paper-ready comparison table.

Frontier numbers (ParetoQ, LC-QAT) are filled in manually — update the
FRONTIER list below once you have the exact Llama-3.2-1B 2-bit numbers from
those papers.

Usage:
    python3 -m scripts.rung3_comparison_table \\
        --bcjr-result cache/llama_bcjr_full_greedy/full_model_eval.json \\
        --ptq-result  results/llama_full_ptq_ppl.json \\
        --ppl-fp 9.70
"""
import argparse
import json


# ---------------------------------------------------------------------------
# Fill these in from the papers AFTER checking Llama-3.2-1B 2-bit numbers.
# Set ppl=None to show "n/a" in the table until filled.
# ---------------------------------------------------------------------------
FRONTIER = [
    # (method,    type,          bits, data,     ppl,   notes)
    ("ParetoQ",  "scalar QAT",  2,    "30B tok", None, "arXiv 2501.06088 — fill from Table 3"),
    ("LC-QAT",   "VQ QAT",      2,    "4B tok",  None, "arXiv 2606.10531 Jun 2026 — fill from paper"),
]
# ---------------------------------------------------------------------------


def fmt_ppl(x):
    return f"{x:.4f}" if x is not None else "  n/a  "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcjr-result",
                        default="cache/llama_bcjr_full_greedy/full_model_eval.json")
    parser.add_argument("--ptq-result",
                        default="results/llama_full_ptq_ppl.json")
    parser.add_argument("--ppl-fp", type=float, default=9.70,
                        help="FP16 baseline PPL (cached from eval_llama_layer)")
    args = parser.parse_args()

    with open(args.bcjr_result) as f:
        bcjr = json.load(f)
    with open(args.ptq_result) as f:
        ptq = json.load(f)

    ppl_fp   = args.ppl_fp
    ppl_ptq  = ptq["wikitext2_ppl"]
    ppl_bcjr = bcjr["wikitext2_ppl"]
    delta    = ppl_ptq - ppl_bcjr

    n_steps = bcjr.get("args", {}).get("total_steps", "?")
    n_seqs  = bcjr.get("args", {}).get("batch_seqs", "?")
    seq_len = bcjr.get("args", {}).get("seq_len", "?")
    data_desc = f"{n_steps}s×{n_seqs}×{seq_len}t/layer"

    rows = [
        ("FP16",               "—",            16, "—",          ppl_fp,   "ceiling"),
        ("Hard-Viterbi PTQ",   "trellis PTQ",   2, "calib only", ppl_ptq,  "our PTQ baseline (seed 0)"),
    ] + [
        (m, t, b, d, p, n) for m, t, b, d, p, n in FRONTIER
    ] + [
        ("BCJR-QAT (greedy)", "trellis QAT",   2, data_desc,    ppl_bcjr,
         f"per-layer greedy, 16 layers, seed 0"),
    ]

    w0, w1, w2, w3, w4 = 22, 14, 5, 16, 10

    def row_str(method, typ, bits, data, ppl, notes):
        return (f"  {method:<{w0}} {typ:<{w1}} {bits:>{w2}} "
                f"{data:<{w3}} {fmt_ppl(ppl):>{w4}}  {notes}")

    sep = "  " + "─" * (w0 + w1 + w2 + w3 + w4 + 10)

    print()
    print(f"Llama-3.2-1B  2-bit WikiText-2 PPL comparison  (Rung 3)")
    print(sep)
    print(f"  {'Method':<{w0}} {'Type':<{w1}} {'Bits':>{w2}} "
          f"{'Data':<{w3}} {'Wiki2 PPL':>{w4}}  Notes")
    print(sep)
    for r in rows:
        print(row_str(*r))
    print(sep)
    print()

    print(f"  Key result: BCJR-QAT (greedy) vs Hard-Viterbi PTQ = {delta:+.4f} PPL")
    if delta > 0.42:
        print(f"  ✓ SUCCESS — beats PTQ by {delta:.4f} > 0.42 threshold")
    elif delta > 0:
        print(f"  ≈ MARGINAL — positive ({delta:+.4f}) but < 0.42 threshold")
    else:
        print(f"  ✗ NO WIN — per-layer gains ({delta:+.4f}) did not compound at scale")

    print()
    print(f"  Data-efficiency note:")
    print(f"    Hard-Vit PTQ: calibration data only (~0 tokens)")
    print(f"    BCJR-QAT:     {data_desc} per layer (far less than LC-QAT 4B / ParetoQ 30B)")
    if ppl_bcjr < ppl_ptq:
        print(f"    → BCJR-QAT improves PPL with negligible training data vs frontier QAT.")
    print()


if __name__ == "__main__":
    main()
