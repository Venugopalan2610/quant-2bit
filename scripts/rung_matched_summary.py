"""Summarize the matched-arm L4 comparison and apply the PRE-REGISTERED
decision rule (see PREREGISTRATION.md): two arms separate iff |Δ| > σ, where σ
is the pooled per-seed std. With n=3 we make no significance claim below that.

Parses logs/matched_<arm>_seed<N>_eval.log for the hardened snapshot PPL (the
harness prints it as 'PPL_BCJR =' for any snapshot) plus the FP and PTQ refs.
"""
import argparse
import glob
import re
import statistics as st

ARMS = ["ste_trellis", "scalar_ctrl", "scalar_faithful"]


def parse_eval(path):
    out = {}
    with open(path) as f:
        txt = f.read()
    for key, pat in (("fp", r"PPL_FP\s*=\s*([\d.]+)"),
                     ("ptq", r"PPL_HardViterbi\s*=\s*([\d.]+)"),
                     ("snap", r"PPL_BCJR\s*=\s*([\d.]+)")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    return out


def collect(arm):
    """Return (snap_ppls, ptq_ref, fp_ref) across seeds for an arm."""
    snaps, ptqs, fps = [], [], []
    for path in sorted(glob.glob(f"logs/matched_{arm}_seed*_eval.log")):
        d = parse_eval(path)
        if "snap" in d:
            snaps.append(d["snap"])
        if "ptq" in d:
            ptqs.append(d["ptq"])
        if "fp" in d:
            fps.append(d["fp"])
    return snaps, ptqs, fps


def mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.pstdev(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    args = ap.parse_args()

    print(f"\n{'='*66}\nMatched-arm comparison — L{args.layer}, WikiText-2, n=3\n{'='*66}")
    results = {}
    fp_ref = ptq_ref = None
    for arm in ARMS:
        snaps, ptqs, fps = collect(arm)
        if not snaps:
            continue
        m, s = mean_std(snaps)
        results[arm] = (m, s, snaps)
        if fps:
            fp_ref = mean_std(fps)[0]
        if ptqs:
            ptq_ref = mean_std(ptqs)[0]
        d_ptq = m - ptq_ref if ptq_ref else float("nan")
        print(f"  {arm:16s} PPL = {m:.4f} ± {s:.4f}  "
              f"(Δ vs PTQ {d_ptq:+.4f})  seeds={[round(x,4) for x in snaps]}")

    if fp_ref:
        print(f"\n  FP  = {fp_ref:.4f}    PTQ = {ptq_ref:.4f}")

    # Pre-registered primary comparison Y: STE-trellis vs scalar_ctrl.
    if "ste_trellis" in results and "scalar_ctrl" in results:
        mt, stt, _ = results["ste_trellis"]
        mc, sc, _ = results["scalar_ctrl"]
        delta = mc - mt                     # >0 ⇒ trellis lower PPL ⇒ trellis wins
        sigma = max(stt, sc)                # pooled per-seed σ (conservative)
        print(f"\n{'-'*66}")
        print(f"  PRIMARY (Y): scalar_ctrl − STE-trellis = {delta:+.4f}  "
              f"(σ ≈ {sigma:.4f})")
        if abs(delta) <= sigma:
            print("  VERDICT: NO SEPARATION (|Δ| ≤ σ). Per pre-registration →")
            print("           trigger SUB-2-BIT PIVOT: rerun with N_BITS=1.")
        elif delta > sigma:
            print("  VERDICT: TRELLIS REPRESENTATION WINS (scalar worse by >σ).")
            print("           Trellis edge is real at 2-bit. Proceed to scaling.")
        else:
            print("  VERDICT: SCALAR WINS at 2-bit (near-optimal scalar ≤ trellis).")
            print("           Publishable negative result; consider sub-2-bit for X.")
        print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
