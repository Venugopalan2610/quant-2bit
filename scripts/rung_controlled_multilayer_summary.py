"""Per-layer summary of the controlled trellis-vs-scalar generalization sweep.
Tests whether the L4 result holds across depth (pre-mortem #12). For each layer
prints: trellis-PTQ, scalar-PTQ, the REPRESENTATION gap (scalar−trellis PTQ),
and the Y gap (scalar_ctrl − STE-trellis, both 10-step STE). A clean
generalization story = representation gap > 0 (trellis better) on every layer.
"""
import glob
import re
import statistics as st

TAGS = ["scalarPTQ", "faithfulPTQ", "steTrellis", "scalarCtrl"]


def parse(path):
    out = {}
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return out
    for k, pat in (("ptq", r"PPL_HardViterbi\s*=\s*([\d.]+)"),
                   ("snap", r"PPL_BCJR\s*=\s*([\d.]+)")):
        m = re.search(pat, txt)
        if m:
            out[k] = float(m.group(1))
    return out


def arm_ppls(layer, tag):
    snaps, ptqs = [], []
    for p in sorted(glob.glob(f"logs/matched_L{layer}_{tag}_seed*_eval.log")):
        d = parse(p)
        if "snap" in d:
            snaps.append(d["snap"])
        if "ptq" in d:
            ptqs.append(d["ptq"])
    return snaps, ptqs


def ms(xs):
    if not xs:
        return None, None
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


def layers_present():
    ls = set()
    for p in glob.glob("logs/matched_L*_*_eval.log"):
        m = re.search(r"matched_L(\d+)_", p)
        if m:
            ls.add(int(m.group(1)))
    return sorted(ls)


def main():
    layers = layers_present()
    if not layers:
        print("no sweep logs yet (logs/matched_L*_*_eval.log)")
        return
    print(f"\n{'='*74}\nControlled trellis-vs-scalar across layers (2-bit, n=3)\n{'='*74}")
    print(f"{'layer':>5} {'trellisPTQ':>11} {'scalarPTQ':>10} {'REPgap':>8} "
          f"{'STEtrel':>8} {'scalCtrl':>9} {'Ygap':>7}")
    rep_signs, y_signs = [], []
    for L in layers:
        sp, ptq = arm_ppls(L, "scalarPTQ")
        _, ptq2 = arm_ppls(L, "steTrellis")
        ste, _ = arm_ppls(L, "steTrellis")
        sc, _ = arm_ppls(L, "scalarCtrl")
        ptq_ref = ms((ptq or []) + (ptq2 or []))[0]
        sp_m = ms(sp)[0]; ste_m = ms(ste)[0]; sc_m = ms(sc)[0]
        rep = (sp_m - ptq_ref) if (sp_m and ptq_ref) else None      # >0 ⇒ trellis better
        y = (sc_m - ste_m) if (sc_m and ste_m) else None            # >0 ⇒ trellis better
        if rep is not None:
            rep_signs.append(rep > 0)
        if y is not None:
            y_signs.append(y > 0)
        def f(x):
            return f"{x:.3f}" if x is not None else "  -  "
        print(f"{L:>5} {f(ptq_ref):>11} {f(sp_m):>10} "
              f"{('+' if (rep or 0)>=0 else '')+f(rep):>8} "
              f"{f(ste_m):>8} {f(sc_m):>9} "
              f"{('+' if (y or 0)>=0 else '')+f(y):>7}")
    print("-" * 74)
    if rep_signs:
        print(f"  REPRESENTATION gap (scalar-PTQ worse than trellis-PTQ): "
              f"{sum(rep_signs)}/{len(rep_signs)} layers favor trellis")
    if y_signs:
        print(f"  Y gap (scalar worse than trellis under STE):           "
              f"{sum(y_signs)}/{len(y_signs)} layers favor trellis")
    if rep_signs and all(rep_signs):
        print("  → representation win GENERALIZES across sampled layers.")
    elif rep_signs:
        print("  → representation win is LAYER-DEPENDENT — report honestly.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
