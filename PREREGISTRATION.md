# Pre-registration — trellis vs scalar 2-bit QAT at matched budget

Written **before** any scalar-arm numbers exist, to remove researcher
degrees of freedom (pre-mortem risk #4: don't pick the layer/arm that wins
after seeing results). Commit this; do not edit results into it after the fact.

## The question

The published BCJR-QAT result shows **trellis-trained beats trellis-untrained**
(BCJR-QAT −0.084 PPL vs QTIP-PTQ on L4). It does **not** show trellis beats a
*scalar* quantizer. That comparison — the one that justifies the method against
the ParetoQ/BitNet family — has never been run. This experiment runs it.

## Arms (all on Llama-3.2-1B, L4, WikiText-2, hardened PPL, n=3 seeds 0/1/2)

| arm | bits | basis | training | role |
|---|---|---|---|---|
| FP | 16 | — | — | ceiling (9.70) |
| PTQ (trellis) | 2.000 | RHT | none | floor reference (10.22) |
| **STE-trellis** | 2.000 | RHT | identity-STE | trellis repr, STE-trained |
| **scalar_ctrl** | 2.000 | RHT | identity-STE | near-Lloyd-Max scalar, SAME basis |
| **scalar_faithful** | ~2.125 | native | STE + learned clip | deployed ParetoQ recipe |
| BCJR-trellis | 2.000 | RHT | soft relaxation | ours (have: −0.08 vs PTQ) |

All arms: identical data, steps (10), LR (2e-4), seed, grad-clip, FP init
(default). `scalar_ctrl` shares the **exact RHT sign vectors** of the trellis
arm (same seed derivation), so the basis is identical and the **only** variable
is the quantizer.

## The three comparisons and what each proves (fixed in advance)

- **Y — representation, training held equal:** STE-trellis vs scalar_ctrl.
  Both identity-STE, both 2.000 b, same basis, same init. *Isolates the trellis
  representation.* This is the load-bearing, no-excuses comparison.
- **Z — value of the soft relaxation:** BCJR-trellis vs STE-trellis. Same
  representation, training differs. *Isolates BCJR.*
- **X — vs the deployed recipe:** BCJR-trellis (2.000 b) vs scalar_faithful
  (~2.125 b). If trellis wins here it beat a scalar given *more* bits.

**Headline claim is Y.** X and Z are secondary. We commit to reporting all
three regardless of which way they fall.

## Success / decision rule (fixed in advance)

Let Δ = mean PPL gap between two arms across 3 seeds, σ = pooled per-seed std
(currently ~0.036 for the trellis arm).

- **Separation requires |Δ| > σ** (gap exceeds one seed-σ). With n=3 we will
  NOT claim significance below this; we report the gap with its error bar and
  call it inconclusive.
- **Primary outcome (Y):**
  - STE-trellis beats scalar_ctrl by > σ → trellis representation is real;
    proceed (BCJR/full-model become the scaling story).
  - gap < σ (no separation) → **the 2-bit regime does not separate methods.**
    This is the pre-registered trigger for the **sub-2-bit pivot** below —
    NOT a model change.

## Pre-registered pivot: sub-2-bit (if 2-bit doesn't separate)

The trellis's structural edge (larger effective codebook at a fixed rate) is
**largest at sub-2-bit**, where scalar collapses to 3 (ternary) or 2 levels.
If comparison Y shows no separation at 2-bit, the next experiment is the
**same arms at n_bits=1** on the **same model** (`--n-bits 1`), not a model
hunt. Rationale (pre-mortem dialogue): model choice affects *power*, not
validity; the lever for power is bit-width. The scalar arms already accept
`--n-bits`; the trellis arm needs a sub-2-bit codebook (separate build, noted).

## Eval discipline (fixed in advance)

- Only **hardened** end-task PPL counts (pre-mortem #5/#6). Scalar arms have no
  soft→hard gap (uniform round = deployed op); BCJR hardens via Viterbi.
- Never compare on training/KL loss.
- Same WikiText-2 harness (`src/qat/eval_llama_layer.py`) for every arm.

## What would falsify "trellis is worth it"

scalar_ctrl (2.000 b, near-optimal scalar, same basis) matching or beating
STE-trellis AND BCJR-trellis within error bars. That is a publishable negative
result: "the trellis representational advantage does not survive QAT at 2-bit."
