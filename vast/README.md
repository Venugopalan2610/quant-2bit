# Vast.ai H100 kit — BCJR per-layer amenability sweep ($100 budget)

**Goal:** add the expensive **BCJR arm (with error bars)** to layers L1/7/11/15,
completing the per-layer picture the local generalization sweep covers with the
cheap scalar/STE arms. Answers the #1 pitch gap (does the L4 win generalize
across depth?) and your paper's open question (which layers benefit from trellis
QAT). Runs fully in parallel with the local 4080's Rung 1.

## Budget math (H100 @ ~$2–2.5/hr)
- ~$100 ≈ **40–50 H100-hours**.
- Each single-layer BCJR run ≈ **~3–4h** on H100 (vs ~14h on the 4080).
- L1/7/11/15 × 3 seeds = **12 runs ≈ ~42h ≈ ~$85–95** — fits, with a little buffer.
- Want it cheaper/faster? Drop to 2 seeds (8 runs) or 3 layers.

## Two paths to set up

**A. Docker (reproducible):** build & push `vast/Dockerfile`, rent an H100 with it.
**B. Stock instance (fastest to start):** rent a Vast H100 with any recent
PyTorch image, then:
```
export HF_TOKEN=hf_xxx          # Llama-3.2 is gated
bash vast/bootstrap.sh          # deps + FHT build + repo + model + calib
bash vast/bcjr_amenability_sweep.sh
```

## Spend-protection (or you'll burn the $100)
1. **On-demand, NOT interruptible/spot** for the ~3–4h runs — a spot kill mid-run
   wastes money. (The sweep also checkpoints every 5 steps and skips finished
   layer/seeds on re-run, so a restart resumes — but on-demand is simpler.)
2. **`tmux`/`nohup`** the sweep so an SSH drop doesn't kill it:
   `nohup bash vast/bcjr_amenability_sweep.sh > sweep.log 2>&1 &`
3. **Pull results, then DESTROY the instance** (Vast bills while it exists, even
   stopped, for storage). Copy `cache/llama_amenability/` + `logs/amen_*` off first:
   `scp -r ... cache/llama_amenability logs/`
4. **First run builds the Triton BCJR kernel (~1 min, one-time)** and downloads
   the model + streams C4 for calib — budget ~15–20 min of setup before step 1.

## What you get
`logs/amen_L*_seed*_eval.log` → BCJR PPL and PTQ PPL per layer/seed. Combined
with L4 (local), that's **5 layers × 3 seeds of BCJR-vs-PTQ with error bars** —
the generalization evidence that turns the single-layer result into a paper.

## Hyperparameters
Identical to the L4 run for comparability (seq-len 512, T 0.3→0.05, lr 2e-4,
10 steps, init-from-ptq). Only `--bcjr-chunk` is raised to 32 (memory/speed
only — numerically identical; H100 has the VRAM).
