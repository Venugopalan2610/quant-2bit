#!/usr/bin/env bash
# Llama-3.2-1B BCJR-QAT — 10-step PTQ-init at LR=2e-4 (10× previous run).
#
# Hypothesis from Appendix A drift-budget theory:
#   per-element drift over N steps = N * η * |g|_max
#   For LR=2e-4, N=10: drift = 10 * 2e-4 * 1.0 = 2e-3
#   r_Voronoi ≈ σ_w / sqrt(2π·S) ≈ 1e-3
#   → drift = 2× r_Voronoi → BASIN ESCAPE PREDICTED
#
# Decision rules for the result:
#   PPL ≤ 10.20  → STRONG support for theory; LR knob is the right fix.
#                  Apply Scenario A drafts. Headline win.
#   PPL ≈ 10.31  → theory bound is loose; basin escape requires more than
#                  just drift > r_Voronoi. Refine theory in Appendix A.
#                  Still publishable; tightens the prediction.
#   PPL > 10.50 / NaN → LR too aggressive at low T. Try LR=1e-4 next.
#
# T schedule capped at 0.05 (not 0.02) — avoids the sharp-gradient regime
# that caused late-T quench-jumps in the 30-step run. Soft codeword is
# still ~90% concentrated on Viterbi at T=0.05, so hardening is meaningful.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="cache/llama_bcjr_lr10x"
LOG="logs/llama_lr10x.log"

mkdir -p logs "$OUT_DIR"

LLAMA_LOCAL="cache/model/Llama-3.2-1B"
if [ ! -d "$LLAMA_LOCAL" ]; then
    echo "FAIL: ${LLAMA_LOCAL} missing. Run vast_setup_llama.sh first." >&2
    exit 1
fi

echo "=== Llama BCJR-QAT LR=2e-4 / 10-step start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  out_dir:    ${OUT_DIR}"
echo "  log:        ${LOG}"
echo "  LR:         2e-4 (10× the 30-step run)"
echo "  steps:      10"
echo "  T schedule: 1.0 -> 0.05 (capped)"
echo "  ckpt_every: 2 (saves snapshots at steps 2, 4, 6, 8, 10)"
echo "  predicted cumulative drift: ~2e-3 = 2× r_Voronoi"
echo "==="

python3 -u -m src.qat.train_llama_single_layer \
    --model "$LLAMA_LOCAL" \
    --calib cache/calibration/tokens_llama.npy \
    --out-dir "$OUT_DIR" \
    --target-layer 8 \
    --total-steps 10 \
    --batch-seqs 1 \
    --seq-len 1024 \
    --lr 2e-4 \
    --T-init 1.0 \
    --T-min 0.05 \
    --grad-clip 1.0 \
    --bcjr-chunk 16 \
    --reencode-every 1 \
    --init-from-ptq \
    --use-adam8 \
    --ckpt-every 2 \
    --seed 0 \
    2>&1 | tee "$LOG"

echo
echo "=== Llama BCJR-QAT LR=2e-4 done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Final snapshot: ${OUT_DIR}/layer_08_wq_ptqinit.pt"
echo "Trajectory ckpts: ${OUT_DIR}/layer_08_step{02,04,06,08,10}_ptqinit_latent.pt"
echo
echo "Next: trajectory eval"
echo "  python3 -m scripts.eval_llama_trajectory \\"
echo "    --ckpt-dir ${OUT_DIR} \\"
echo "    --target-layer 8 \\"
echo "    --output results/llama_lr10x_trajectory.json \\"
echo "    --skip-baselines --ppl-fp-cached 9.70 --ppl-ptq-cached 10.31"
