#!/usr/bin/env bash
# Llama-3.2-1B BCJR-QAT — skip-high-T schedule, target LAYER 4.
#
# HYPOTHESIS:
# In the previous L4 run with T_init=1.0, the bit-equivalence test showed
# step-0 PPL = production PTQ (10.2189) but step-2 PPL had drifted to
# 10.287 (+0.07 above PTQ). I.e., the FIRST TWO HIGH-T BCJR STEPS
# OVERSHOOT — they push W_latent away from the PTQ codeword in directions
# that the soft codeword "sees" as good but that don't correspond to
# better hard-Viterbi codewords.
#
# This script tests the constructive prescription: skip the high-T phase
# entirely. Start at T_init=0.3 (where soft codeword is ~70% concentrated
# on Viterbi MAP, so gradient signal is closer to STE), anneal to 0.05
# over 10 steps. Avoids the regime where overshoot happens.
#
# DECISION RULES for the result:
#   PPL ≤ 10.20  → big win, schedule fix works, BCJR-QAT clearly beats PTQ
#   PPL ≈ 10.21  → marginal win or null, comparable to PTQ
#   PPL > 10.22  → schedule is not the only problem; structural limitation
#
# Cost: same as the LR=2e-4 / N=10 run = ~$17 on H100 SXM.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="cache/llama_bcjr_skipT"
LOG="logs/llama_skipT.log"

mkdir -p logs "$OUT_DIR"

LLAMA_LOCAL="cache/model/Llama-3.2-1B"
if [ ! -d "$LLAMA_LOCAL" ]; then
    echo "FAIL: ${LLAMA_LOCAL} missing. Run vast_setup_llama.sh first." >&2
    exit 1
fi

echo "=== Llama BCJR-QAT skip-high-T / 10-step / LAYER 4 start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  out_dir:    ${OUT_DIR}"
echo "  log:        ${LOG}"
echo "  LR:         2e-4 (same as prior L4 run)"
echo "  steps:      10  (same)"
echo "  T schedule: 0.3 -> 0.05 (skip high-T to avoid overshoot)"
echo "  ckpt_every: 2"
echo "  target_layer: 4"
echo
echo "  PRIOR L4 run with T_init=1.0:"
echo "    step 0 (untrained):  10.2189 (= production PTQ)"
echo "    step 2:              10.2869 (drifted +0.07)"
echo "    step 10 (trained):   10.2243 (recovered to +0.005 above PTQ)"
echo
echo "  HYPOTHESIS: skipping T > 0.3 avoids the early overshoot."
echo "  If this run lands < 10.21, paper has a real headline win."
echo "==="

python3 -u -m src.qat.train_llama_single_layer \
    --model "$LLAMA_LOCAL" \
    --calib cache/calibration/tokens_llama.npy \
    --out-dir "$OUT_DIR" \
    --target-layer 4 \
    --total-steps 10 \
    --batch-seqs 1 \
    --seq-len 1024 \
    --lr 2e-4 \
    --T-init 0.3 \
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
echo "=== Skip-high-T training done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Final snapshot: ${OUT_DIR}/layer_04_wq_ptqinit.pt"
echo "Trajectory ckpts: ${OUT_DIR}/layer_04_step{02,04,06,08,10}_ptqinit_latent.pt"
echo
echo "Next: trajectory eval (compares each ckpt against the production PTQ baseline 10.2189)"
echo "  python3 -m scripts.eval_llama_trajectory \\"
echo "    --ckpt-dir ${OUT_DIR} \\"
echo "    --target-layer 4 \\"
echo "    --output results/llama_skipT_trajectory.json \\"
echo "    --skip-baselines --ppl-fp-cached 9.70 --ppl-ptq-cached 10.2189"
