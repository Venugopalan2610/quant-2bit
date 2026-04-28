#!/usr/bin/env bash
# Resume an interrupted Llama BCJR-QAT run from a per-step W_latent ckpt.
#
# Use case: instance died at step 17 of 30. The ckpt at step 15
# (cache/llama_bcjr_30step/layer_08_step15_ptqinit_latent.pt) is intact.
# Resume from step 15, run steps 15-29.
#
# Usage:
#   bash scripts/vast_resume_llama.sh <ckpt_path> [total_steps]
#   bash scripts/vast_resume_llama.sh cache/llama_bcjr_30step/layer_08_step15_ptqinit_latent.pt
#   bash scripts/vast_resume_llama.sh cache/llama_bcjr_30step/layer_08_step15_ptqinit_latent.pt 30
#
# Cost on H100 SXM at ~$3/hr:
#   Resume from step 15 → 15 more steps → ~6.85h → ~$20
#   Resume from step 25 → 5 more steps   → ~2.3h  → ~$7
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${1:?usage: $0 <ckpt_path> [total_steps]}"
TOTAL_STEPS="${2:-30}"

if [ ! -f "$CKPT" ]; then
    echo "FAIL: ckpt not found: $CKPT" >&2
    exit 1
fi

OUT_DIR="$(dirname "$CKPT")"
LOG="logs/llama_resume_$(basename "$CKPT" .pt).log"
mkdir -p logs

LLAMA_LOCAL="cache/model/Llama-3.2-1B"
if [ ! -d "$LLAMA_LOCAL" ]; then
    echo "FAIL: ${LLAMA_LOCAL} missing. Run vast_setup_llama.sh first." >&2
    exit 1
fi

echo "=== Llama BCJR-QAT RESUME start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  ckpt:        ${CKPT}"
echo "  total_steps: ${TOTAL_STEPS}"
echo "  out_dir:     ${OUT_DIR}"
echo "  log:         ${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==="

python3 -u -m src.qat.train_llama_single_layer \
    --model "$LLAMA_LOCAL" \
    --calib cache/calibration/tokens_llama.npy \
    --out-dir "$OUT_DIR" \
    --target-layer 8 \
    --total-steps "$TOTAL_STEPS" \
    --batch-seqs 1 \
    --seq-len 1024 \
    --lr 2e-5 \
    --T-init 1.0 \
    --T-min 0.02 \
    --grad-clip 1.0 \
    --bcjr-chunk 16 \
    --reencode-every 1 \
    --resume-from "$CKPT" \
    --use-adam8 \
    --ckpt-every 5 \
    --seed 0 \
    2>&1 | tee "$LOG"

echo "=== Llama BCJR-QAT RESUME done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "Final snapshot: ${OUT_DIR}/layer_08_wq_ptqinit.pt"
echo "Trajectory ckpts also in ${OUT_DIR}/"
echo ""
echo "Next: bash scripts/vast_eval_llama_post.sh ${OUT_DIR}/layer_08_wq_ptqinit.pt"
