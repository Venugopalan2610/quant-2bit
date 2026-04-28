#!/usr/bin/env bash
# Llama-3.2-1B BCJR-QAT — 15-step PTQ-init E2E KL distillation on layer 8.
#
# This is the hail-mary: 5x the step count of the 4080 overnight run, with a
# proper anneal schedule that actually reaches T_min. Same LR as before
# (2e-5) so the only knob changing is W_latent drift budget.
#
# Modes:
#   bash scripts/vast_train_llama_15step.sh              # SMOKE: 1 step, ~1h on H100
#   bash scripts/vast_train_llama_15step.sh full         # FULL: 15 steps, ~10-12h on H100
#   bash scripts/vast_train_llama_15step.sh full 12      # FULL with custom step count
#
# Budget estimate at $2/hr H100 PCIe:
#   smoke (1 step including compile): ~$2
#   full 15 steps:                    ~$22
#   full 12 steps:                    ~$18
#
# What we look for in smoke output (vs the 4080 step-3 PTQ-init number 5.63e-2):
#   step 1 KL should land near 1.71e-1 (matching the 4080 step-1 PTQ-init).
#   If it's wildly different, STOP — kernel numerics differ between the two
#   GPUs and we need to debug before burning 10h of compute.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-smoke}"
TOTAL_STEPS="${2:-30}"
CKPT_EVERY="${3:-5}"

if [ "$MODE" = "smoke" ]; then
    STEPS=1
    OUT_DIR="cache/llama_bcjr_smoke"
    LOG="logs/llama_smoke.log"
    T_MIN=0.1   # match the 4080 step-1 anneal point so step-1 KL is comparable
elif [ "$MODE" = "full" ]; then
    STEPS="$TOTAL_STEPS"
    OUT_DIR="cache/llama_bcjr_${STEPS}step"
    LOG="logs/llama_${STEPS}step.log"
    T_MIN=0.02   # extended anneal: longer high-T exploration before sharpening
else
    echo "usage: $0 [smoke|full] [total_steps]" >&2
    exit 1
fi

mkdir -p logs "$OUT_DIR"

echo "=== Llama BCJR-QAT (mode=${MODE}, steps=${STEPS}) start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  out_dir=${OUT_DIR}"
echo "  log=${LOG}"
echo "  T schedule: 1.0 -> ${T_MIN} over ${STEPS} steps"
echo "==="

# Use the local Llama path so the trainer doesn't try to redownload
LLAMA_LOCAL="cache/model/Llama-3.2-1B"
if [ ! -d "$LLAMA_LOCAL" ]; then
    echo "FAIL: ${LLAMA_LOCAL} missing. Run vast_setup_llama.sh first." >&2
    exit 1
fi

CKPT_FLAG=""
if [ "$MODE" = "full" ] && [ "$CKPT_EVERY" -gt 0 ]; then
    CKPT_FLAG="--ckpt-every ${CKPT_EVERY}"
    echo "  ckpt-every: ${CKPT_EVERY}  (W_latent snapshots for trajectory + interruption insurance)"
fi

python3 -u -m src.qat.train_llama_single_layer \
    --model "$LLAMA_LOCAL" \
    --calib cache/calibration/tokens_llama.npy \
    --out-dir "$OUT_DIR" \
    --target-layer 8 \
    --total-steps "$STEPS" \
    --batch-seqs 1 \
    --seq-len 1024 \
    --lr 2e-5 \
    --T-init 1.0 \
    --T-min "$T_MIN" \
    --grad-clip 1.0 \
    --bcjr-chunk 16 \
    --reencode-every 1 \
    --init-from-ptq \
    --use-adam8 \
    $CKPT_FLAG \
    --seed 0 \
    2>&1 | tee "$LOG"

echo "=== Llama BCJR-QAT (mode=${MODE}) done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Snapshot: ${OUT_DIR}/layer_08_wq_ptqinit.pt"
echo ""
if [ "$MODE" = "smoke" ]; then
    echo "Next: verify step-1 KL is near 1.71e-1 (4080 reference)."
    echo "If yes: bash scripts/vast_train_llama_15step.sh full"
else
    echo "Next: bash scripts/vast_eval_llama_post.sh ${OUT_DIR}/layer_08_wq_ptqinit.pt"
fi
