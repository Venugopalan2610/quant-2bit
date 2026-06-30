#!/usr/bin/env bash
# Rung 0: single-layer L4 BCJR-QAT, 3 seeds.
#
# PURPOSE: Verify the -0.084 PPL win is reproducible across seeds before
# committing to the 16-layer Rung 1 run. If seeds straddle zero, the
# headline was noise — stop and fix before scaling.
#
# SUCCESS CRITERION: BCJR beats PTQ by >= 0.04 PPL on mean - 1σ.
#
# TIMING: ~3 x (10 steps x step_time) + 3 x eval_time on your 4080.
# Check step timing from seed 0 log before the run is done.
#
# Usage:
#   bash scripts/rung0_multiseed.sh          # L4 (default)
#   bash scripts/rung0_multiseed.sh 8        # L8
#   LLAMA_MODEL=/path/to/model bash scripts/rung0_multiseed.sh
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BCJR_TRITON=1
export BCJR_MONOLITHIC=1

LAYER="${1:-4}"
MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"

mkdir -p logs results
LAYER_PAD="$(printf '%02d' "$LAYER")"

echo "=== Rung 0: L${LAYER} BCJR-QAT 3-seed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  model:  ${MODEL}"
echo "  layer:  ${LAYER}"
echo "  calib:  ${CALIB}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

for SEED in 0 1 2; do
    OUT_DIR="cache/llama_rung0/seed${SEED}"
    TRAIN_LOG="logs/rung0_train_seed${SEED}_L${LAYER}.log"
    EVAL_LOG="logs/rung0_eval_seed${SEED}_L${LAYER}.log"
    SNAP="${OUT_DIR}/layer_${LAYER_PAD}_wq_ptqinit.pt"

    mkdir -p "$OUT_DIR"

    echo "────────────────────────────────────"
    echo "Seed ${SEED}: training  $(date -u +%H:%M:%SZ)"
    echo "  out_dir: ${OUT_DIR}"
    echo "  log:     ${TRAIN_LOG}"

    python3 -u -m src.qat.train_llama_single_layer \
        --model "$MODEL" \
        --calib "$CALIB" \
        --out-dir "$OUT_DIR" \
        --target-layer "$LAYER" \
        --total-steps 10 \
        --batch-seqs 1 \
        --seq-len 512 \
        --lr 2e-4 \
        --T-init 0.3 \
        --T-min 0.05 \
        --grad-clip 1.0 \
        --bcjr-chunk 8 \
        --reencode-every 1 \
        --init-from-ptq \
        --use-adam8 \
        --ckpt-every 0 \
        --seed "$SEED" \
        2>&1 | tee "$TRAIN_LOG"

    echo "Seed ${SEED}: training done  $(date -u +%H:%M:%SZ)"

    echo "Seed ${SEED}: eval  $(date -u +%H:%M:%SZ)"
    echo "  snapshot: ${SNAP}"
    echo "  log:      ${EVAL_LOG}"

    python3 -u -m src.qat.eval_llama_layer \
        --model "$MODEL" \
        --target-layer "$LAYER" \
        --snapshot "$SNAP" \
        --seq-len 2048 \
        --seed "$SEED" \
        2>&1 | tee "$EVAL_LOG"

    echo "Seed ${SEED}: eval done  $(date -u +%H:%M:%SZ)"
    echo ""
done

echo "=== All 3 seeds done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "Summary table:"
echo "  python3 -m scripts.rung0_summary --layer ${LAYER}"
echo ""
echo "Decision rule:"
echo "  mean ΔPPL - 1σ > 0.04  → PASS, proceed to Rung 1"
echo "  seeds straddle 0        → FAIL, fix before scaling"
