#!/usr/bin/env bash
# Airtight-ladder cleanup for the matched-arm L4 result (see PREREGISTRATION.md).
# Adds the two cells needed to make the comparison reviewer-proof:
#   (1) scalar-PTQ  : scalar_ctrl, 0 training steps → isolates the PURE
#       representational gap vs trellis-PTQ (no training confound).
#   (2) STE-trellis from PTQ init → init-matches BCJR (which used PTQ init),
#       giving a CLEAN comparison Z (soft relaxation vs STE on the same trellis,
#       same start point).
# Both cheap (no BCJR kernel). Run while Rung 1 is paused.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BCJR_TRITON=1 BCJR_MONOLITHIC=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
source "$REPO/venv/bin/activate"
cd "$REPO"
MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"
OUT="cache/llama_matched"; LAYER=4
mkdir -p "$OUT" logs

run () {  # arm init steps tag
  local ARM=$1 INIT=$2 STEPS=$3 TAG=$4
  for SEED in 0 1 2; do
    SNAP="$OUT/layer_04_${ARM}_${INIT}_seed${SEED}.pt"
    echo "──── $TAG arm=$ARM init=$INIT steps=$STEPS seed=$SEED $(date -u +%H:%M:%SZ) ────"
    python3 -u -m src.qat.train_llama_matched_arm \
        --model "$MODEL" --calib "$CALIB" --out-dir "$OUT" \
        --arm "$ARM" --init "$INIT" --target-layer "$LAYER" \
        --total-steps "$STEPS" --seq-len 512 --lr 2e-4 --grad-clip 1.0 \
        --n-bits 2 --group-size 128 --use-adam8 --seed "$SEED" \
        2>&1 | tee "logs/matched_${TAG}_seed${SEED}_train.log"
    python3 -u -m src.qat.eval_llama_layer \
        --model "$MODEL" --target-layer "$LAYER" \
        --snapshot "$SNAP" --seq-len 2048 --seed "$SEED" \
        2>&1 | tee "logs/matched_${TAG}_seed${SEED}_eval.log"
  done
}

echo "=== cleanup: scalar-PTQ (pure representation) + STE-trellis-from-PTQ (clean Z) ==="
run scalar_ctrl fp  0  scalarPTQ
run ste_trellis ptq 10 steTrellisPTQinit
echo "=== cleanup done $(date -u +%FT%TZ) ==="
