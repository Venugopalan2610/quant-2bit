#!/usr/bin/env bash
# Generalization sweep: the controlled trellis-vs-scalar comparison on several
# layers spanning network depth, to test whether the L4 win holds across layers
# (pre-mortem risk #12 — the biggest live risk). Cheap arms only (no BCJR
# kernel): per layer/seed we measure
#   - trellis-PTQ (free, from eval's PPL_HardViterbi)
#   - scalar-PTQ   (scalar_ctrl, 0 steps)  → REPRESENTATION gap, no training
#   - faithful-PTQ (scalar_faithful, 0 steps, per-group MSE, ~2.125b)
#   - STE-trellis  (10 steps)              ┐ comparison Y: representation
#   - scalar_ctrl  (10 steps)              ┘ under matched STE training
# trellis-PTQ vs scalar/faithful-PTQ isolates representation; STE-trellis vs
# scalar_ctrl isolates representation-under-training. (BCJR per layer is the
# expensive arm — deferred; the full-model BCJR story is Rung 1.)
#
# Eval log names include the layer + a per-arm tag so nothing collides.
# Usage:  bash scripts/rung_controlled_multilayer.sh
#         LAYERS="2 9 13" bash scripts/rung_controlled_multilayer.sh
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BCJR_TRITON=1 BCJR_MONOLITHIC=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
source "$REPO/venv/bin/activate"
cd "$REPO"
MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"
OUT="cache/llama_matched"
LAYERS="${LAYERS:-1 6 11 15}"
mkdir -p "$OUT" logs results

run () {  # layer arm steps tag
  local LAYER=$1 ARM=$2 STEPS=$3 TAG=$4 PAD
  PAD=$(printf '%02d' "$LAYER")
  for SEED in 0 1 2; do
    local SNAP="$OUT/layer_${PAD}_${ARM}_fp_seed${SEED}.pt"
    local EL="logs/matched_L${LAYER}_${TAG}_seed${SEED}_eval.log"
    echo "──── L$LAYER $TAG seed$SEED ($(date -u +%H:%M:%SZ)) ────"
    python3 -u -m src.qat.train_llama_matched_arm \
      --model "$MODEL" --calib "$CALIB" --out-dir "$OUT" \
      --arm "$ARM" --init fp --target-layer "$LAYER" \
      --total-steps "$STEPS" --seq-len 512 --lr 2e-4 --grad-clip 1.0 \
      --n-bits 2 --group-size 128 --use-adam8 --seed "$SEED" \
      > "logs/matched_L${LAYER}_${TAG}_seed${SEED}_train.log" 2>&1
    python3 -u -m src.qat.eval_llama_layer \
      --model "$MODEL" --target-layer "$LAYER" --snapshot "$SNAP" \
      --seq-len 2048 --seed "$SEED" > "$EL" 2>&1
    echo "  L$LAYER $TAG seed$SEED PPL=$(grep 'PPL_BCJR =' "$EL" | grep -oE '[0-9.]+$')"
  done
}

echo "=== controlled multilayer sweep: layers=[$LAYERS]  $(date -u +%FT%TZ) ==="
for LAYER in $LAYERS; do
  run "$LAYER" scalar_ctrl     0  scalarPTQ     # representation gap (global)
  run "$LAYER" scalar_faithful 0  faithfulPTQ   # representation gap (per-group)
  run "$LAYER" ste_trellis     10 steTrellis    # Y arm
  run "$LAYER" scalar_ctrl     10 scalarCtrl    # Y arm
done
echo "=== sweep done $(date -u +%FT%TZ) ==="
echo "Summarize: python3 -m scripts.rung_controlled_multilayer_summary"
