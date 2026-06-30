#!/usr/bin/env bash
# Matched-budget trellis-vs-scalar comparison on L4 (see PREREGISTRATION.md).
#
# Runs the DECISIVE comparison Y first — STE-trellis vs scalar_ctrl, both
# 2.000b, same RHT basis, same FP init, identity-STE, 3 seeds. Then evals each
# hardened snapshot with the SAME WikiText-2 harness used for BCJR.
#
# These arms have NO BCJR kernel, so steps are seconds, not 82 min — the whole
# thing is GPU-hours. Launch in Phase B after pausing Rung 1 (which resumes via
# `bash scripts/rung1_full_greedy.sh <next_layer>`).
#
# Usage:
#   bash scripts/rung_matched_l4.sh              # Y: ste_trellis + scalar_ctrl
#   WITH_FAITHFUL=1 bash scripts/rung_matched_l4.sh   # + scalar_faithful (X)
#   N_BITS=1 bash scripts/rung_matched_l4.sh      # sub-2-bit pivot (scalar arms)
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BCJR_TRITON=1 BCJR_MONOLITHIC=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
source "$REPO/venv/bin/activate"
cd "$REPO"

MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"
OUT="cache/llama_matched"
LAYER=4
N_BITS="${N_BITS:-2}"
mkdir -p "$OUT" logs results

ARMS=(ste_trellis scalar_ctrl)
[ "${WITH_FAITHFUL:-0}" = "1" ] && ARMS+=(scalar_faithful)

echo "=== Matched-arm L4: arms=[${ARMS[*]}]  n_bits=$N_BITS  $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || true

for ARM in "${ARMS[@]}"; do
  for SEED in 0 1 2; do
    SNAP="$OUT/layer_04_${ARM}_fp_seed${SEED}.pt"
    TLOG="logs/matched_${ARM}_seed${SEED}_train.log"
    ELOG="logs/matched_${ARM}_seed${SEED}_eval.log"
    echo ""
    echo "──── arm=$ARM seed=$SEED  $(date -u +%H:%M:%SZ) ────"
    python3 -u -m src.qat.train_llama_matched_arm \
        --model "$MODEL" --calib "$CALIB" --out-dir "$OUT" \
        --arm "$ARM" --init fp --target-layer "$LAYER" \
        --total-steps 10 --seq-len 512 --lr 2e-4 --grad-clip 1.0 \
        --n-bits "$N_BITS" --group-size 128 --use-adam8 --seed "$SEED" \
        2>&1 | tee "$TLOG"

    # Hardened end-task PPL via the same harness as BCJR (quantizer-agnostic).
    python3 -u -m src.qat.eval_llama_layer \
        --model "$MODEL" --target-layer "$LAYER" \
        --snapshot "$SNAP" --seq-len 2048 --seed "$SEED" \
        2>&1 | tee "$ELOG"
  done
done

echo ""
echo "=== done $(date -u +%FT%TZ) ==="
echo "Summarize:  python3 -m scripts.rung_matched_summary  (PPL per arm, n=3)"
echo "Compare against BCJR L4: -0.0809 / -0.1161 / -0.0439 (mean -0.080 ± 0.036)"
