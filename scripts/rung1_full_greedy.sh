#!/usr/bin/env bash
# Rung 1: full-model 2-bit BCJR-QAT via per-layer greedy sequential.
#
# BEFORE RUNNING:
#   1. Rung 0 must PASS (3-seed L4 win confirmed).
#   2. Check VRAM on first layer: the first layer prints peak_vram.
#      If > 10 GB: reduce --seq-len to 512 or add --use-adam8.
#   3. Check step timing after layer 0. With 3 steps/layer × 16 layers,
#      multiply one-step wall time × 48 for the total estimate.
#
# HONEST LABEL: "per-layer greedy assembly," NOT "joint end-to-end."
# Gains may not show super-additivity (that is Rung 2, DEFERRED, H100+).
#
# SUCCESS CRITERION:
#   WikiText-2 PPL < full-model hard-Viterbi PTQ by > 0.42 PPL.
#   Honest target range: -0.5 to -1.7 PPL (low end = success).
#
# RESUME after interruption:
#   bash scripts/rung1_full_greedy.sh 8    # resume from layer 8
#
# Usage:
#   bash scripts/rung1_full_greedy.sh            # all 16 layers from 0
#   bash scripts/rung1_full_greedy.sh <N>        # resume from layer N
#   LLAMA_MODEL=/path/to/model bash scripts/rung1_full_greedy.sh

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BCJR_TRITON=1
export BCJR_MONOLITHIC=1

# Activate project venv if not already active
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$(dirname "$SCRIPT_DIR")/venv"
source "$VENV/bin/activate"

START_LAYER="${1:-0}"
MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"
OUT_DIR="cache/llama_bcjr_full_greedy"
LOG="logs/rung1_full_greedy.log"

mkdir -p logs "$OUT_DIR" results

echo "=== Rung 1: full-model BCJR-QAT greedy: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  model:       ${MODEL}"
echo "  start_layer: ${START_LAYER}"
echo "  out_dir:     ${OUT_DIR}"
echo "  log:         ${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""
echo "NOTE: check step timing after layer 0 before committing overnight."
echo "      First step print: 'dt=X.Xmin  peak_vram=X.XGB'"
echo "      If peak_vram > 10GB: Ctrl-C and add --use-adam8 or --seq-len 512."
echo ""

python3 -u -m src.qat.train_llama_full_greedy \
    --model "$MODEL" \
    --calib "$CALIB" \
    --out-dir "$OUT_DIR" \
    --start-layer "$START_LAYER" \
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
    --seed 0 \
    --eval-seq-len 2048 \
    2>&1 | tee -a "$LOG"

echo ""
echo "=== Rung 1 done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "BCJR greedy result:"
echo "  python3 -c \"import json; r=json.load(open('${OUT_DIR}/full_model_eval.json')); print(f'PPL={r[\\\"wikitext2_ppl\\\"]:.4f}')\""
echo ""
echo "Next step — Rung 3 comparison table:"
echo "  bash scripts/rung3_comparison.sh"
