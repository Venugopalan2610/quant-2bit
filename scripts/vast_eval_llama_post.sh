#!/usr/bin/env bash
# Post-training PPL eval for the 15-step Llama BCJR-QAT snapshot.
#
# Compares (a) FP baseline 9.70, (b) hard-Viterbi PTQ on layer 8 = 10.31,
# (c) trained snapshot from this run.
#
# Usage:
#   bash scripts/vast_eval_llama_post.sh cache/llama_bcjr_15step/layer_08_wq_ptqinit.pt
#
# Decision rule for the paper:
#   PPL < 10.31  → BCJR-QAT beats PTQ on Llama-1B at 15 steps. New positive
#                  datapoint for §4.5. Add to Table 2, refresh §5.4.
#   PPL ≈ 10.31  → 15 steps still insufficient to escape Voronoi basin.
#                  §5.4 limitation tightens; hail-mary failed cleanly.
#   PPL > 10.31  → overshoot / overfitting. Try lower LR (1e-5) on remaining budget.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SNAP="${1:-cache/llama_bcjr_15step/layer_08_wq_ptqinit.pt}"
if [ ! -f "$SNAP" ]; then
    echo "FAIL: snapshot not found: $SNAP" >&2
    exit 1
fi

mkdir -p logs results
LOG="logs/llama_eval_$(basename "$SNAP" .pt).log"

echo "=== Llama PPL eval start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  snapshot: $SNAP"
echo "  log: $LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LLAMA_LOCAL="cache/model/Llama-3.2-1B"
python3 -u -m src.qat.eval_llama_layer \
    --model "$LLAMA_LOCAL" \
    --target-layer 8 \
    --snapshot "$SNAP" \
    2>&1 | tee "$LOG"

echo "=== Llama PPL eval done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "Reference numbers from §4.5 of the paper:"
echo "  FP16 baseline:                   9.70"
echo "  Hard Viterbi PTQ (layer 8 only): 10.31"
echo "  3-step BCJR PTQ-init (4080):     10.35  (+0.04 vs PTQ)"
echo ""
echo "Look for the (c) BCJR-trained line in the log above."
