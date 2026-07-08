#!/usr/bin/env bash
# Watch Rung 1 to completion, then auto-run Rung 3 (PTQ baseline + comparison
# table). Exits 0 on success, 1 if Rung 1 dies without a completion marker (so
# the background-task notification tells us a crash happened, not silence).
#
# Usage:  nohup bash scripts/watch_rung1_then_rung3.sh <rung1_log> &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-logs/rung1_resume2_l5.log}"
DONE_MARK="=== Rung 1 done"
POLL=300

echo "[watch] watching $LOG for Rung 1 completion (poll ${POLL}s)"
while true; do
  if grep -q "$DONE_MARK" "$LOG" 2>/dev/null; then
    echo "[watch] Rung 1 COMPLETE → launching Rung 3"
    bash scripts/rung3_comparison.sh 2>&1 | tee logs/rung3_auto.log
    echo "[watch] Rung 3 done → launching controlled multi-layer generalization sweep"
    bash scripts/rung_controlled_multilayer.sh 2>&1 | tee logs/multilayer_auto.log
    source venv/bin/activate 2>/dev/null || true
    python3 -m scripts.rung_controlled_multilayer_summary 2>&1 | tee logs/multilayer_summary.log
    echo "[watch] → KV composition eval (BCJR weights × real TurboQuant KV, 2×2)"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BCJR_TRITON=1 BCJR_MONOLITHIC=1
    python3 -m src.eval.eval_kv_composition \
        --weights-dir cache/llama_bcjr_full_greedy --seq-len 2048 \
        --kv-mode turboquant --key-bits 3 --val-bits 2 \
        2>&1 | tee logs/kv_composition_auto.log
    echo "[watch] → drift-budget analysis: does basin-escape predict greedy reversals?"
    python3 -m src.eval.drift_budget_analysis \
        --weights-dir cache/llama_bcjr_full_greedy \
        2>&1 | tee logs/drift_budget_auto.log
    echo "[watch] → FREE local isolated-BCJR diagnostic: L11,L13 × 2 seeds (chunk 8)"
    echo "[watch]   answers 'back-half trouble = the layer, or the greedy accumulation?'"
    source venv/bin/activate 2>/dev/null || true
    CHUNK=8 LAYERS="11 13" SEEDS="0 1" bash vast/bcjr_amenability_sweep.sh \
        2>&1 | tee logs/amenability_local_auto.log
    echo "[watch] ALL DONE. Rung3: logs/rung3_auto.log | sweep: logs/multilayer_summary.log"
    echo "[watch]   KV: logs/kv_composition_auto.log | diagnostic: logs/amenability_local_auto.log"
    exit 0
  fi
  # liveness: is the greedy trainer still running?
  if ! pgrep -f "train_llama_full_greedy" >/dev/null 2>&1; then
    # process gone and no completion marker → crash/interruption
    LAST_LAYER=$(grep -oE "Layer [0-9]+ done" "$LOG" 2>/dev/null | tail -1)
    echo "[watch] FAILURE: Rung 1 process gone with no completion marker."
    echo "[watch] last completed: ${LAST_LAYER:-none}.  Resume with:"
    echo "[watch]   bash scripts/rung1_full_greedy.sh <next_layer>"
    exit 1
  fi
  sleep "$POLL"
done
