#!/usr/bin/env bash
# Rung 3: compute PTQ baseline and print comparison table.
#
# Requires:
#   - cache/llama_bcjr_full_greedy/full_model_eval.json   (from Rung 1)
#   - Llama model at cache/model/Llama-3.2-1B (or $LLAMA_MODEL)
#
# Usage:
#   bash scripts/rung3_comparison.sh

set -euo pipefail
# Activate project venv if not already active (matches rung1_full_greedy.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$(dirname "$SCRIPT_DIR")/venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export BCJR_TRITON=1 BCJR_MONOLITHIC=1
MODEL="${LLAMA_MODEL:-cache/model/Llama-3.2-1B}"
mkdir -p logs results

echo "=== Rung 3: comparison table: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Step 1: full-model hard-Viterbi PTQ baseline
echo ""
echo "--- Step 1: full-model hard-Viterbi PTQ baseline ---"
python3 -u -m src.eval.run_llama_full_ptq_baseline \
    --model "$MODEL" \
    --output results/llama_full_ptq_ppl.json \
    --seq-len 2048 \
    --seed 0 \
    2>&1 | tee logs/rung3_ptq_baseline.log

# Step 2: print comparison table
echo ""
echo "--- Step 2: comparison table ---"
python3 -m scripts.rung3_comparison_table \
    --bcjr-result cache/llama_bcjr_full_greedy/full_model_eval.json \
    --ptq-result  results/llama_full_ptq_ppl.json \
    --ppl-fp 9.70

echo ""
echo "=== Rung 3 done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "Fill in ParetoQ / LC-QAT Llama-3.2-1B 2-bit numbers in:"
echo "  scripts/rung3_comparison_table.py  (FRONTIER list, top of file)"
