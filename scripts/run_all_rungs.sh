#!/usr/bin/env bash
# Chain: Rung 0 → Rung 1 → Rung 3 (run overnight, unattended).
# Uses the project venv at quant-2bit/venv/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/venv"

# Activate project venv
source "$VENV/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BCJR_TRITON=1
export BCJR_MONOLITHIC=1

cd "$REPO_DIR"

mkdir -p logs results

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo "= Full experiment chain: $TIMESTAMP"
echo "= venv: $VENV"
echo "=========================================="

# ── Rung 0: 3-seed L4 validation ──────────────────────────────────────────
echo ""
echo "### RUNG 0: 3-seed L4 ($(date -u +%H:%M:%SZ)) ###"
bash scripts/rung0_multiseed.sh 4 2>&1 | tee logs/rung0_all.log

echo ""
echo "### Rung 0 summary: ###"
python3 -m scripts.rung0_summary --layer 4 2>&1 | tee logs/rung0_summary.log

# Gate on PASS
if ! grep -q "PASS" logs/rung0_summary.log; then
    echo ""
    echo "ERROR: Rung 0 did not PASS — aborting before Rung 1."
    echo "Check logs/rung0_summary.log for details."
    exit 1
fi

echo ""
echo "Rung 0 PASSED — proceeding to Rung 1."

# ── Rung 1: 16-layer greedy QAT ───────────────────────────────────────────
echo ""
echo "### RUNG 1: 16-layer greedy ($(date -u +%H:%M:%SZ)) ###"
bash scripts/rung1_full_greedy.sh 2>&1 | tee logs/rung1_all.log

# ── Rung 3: comparison table ──────────────────────────────────────────────
echo ""
echo "### RUNG 3: comparison table ($(date -u +%H:%M:%SZ)) ###"
bash scripts/rung3_comparison.sh 2>&1 | tee logs/rung3_all.log

echo ""
echo "=========================================="
echo "= All rungs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
