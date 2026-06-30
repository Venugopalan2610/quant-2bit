#!/usr/bin/env bash
# BCJR per-layer amenability sweep — the $100 H100 experiment.
# Single-layer BCJR-QAT on a spread of layers × seeds, each evaluated vs PTQ.
# Adds the EXPENSIVE BCJR arm (with error bars) to the layers the local
# generalization sweep already covers with cheap scalar/STE arms → completes
# the per-layer picture and answers "which layers benefit from trellis QAT."
#
# HYPERPARAMS ARE IDENTICAL TO THE L4 RUN (so layers are comparable): seq-len
# 512, T 0.3→0.05, lr 2e-4, 10 steps, init-from-ptq. Only --bcjr-chunk is
# bumped (32) — that's memory/speed only, NUMERICALLY IDENTICAL, and the H100
# has the VRAM for it.
#
# Resumable: a finished layer/seed (final snapshot present) is skipped, so
# re-running after an interruption continues. Recommend on-demand (not spot).
#
#   LAYERS="1 7 11 15" SEEDS="0 1 2" bash vast/bcjr_amenability_sweep.sh
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BCJR_TRITON=1 BCJR_MONOLITHIC=1

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-cache/model/Llama-3.2-1B}"
CALIB="cache/calibration/tokens_llama.npy"
LAYERS="${LAYERS:-1 7 11 15}"
SEEDS="${SEEDS:-0 1 2}"
CHUNK="${CHUNK:-32}"          # H100: bigger chunk = faster, identical numerics
mkdir -p logs results

echo "=== BCJR amenability sweep: layers=[$LAYERS] seeds=[$SEEDS] $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

for L in $LAYERS; do
  PAD=$(printf '%02d' "$L")
  for S in $SEEDS; do
    OUT="cache/llama_amenability/seed${S}"
    SNAP="${OUT}/layer_${PAD}_wq_ptqinit.pt"
    ELOG="logs/amen_L${L}_seed${S}_eval.log"
    mkdir -p "$OUT"
    if [ -f "$SNAP" ] && grep -q "PPL_BCJR" "$ELOG" 2>/dev/null; then
      echo "── L$L seed$S already done ($(grep 'PPL_BCJR =' "$ELOG" | grep -oE '[0-9.]+$')) — skip"
      continue
    fi
    echo "──── L$L seed$S TRAIN $(date -u +%H:%M:%SZ) ────"
    python3 -u -m src.qat.train_llama_single_layer \
        --model "$MODEL" --calib "$CALIB" --out-dir "$OUT" \
        --target-layer "$L" --total-steps 10 --batch-seqs 1 --seq-len 512 \
        --lr 2e-4 --T-init 0.3 --T-min 0.05 --grad-clip 1.0 \
        --bcjr-chunk "$CHUNK" --reencode-every 1 --init-from-ptq --use-adam8 \
        --ckpt-every 5 --seed "$S" \
        2>&1 | tee "logs/amen_L${L}_seed${S}_train.log"
    echo "──── L$L seed$S EVAL ────"
    python3 -u -m src.qat.eval_llama_layer \
        --model "$MODEL" --target-layer "$L" --snapshot "$SNAP" \
        --seq-len 2048 --seed "$S" 2>&1 | tee "$ELOG"
    echo "  L$L seed$S: PPL=$(grep 'PPL_BCJR =' "$ELOG" | grep -oE '[0-9.]+$') "\
"(PTQ $(grep 'PPL_HardViterbi =' "$ELOG" | grep -oE '[0-9.]+$'))"
  done
done

echo ""
echo "=== sweep done $(date -u +%FT%TZ) ==="
echo "Per-layer BCJR vs PTQ (Δ<0 = BCJR wins):"
for L in $LAYERS; do
  for S in $SEEDS; do
    EL="logs/amen_L${L}_seed${S}_eval.log"
    b=$(grep 'PPL_BCJR =' "$EL" 2>/dev/null | grep -oE '[0-9.]+$')
    p=$(grep 'PPL_HardViterbi =' "$EL" 2>/dev/null | grep -oE '[0-9.]+$')
    [ -n "$b" ] && echo "  L$L seed$S: BCJR=$b PTQ=$p"
  done
done
