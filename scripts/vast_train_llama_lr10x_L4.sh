#!/usr/bin/env bash
# Llama-3.2-1B BCJR-QAT — 10-step PTQ-init at LR=2e-4, target LAYER 4.
#
# Same recipe as vast_train_llama_lr10x.sh but on a different layer.
# Tests whether the BCJR-QAT win at layer 8 (Δ ≈ -0.022 PPL) reproduces
# at a different layer in the network. The output snapshot will be combined
# with the layer-8 snapshot in scripts/eval_llama_multilayer.py to test
# whether per-layer wins compound.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="cache/llama_bcjr_lr10x_L4"
LOG="logs/llama_lr10x_L4.log"

mkdir -p logs "$OUT_DIR"

LLAMA_LOCAL="cache/model/Llama-3.2-1B"
if [ ! -d "$LLAMA_LOCAL" ]; then
    echo "FAIL: ${LLAMA_LOCAL} missing. Run vast_setup_llama.sh first." >&2
    exit 1
fi

echo "=== Llama BCJR-QAT LR=2e-4 / 10-step / LAYER 4 start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  out_dir:    ${OUT_DIR}"
echo "  log:        ${LOG}"
echo "  LR:         2e-4 (same as L8 run)"
echo "  steps:      10"
echo "  T schedule: 1.0 -> 0.05"
echo "  ckpt_every: 2"
echo "  target_layer: 4"
echo "==="

python3 -u -m src.qat.train_llama_single_layer \
    --model "$LLAMA_LOCAL" \
    --calib cache/calibration/tokens_llama.npy \
    --out-dir "$OUT_DIR" \
    --target-layer 4 \
    --total-steps 10 \
    --batch-seqs 1 \
    --seq-len 1024 \
    --lr 2e-4 \
    --T-init 1.0 \
    --T-min 0.05 \
    --grad-clip 1.0 \
    --bcjr-chunk 16 \
    --reencode-every 1 \
    --init-from-ptq \
    --use-adam8 \
    --ckpt-every 2 \
    --seed 0 \
    2>&1 | tee "$LOG"

echo
echo "=== Layer-4 BCJR done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Final snapshot: ${OUT_DIR}/layer_04_wq_ptqinit.pt"
echo
echo "Next: multi-layer eval (compounding test)"
echo "  python3 -m scripts.eval_llama_multilayer \\"
echo "    --target-layers 4,8 \\"
echo "    --snapshots ${OUT_DIR}/layer_04_wq_ptqinit.pt,cache/llama_bcjr_lr10x/layer_08_wq_ptqinit.pt \\"
echo "    --output results/llama_multilayer_compounding.json \\"
echo "    --ppl-fp-cached 9.70"
