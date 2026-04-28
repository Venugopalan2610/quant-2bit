#!/usr/bin/env bash
# Path Y single-layer: BCJR + E2E KL on ONE decoder layer of OLMoE.
#
# Budget estimate: ~3-8 h wall clock on H200 for 100 steps on one layer.
# At $3.386/hr that's $10-27 — well inside the $78 remaining budget.
#
# Other 15 layers run at v2 PTQ (frozen). Only target layer's W_latent trains.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TARGET_LAYER="${1:-8}"   # default middle-ish layer if unspecified

mkdir -p logs cache/qat_bcjr_path_y_single

echo "=== Path Y single-layer (L${TARGET_LAYER}) start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u -m src.qat.train_single_layer_path_y \
    --v2-dir cache/qat_bcjr_full_v2 \
    --out-dir cache/qat_bcjr_path_y_single \
    --calib cache/calibration/tokens.npy \
    --target-layer "${TARGET_LAYER}" \
    --total-steps 100 \
    --batch-seqs 1 \
    --lr 1e-5 \
    --T-init 1.0 \
    --T-min 0.02 \
    --grad-clip 1.0 \
    --log-every 1 \
    --ckpt-every 25 \
    --bcjr-chunk 256 \
    --reencode-every 10 \
    --use-adam8 \
    --seed 0 \
    2>&1 | tee logs/path_y_single_L${TARGET_LAYER}.log

echo "=== Path Y single-layer (L${TARGET_LAYER}) done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
