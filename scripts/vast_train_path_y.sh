#!/usr/bin/env bash
# Path Y: full-model BCJR E2E KL distillation on H200 NVL (140 GB).
#
# Budget: ~11-12 h wall clock on H200, ~$25-30 at $2.50/hr.
# Prerequisites: vast_sync.sh + vast_setup.sh have run.
#
# Logs stream to logs/path_y.log; also mirrored to stdout.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs cache/qat_bcjr_e2e_kl

echo "=== Path Y training start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u -m src.qat.train_e2e_kl \
    --v2-dir cache/qat_bcjr_full_v2 \
    --out-dir cache/qat_bcjr_e2e_kl \
    --calib cache/calibration/tokens.npy \
    --total-steps 100 \
    --batch-seqs 1 \
    --lr 1e-5 \
    --T-init 1.0 \
    --T-min 0.02 \
    --grad-clip 1.0 \
    --log-every 1 \
    --eval-every 25 \
    --ckpt-every 25 \
    --mode bcjr \
    --bcjr-chunk 32 \
    --reencode-every 10 \
    --use-adam8 \
    --seed 0 \
    2>&1 | tee logs/path_y.log

echo "=== Path Y training done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
