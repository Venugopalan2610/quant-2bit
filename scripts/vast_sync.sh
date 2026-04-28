#!/usr/bin/env bash
# Path Y: upload code + v2 snapshots + calibration to a vast.ai H200 pod.
#
# Usage:
#   bash scripts/vast_sync.sh <host> <port> [remote_path]
# Example:
#   bash scripts/vast_sync.sh 1.2.3.4 12345
#   bash scripts/vast_sync.sh ssh4.vast.ai 12345 /workspace/quant-olmoe
#
# Skips the 14 GB base model — re-download on the pod via `huggingface-cli`
# (handled by vast_setup.sh). v2 snapshots (~26 GB) dominate the upload.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <host> <port> [remote_path]" >&2
    exit 1
fi

HOST="$1"
PORT="$2"
REMOTE="${3:-/workspace/quant-olmoe}"

SSH_OPTS="-p ${PORT} -o StrictHostKeyChecking=accept-new"

echo "=== sync target: root@${HOST}:${REMOTE} (port ${PORT}) ==="
ssh ${SSH_OPTS} "root@${HOST}" "mkdir -p ${REMOTE}/cache"

# --- code + scripts (small, fast) ---
rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    src scripts requirements.txt \
    "root@${HOST}:${REMOTE}/"

# --- v2 snapshots (26 GB) ---
rsync -avz --progress --partial --inplace \
    -e "ssh ${SSH_OPTS}" \
    cache/qat_bcjr_full_v2/ \
    "root@${HOST}:${REMOTE}/cache/qat_bcjr_full_v2/"

# --- calibration tokens (33 MB) ---
rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    cache/calibration/tokens.npy cache/calibration/meta.txt \
    "root@${HOST}:${REMOTE}/cache/calibration/"

echo "=== sync done ==="
echo "Next on the pod:"
echo "  cd ${REMOTE} && bash scripts/vast_setup.sh && bash scripts/vast_train_path_y.sh"
