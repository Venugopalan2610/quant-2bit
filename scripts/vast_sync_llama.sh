#!/usr/bin/env bash
# Upload code + Llama calibration to a vast.ai pod for the 15-step run.
# Total upload: ~few MB (no v2 snapshots, no models — Llama-1B is downloaded
# from HF on the pod by vast_setup_llama.sh).
#
# Usage:
#   bash scripts/vast_sync_llama.sh <host> <port> [remote_path]
#   bash scripts/vast_sync_llama.sh ssh4.vast.ai 12345
#   bash scripts/vast_sync_llama.sh ssh4.vast.ai 12345 /workspace/quant-olmoe
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
ssh ${SSH_OPTS} "root@${HOST}" "mkdir -p ${REMOTE}/cache/calibration ${REMOTE}/cache/llama_bcjr_single"

# Code + scripts (~few MB)
rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    src scripts requirements.txt \
    "root@${HOST}:${REMOTE}/"

# Llama calibration tokens (4 MB)
rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    cache/calibration/tokens_llama.npy \
    cache/calibration/tokens_llama.meta.txt \
    "root@${HOST}:${REMOTE}/cache/calibration/"

# Optional: the prior 3-step PTQ-init snapshot — for reference comparison only,
# not for resume. ~232 MB.
if [ -f cache/llama_bcjr_single/layer_08_wq_ptqinit.pt ]; then
    echo "=== uploading prior 3-step snapshot (reference) ==="
    rsync -avz --progress --partial \
        -e "ssh ${SSH_OPTS}" \
        cache/llama_bcjr_single/layer_08_wq_ptqinit.pt \
        "root@${HOST}:${REMOTE}/cache/llama_bcjr_single/"
fi

# Optional: the LR=2e-4 / 10-step layer-8 snapshot from the previous H100 run.
# Required for the multi-layer compounding eval. ~232 MB.
if [ -f cache/llama_bcjr_lr10x/layer_08_wq_ptqinit.pt ]; then
    echo "=== uploading LR=2e-4 layer-8 snapshot (for multi-layer eval) ==="
    ssh ${SSH_OPTS} "root@${HOST}" "mkdir -p ${REMOTE}/cache/llama_bcjr_lr10x"
    rsync -avz --progress --partial \
        -e "ssh ${SSH_OPTS}" \
        cache/llama_bcjr_lr10x/layer_08_wq_ptqinit.pt \
        "root@${HOST}:${REMOTE}/cache/llama_bcjr_lr10x/"
fi

echo "=== sync done ==="
echo "Next on the pod:"
echo "  cd ${REMOTE}"
echo "  export HF_TOKEN=<your-hf-token>     # required for gated Llama-3.2-1B"
echo "  bash scripts/vast_setup_llama.sh"
echo "  bash scripts/vast_train_llama_15step.sh           # smoke test (1 step) FIRST"
echo "  bash scripts/vast_train_llama_15step.sh full      # then full 15-step run"
