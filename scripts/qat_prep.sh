#!/usr/bin/env bash
# qat_prep.sh — Lighter prep for QAT (no Hessians, no PTQ).
#
# Produces the minimum needed to run per-expert QAT teacher-matching:
#   cache/calibration/tokens.npy            (C4/Dolma packed token ids)
#   cache/hidden_states/layer_{NN}_input/   (FP16 teacher hidden states per layer)
#
# Not produced (QAT doesn't need these):
#   cache/hessian/                          (PTQ weighted-loss artifact)
#   cache/quantized/                        (PTQ output)
#
# Defaults are scoped for Q.1 (single-layer, single-expert teacher matching):
#   - N_SEQS=256, SEQ_LEN=1024                (smaller than PTQ's 2048×1024)
#   - END_LAYER=1                              (only layers 0-1 hidden states)
#
# Scale these up for Q.2 (per-layer all experts) or Q.3 (full 16 layers) via env.
#
# Usage:
#   bash scripts/qat_prep.sh
#   N_SEQS=2048 END_LAYER=16 bash scripts/qat_prep.sh   # full pipeline
set -euo pipefail

N_SEQS="${N_SEQS:-256}"
SEQ_LEN="${SEQ_LEN:-1024}"
END_LAYER="${END_LAYER:-1}"
SHARD_BATCH_SIZE="${SHARD_BATCH_SIZE:-8}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "QAT prep — N_SEQS=$N_SEQS SEQ_LEN=$SEQ_LEN END_LAYER=$END_LAYER"

if [ ! -e "cache/model/olmoe-1b-7b-0125/config.json" ]; then
  echo "FAIL: cache/model/olmoe-1b-7b-0125/config.json not found."
  echo "      Symlink or place the FP16 OLMoE-1B-7B-0125 checkpoint there."
  exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"

log "Step 1/2: calibration tokens (cache/calibration/tokens.npy)"
python3 -m src.hessian.prepare_calib \
  --n_seqs "$N_SEQS" \
  --seq_len "$SEQ_LEN" \
  --seed 0

log "Step 2/2: teacher hidden states (cache/hidden_states/layer_*_input/)"
log "  layers 0..$END_LAYER (inputs to each)"
python3 -m src.finetune.regenerate_hidden_states \
  --shard-batch-size "$SHARD_BATCH_SIZE" \
  --start-layer 0 \
  --end-layer "$END_LAYER"

log "QAT prep complete."
log "  Ready for Q.1 per-expert QAT (needs cache/hidden_states/layer_00_input/)."
