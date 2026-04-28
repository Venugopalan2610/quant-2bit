#!/usr/bin/env bash
# Path Y setup on vast.ai H200 NVL (140 GB).
#
# Assumptions:
#   - Base image: CUDA 12.8 + PyTorch 2.4+ (e.g. pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#     or vast.ai's nvidia/cuda:12.8 with pip torch).
#   - /workspace is the persistent mount.
#
# Upload from local:
#   rsync -avz --exclude 'cache/qat_bcjr_full_n4' \
#              --exclude 'cache/qat_ip_*' \
#              --exclude 'cache/qat/' \
#              --exclude 'cache/hidden_states' \
#              quant-olmoe-repo/ user@vast:/workspace/quant-olmoe/
#
# Then on the vast host:
#   bash scripts/vast_setup.sh
#   python -m src.qat.train_e2e_kl --use-adam8 --grad-ckpt \
#       --v2-dir cache/qat_bcjr_full_v2 \
#       --out-dir cache/qat_bcjr_e2e_kl \
#       --total-steps 4000 --lr 1e-5 --batch-seqs 2

set -euo pipefail

echo "=== vast_setup: Python + pip ==="
python3 --version
pip --version

echo "=== vast_setup: detect CUDA ==="
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "unknown")
echo "  nvcc reports CUDA ${CUDA_VER}"

echo "=== vast_setup: install base python deps ==="
pip install --upgrade pip
# Pick the torch wheel that matches the host CUDA (12.1 or 12.4 both work on H200).
case "$CUDA_VER" in
    12.1)      TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    12.4|12.5) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    12.6|12.7|12.8) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;  # 12.6+ runs cu124 wheels fine
    *)         TORCH_INDEX="" ;;
esac
if [ -n "$TORCH_INDEX" ]; then
    pip install 'torch>=2.4,<2.7' --index-url "$TORCH_INDEX"
else
    pip install 'torch>=2.4,<2.7'
fi
pip install \
    'transformers>=4.44' \
    accelerate \
    'datasets>=2.20' \
    numpy \
    safetensors \
    tqdm \
    'bitsandbytes>=0.43' \
    'lm-eval==0.4.11'

echo "=== vast_setup: build fast_hadamard_transform from source ==="
# pip wheel doesn't target CUDA 12.8; build from source.
if [ ! -d /workspace/fast-hadamard-transform ]; then
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
        /workspace/fast-hadamard-transform
fi
cd /workspace/fast-hadamard-transform
python3 setup.py install
cd -

echo "=== vast_setup: smoke test fast_hadamard_transform ==="
python3 - <<'PY'
import torch
from fast_hadamard_transform import hadamard_transform
x = torch.randn(1, 256, device="cuda", dtype=torch.float32)
y = hadamard_transform(x)
print(f"  FHT smoke: input {tuple(x.shape)} -> output {tuple(y.shape)}")
print(f"  norm ratio: {y.norm()/x.norm():.4f} (should be ~16 = sqrt(256))")
PY

echo "=== vast_setup: smoke test bitsandbytes AdamW8bit ==="
python3 - <<'PY'
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
loss = (p ** 2).sum(); loss.backward(); opt.step()
print(f"  AdamW8bit smoke: loss={loss.item():.4f}  bnb ver={bnb.__version__}")
PY

echo "=== vast_setup: download base model from HF ==="
# Base OLMoE (~14 GB). Faster than rsyncing from the 4080 host.
# Use the new `hf` CLI (huggingface-cli is deprecated in hub >= 0.26).
pip install 'huggingface_hub>=0.26'
mkdir -p cache/model
hf download allenai/OLMoE-1B-7B-0125 \
    --local-dir cache/model/olmoe-1b-7b-0125

echo "=== vast_setup: GPU + VRAM ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "=== vast_setup: verify repo layout ==="
ls cache/qat_bcjr_full_v2/ | wc -l   # expect 16
ls cache/calibration/
ls cache/model/olmoe-1b-7b-0125/ | head

echo "=== vast_setup: done ==="
