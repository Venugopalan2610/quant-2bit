#!/usr/bin/env bash
# Vast.ai setup for the Llama-3.2-1B BCJR-QAT 15-step run.
#
# Lighter than vast_setup.sh (the OLMoE Path-Y version) — no v2 snapshots,
# no MoE deps, no fast_hadamard_transform. Llama-1B is ~2.5 GB; downloads
# fast on a pod.
#
# Assumes:
#   - Base image: PyTorch 2.4+ with CUDA 12.1+ (e.g. pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#     or vast.ai's nvidia/cuda:12.4 with pip torch).
#   - /workspace mount (or just $PWD).
#
# Run AFTER vast_sync_llama.sh has uploaded code + tokens_llama.npy.
set -euo pipefail

echo "=== vast_setup_llama: GPU + Python ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
pip --version

echo "=== vast_setup_llama: install deps ==="
pip install --upgrade pip
# Torch — match host CUDA. H100 PCIe pods are usually 12.1/12.4.
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
case "$CUDA_VER" in
    12.1) IDX="https://download.pytorch.org/whl/cu121" ;;
    12.4|12.5|12.6|12.7|12.8) IDX="https://download.pytorch.org/whl/cu124" ;;
    *)    IDX="" ;;
esac
if [ -n "$IDX" ]; then
    pip install 'torch>=2.4,<2.7' --index-url "$IDX"
else
    pip install 'torch>=2.4,<2.7'
fi
pip install \
    'transformers>=4.44' \
    accelerate \
    'datasets>=2.20' \
    numpy \
    scipy \
    safetensors \
    tqdm \
    triton \
    'bitsandbytes>=0.43' \
    'huggingface_hub>=0.26'

echo "=== vast_setup_llama: build fast_hadamard_transform from source ==="
# The pip wheel doesn't target recent CUDAs (12.4+). Build from source.
if [ ! -d /workspace/fast-hadamard-transform ]; then
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
        /workspace/fast-hadamard-transform
fi
( cd /workspace/fast-hadamard-transform && python3 setup.py install )
python3 -c "
import torch
from fast_hadamard_transform import hadamard_transform
x = torch.randn(1, 256, device='cuda', dtype=torch.float32)
y = hadamard_transform(x)
print(f'  FHT smoke: norm ratio {y.norm()/x.norm():.4f} (should be ~16 = sqrt(256))')
"

echo "=== vast_setup_llama: HF login (set HF_TOKEN env beforehand for Llama gated access) ==="
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "  WARN: HF_TOKEN not set. Llama-3.2-1B is gated; download will fail."
    echo "  export HF_TOKEN=<your-hf-token>  before running this script."
fi

echo "=== vast_setup_llama: download Llama-3.2-1B (~2.5 GB) ==="
mkdir -p cache/model
hf download meta-llama/Llama-3.2-1B \
    --local-dir cache/model/Llama-3.2-1B
# Patch the script to use the local path so it doesn't redownload.
echo "  Llama files at: cache/model/Llama-3.2-1B"

echo "=== vast_setup_llama: smoke tests ==="
python3 - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  bf16={torch.cuda.is_bf16_supported()}")
import triton; print(f"  triton {triton.__version__}")
import bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
loss = (p ** 2).sum(); loss.backward(); opt.step()
print(f"  AdamW8bit ok  bnb={bnb.__version__}")
PY

echo "=== vast_setup_llama: verify BCJR + trainer imports ==="
cd "$(dirname "$0")/.."
python3 -c "
import os; os.environ['BCJR_TRITON']='1'; os.environ['BCJR_MONOLITHIC']='1'
from src.bcjr.anneal import convert_layer_to_bcjr, convert_layer_to_ste, set_layer_temperature
from src.qat.qat_dense_decoder_layer import QATDenseDecoderLayer
from src.qat.train_e2e_kl import kl_loss_full_vocab, exp_temperature_schedule
print('  trainer imports ok')
" || echo "  WARN: trainer import failed — DO NOT run training"

echo "=== vast_setup_llama: layout check ==="
ls cache/calibration/tokens_llama.npy && echo "  calib ok"
ls cache/llama_bcjr_single/layer_08_wq_ptqinit.pt 2>/dev/null && echo "  prior 3-step snap present (reference)" || echo "  no prior snap (ok — fresh 15-step run starts from PTQ-init, not resume)"

echo "=== vast_setup_llama: done ==="
