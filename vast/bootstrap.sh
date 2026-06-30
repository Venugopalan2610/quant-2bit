#!/usr/bin/env bash
# Bootstrap the BCJR sweep on a STOCK Vast.ai PyTorch instance (no custom image).
# Installs the delta deps, builds fast-hadamard-transform for the instance GPU,
# clones the repo, downloads the (gated) model, and regenerates calibration.
#
# PREREQ: export HF_TOKEN=hf_xxx   (Llama-3.2 is gated — needed to download)
#
#   export HF_TOKEN=hf_xxx
#   bash vast/bootstrap.sh
#   bash vast/bcjr_amenability_sweep.sh
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN to your HuggingFace token (Llama-3.2 is gated)}"
REPO_URL="${REPO_URL:-https://github.com/Venugopalan2610/quant-2bit}"
WORK="${WORK:-/workspace/quant-2bit}"

echo "=== [1/5] deps ==="
# Match the local torch so FHT + kernels behave identically. Comment the torch
# line out to keep the instance's stock torch (faster setup, slight risk).
python3 -m pip install -q torch==2.12.1 || echo "torch pin failed; using stock torch"
python3 -m pip install -q "transformers>=5.0" accelerate safetensors bitsandbytes \
    datasets numpy scipy tqdm huggingface_hub "lm-eval[api]"

echo "=== [2/5] fast-hadamard-transform (build for this GPU) ==="
if ! python3 -c "import fast_hadamard_transform" 2>/dev/null; then
  ARCH=$(python3 -c "import torch;cc=torch.cuda.get_device_capability();print(f'{cc[0]}.{cc[1]}')")
  echo "  building FHT for sm_${ARCH/./}"
  git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform /tmp/fht
  TORCH_CUDA_ARCH_LIST="$ARCH" python3 -m pip install --no-build-isolation /tmp/fht
fi

echo "=== [3/5] repo ==="
[ -d "$WORK" ] || git clone "$REPO_URL" "$WORK"
cd "$WORK"

echo "=== [4/5] model (gated — needs HF_TOKEN) ==="
mkdir -p cache/model cache/calibration
if [ ! -f cache/model/Llama-3.2-1B/config.json ]; then
  huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir cache/model/Llama-3.2-1B --token "$HF_TOKEN"
fi

echo "=== [5/5] calibration (512×2048 from C4, must match local) ==="
if [ ! -f cache/calibration/tokens_llama.npy ]; then
  python3 scripts/regen_calib_llama.py \
    --model cache/model/Llama-3.2-1B \
    --out cache/calibration/tokens_llama.npy --n-seqs 512 --seq-len 2048
fi

echo ""
echo "Bootstrap done. Sanity check:"
python3 -c "import torch,fast_hadamard_transform;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0))"
echo "Next: bash vast/bcjr_amenability_sweep.sh"
