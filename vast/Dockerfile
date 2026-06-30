# Reproducible env for the BCJR amenability sweep on a Vast.ai H100 (sm_90).
# Bakes the finicky stack (torch 2.12/cu130 + from-source fast-hadamard-transform
# + Triton) so paid GPU hours aren't spent rebuilding. Builds WITHOUT a GPU
# (FHT compiles for sm_90 via arch list), so you can build anywhere & push.
#
#   docker build -t <you>/quant2bit:h100 -f vast/Dockerfile .
#   docker push <you>/quant2bit:h100
# then rent a Vast H100 with that image. (Or skip Docker and use bootstrap.sh
# on a stock PyTorch instance — see vast/README.md.)
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip git ca-certificates ninja-build && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && python3 -m pip install --upgrade pip

# Core stack — pin torch to match the local run (cu130 wheels support sm_90/H100).
RUN python3 -m pip install torch==2.12.1
RUN python3 -m pip install \
    "transformers>=5.0" accelerate safetensors bitsandbytes datasets \
    numpy scipy tqdm huggingface_hub "lm-eval[api]"

# Build fast-hadamard-transform from source for H100 (sm_90).
ENV TORCH_CUDA_ARCH_LIST=9.0
RUN git clone https://github.com/Dao-AILab/fast-hadamard-transform /tmp/fht && \
    cd /tmp/fht && python3 -m pip install --no-build-isolation . && rm -rf /tmp/fht

ENV BCJR_TRITON=1 BCJR_MONOLITHIC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /workspace
CMD ["/bin/bash"]
