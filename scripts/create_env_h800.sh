#!/bin/bash

set -e

######################################## nvidia-fabricmanager ########################################

echo "=== Installing NVIDIA Fabric Manager ==="

# Install and start nvidia-fabricmanager for NVSwitch-based multi-GPU systems.
#
# H800 (and other NVLink/NVSwitch-based GPUs) require the fabric manager
# service to initialize the NVLink fabric. Without it, cuInit() fails with
# Error 802 (CUDA_ERROR_NOT_INITIALIZED) even though nvidia-smi works.

# --- Detect driver version to pick matching fabricmanager ---
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
FM_VER=$(echo "${DRIVER_VER}" | cut -d'.' -f1)

echo "=== NVIDIA Fabric Manager Installation ==="
echo "Driver version: ${DRIVER_VER}"
echo "Fabric Manager version: ${FM_VER}"
echo ""

# --- Ensure NVIDIA CUDA repository is available ---
if ! apt-cache show nvidia-fabricmanager-${FM_VER} &>/dev/null; then
    echo "nvidia-fabricmanager-${FM_VER} not found in repos, adding NVIDIA CUDA repo..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb
    apt-get update -qq
fi

# --- Install fabric manager ---
echo "Installing nvidia-fabricmanager-${FM_VER}..."
apt-get install -y nvidia-fabricmanager-${FM_VER}

# --- Start and enable service ---
echo "Starting nvidia-fabricmanager service..."
systemctl start nvidia-fabricmanager
systemctl enable nvidia-fabricmanager

# --- Verify ---
echo ""
echo "=== Verifying ==="
systemctl is-active --quiet nvidia-fabricmanager && echo "Service: active (running)" || echo "Service: NOT running"

echo "---"
# Test cuInit
python3 -c "
import ctypes
cuda = ctypes.CDLL('libcuda.so.1')
cuda.cuInit.restype = int
result = cuda.cuInit(0)
print(f'cuInit(0): {\"OK (0)\" if result == 0 else f\"FAILED ({result})\"}')
" 2>/dev/null || echo "(python3 not available, skip cuInit test)"

echo ""
echo "=== Done ==="


######################################## Install Flash-Attn ########################################

echo "=== Installing Flash-Attn ==="

# Install flash_attn from source on Blackwell GPUs (RTX PRO 6000, sm_120)
#
# Problem: pip install flash_attn --no-build-isolation fails with nvcc segfault
# because torch's BuildExtension falls back to the buggy distutils backend
# when ninja is not on PATH.

CONDA_ENV="${CONDA_ENV:-wan}"

# Ensure conda is activated and env bin is on PATH
CONDA_PREFIX="$(conda info --base 2>/dev/null || echo /data/anaconda3)"
source "${CONDA_PREFIX}/bin/activate" "${CONDA_ENV}"

# Explicitly prepend conda env bin to PATH.
# conda activate should do this, but in practice torch's BuildExtension
# may still fail to find ninja via shutil.which() without this explicit export,
# causing a fallback to the buggy distutils backend (nvcc segfault).
export PATH="$(python -c 'import sys; print(sys.prefix)')/bin:${PATH}"

# Verify ninja is available (critical for stable compilation)
if ! command -v ninja &>/dev/null; then
    echo "ninja not found, installing..."
    pip install ninja packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

export MAX_JOBS="${MAX_JOBS:-32}"              # limit parallel nvcc jobs (192 cores available)
export TORCH_CUDA_ARCH_LIST="9.0"             # H800 (Hopper, sm_90)
export FLASH_ATTENTION_FORCE_BUILD=TRUE       # force source build (no prebuilt wheel for torch 2.9)

echo "=== Installing flash_attn ==="
echo "Python: $(which python)"
echo "ninja:  $(which ninja)"
echo "torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "MAX_JOBS: ${MAX_JOBS}"
echo ""

pip install flash_attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Verifying ==="
python -c "
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 8, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 64, 8, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 64, 8, 64, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, k, v)
print(f'flash_attn {flash_attn.__version__} OK, output shape: {out.shape}')
"