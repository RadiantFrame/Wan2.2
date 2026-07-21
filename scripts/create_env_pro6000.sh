#!/bin/bash

set -e

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

export MAX_JOBS="${MAX_JOBS:-6}"              # limit parallel nvcc jobs (192 cores available)
export TORCH_CUDA_ARCH_LIST="12.0"             # NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 12.0
export FLASH_ATTENTION_FORCE_BUILD=TRUE       # force source build

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