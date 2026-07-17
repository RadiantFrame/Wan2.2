#!/bin/bash
# Install flash_attn from source on Blackwell GPUs (RTX PRO 6000, sm_120)
#
# Problem: pip install flash_attn --no-build-isolation fails with nvcc segfault
# because torch's BuildExtension falls back to the buggy distutils backend
# when ninja is not on PATH.
#
# Fix: ensure conda env bin (containing ninja) is on PATH so ninja backend is used.

set -e

CONDA_ENV="${CONDA_ENV:-wan}"

# Ensure conda is activated and env bin is on PATH
if command -v conda &>/dev/null; then
    CONDA_PREFIX="$(conda info --base)"
    source "${CONDA_PREFIX}/bin/activate" "${CONDA_ENV}"
else
    # fallback: try common anaconda path
    source /mnt/SS4T/anaconda3/bin/activate "${CONDA_ENV}"
fi

# Explicitly prepend conda env bin to PATH.
# conda activate should do this, but in practice torch's BuildExtension
# may still fail to find ninja via shutil.which() without this explicit export,
# causing a fallback to the buggy distutils backend (nvcc segfault).
export PATH="$(python -c 'import sys; print(sys.prefix)')/bin:${PATH}"

# Verify ninja is available (critical for stable compilation)
if ! command -v ninja &>/dev/null; then
    echo "ninja not found, installing..."
    pip install ninja packaging
fi

export MAX_JOBS="${MAX_JOBS:-4}"              # limit parallel nvcc jobs
export FLASH_ATTENTION_FORCE_BUILD=TRUE       # force source build (no prebuilt wheel for torch 2.9)

echo "=== Installing flash_attn ==="
echo "Python: $(which python)"
echo "ninja:  $(which ninja)"
echo "torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "MAX_JOBS: ${MAX_JOBS}"
echo ""

pip install flash_attn --no-build-isolation

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
