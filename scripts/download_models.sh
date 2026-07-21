#!/bin/bash

set -e

######################################## DOWNLOAD MODELS USING MODELSCOPE ########################################

echo "=== Download Models Using ModelScope!!! ==="

# models to download from ModelScope
models=(
    "Wan-AI/Wan2.2-T2V-A14B"
    "Wan-AI/Wan2.2-T2V-A14B-BF16"
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    "Wan-AI/Wan2.2-I2V-A14B"
    "Wan-AI/Wan2.2-I2V-A14B-BF16"
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    "Wan-AI/Wan2.2-TI2V-5B-BF16"
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LOCAL_DIR="/data/models/modelscope"
# Loop over each model
for model in "${models[@]}"; do
    echo "===== $model ====="

    modelscope download --model $model --local-dir $LOCAL_DIR/$model
done

######################################## DOWNLOAD MODELS USING HUGGINGFACE ########################################

echo "=== Download Models Using Huggingface!!! ==="

# models to download from huggingface
repo_ids=(
    "Wan-AI/Wan2.2-T2V-A14B"
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    "Wan-AI/Wan2.2-I2V-A14B"
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    "Wan-AI/Wan2.2-TI2V-5B"
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LOCAL_DIR="/data/models"
# Loop over each model
for repo_id in "${repo_ids[@]}"; do
    echo "===== $repo_id ====="

    hf download $repo_id --local-dir $LOCAL_DIR/$repo_id
done
