#!/bin/bash
# Quick start script for small models (Unet ResNet18/34)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CONFIG=configs/small_model.yaml

# For GPUs without P2P or NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_PROTO=LL

echo "Starting training with SMALL model configuration"
echo "- Model: Unet + ResNet18"
echo "- Batch size: 128 per GPU"
echo "- Gradient accumulation: 2 steps"
echo "- Mixed precision: Enabled"
echo "- NCCL Protocol: LL"
echo ""

bash scripts/train_4gpu.sh
