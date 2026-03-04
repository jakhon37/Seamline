#!/bin/bash
# Quick start script for large models (ResNet50+, EfficientNet, etc.)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CONFIG=configs/large_model.yaml

# For GPUs without P2P or NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_PROTO=LL

echo "Starting training with LARGE model configuration"
echo "- Model: Unet + ResNet50/101 or similar"
echo "- Batch size: 64 per GPU"
echo "- Gradient accumulation: 4 steps"
echo "- Mixed precision: Enabled (halves gradient size)"
echo "- NCCL Protocol: LL"
echo "- Expected speedup: 2.5-2.8x (vs single GPU)"
echo ""
echo ""

bash scripts/train_4gpu.sh
