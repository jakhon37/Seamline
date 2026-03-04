#!/bin/bash
# Optimized training script for 4 GPUs - Uses NCCL with optimal settings

set -e

echo "=============================================="
echo "Semantic Segmentation Training (4 GPU DDP)"
echo "=============================================="
echo "Using GPUs: ${GPUS:-0,1,2,3}"

# NCCL settings for GPUs without P2P or NVLink
export NCCL_P2P_DISABLE=1  # No GPU peer-to-peer
export NCCL_IB_DISABLE=1   # No InfiniBand
export NCCL_PROTO=LL       # Required! Low-latency protocol prevents hangs
export OMP_NUM_THREADS=4   # OpenMP threads per process

echo "NCCL Configuration:"
echo "  - P2P Disabled (not supported)"
echo "  - Protocol: LL (Low-Latency)"
echo "  - Optimized for PCIe-connected GPU topology"
echo ""

# Set visible devices
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"

# Run training
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train.py \
    --config "${CONFIG:-configs/default.yaml}"

echo ""
echo "Training completed!"
