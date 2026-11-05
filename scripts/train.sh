#!/bin/bash
# =============================================================================
# mySegLib Training Script
# =============================================================================
# Usage:
#   ./train.sh                    # Single GPU training
#   ./train.sh --gpus 4           # 4 GPU training with DDP
#   ./train.sh --mode dp          # Force DataParallel mode (legacy fallback)
#   ./train.sh --mode ddp         # Force DDP mode (default)
#   ./train.sh --resume checkpoint.pth  # Resume training
#
# Multi-GPU runs use:
#       - Mixed precision (FP16)
#       - Gradient accumulation
#       - NCCL P2P disabled, for GPUs without peer-to-peer support
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/default.yaml"

# Default values
NUM_GPUS=1
RESUME_CHECKPOINT=""
MASTER_PORT=12355
TRAIN_MODE=""  # auto, dp, ddp

# NCCL settings for GPUs without P2P or NVLink
export NCCL_P2P_DISABLE=1  # No GPU peer-to-peer
export NCCL_IB_DISABLE=1   # No InfiniBand
export NCCL_PROTO=LL       # Low-latency protocol (required for 3+ GPUs)
export OMP_NUM_THREADS=4  # or another value based on your cores/processes
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume|-r)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --port|-p)
            MASTER_PORT="$2"
            shift 2
            ;;
        --mode|-m)
            TRAIN_MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpus, -g NUM      Number of GPUs (default: 1)"
            echo "  --config, -c PATH   Config file path (default: configs/default.yaml)"
            echo "  --resume, -r PATH   Checkpoint to resume from"
            echo "  --port, -p PORT     Master port for DDP (default: 12355)"
            echo "  --mode, -m MODE     Training mode: dp (DataParallel) or ddp (DDP)"
            echo "                      Default: ddp (recommended for all GPU counts)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Single GPU"
            echo "  $0 --gpus 2                  # 2 GPUs with DDP"
            echo "  $0 --config configs/large_model.yaml  # Use large model config"
            echo ""
            echo "Note: For quick start, use bash scripts/train_small.sh or train_large.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "mySegLib Training"
echo "=============================================="
# Activate a conda environment if CONDA_ENV is set
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# Show GPU info
echo ""
echo "GPU Configuration:"
python -c "import torch; print(f'  Available GPUs: {torch.cuda.device_count()}'); [print(f'    GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "  Using GPUs: ${NUM_GPUS}"
echo "  Config: ${CONFIG_FILE}"

# Determine training mode - DDP by default (optimized for PCIe)
if [ -z "${TRAIN_MODE}" ]; then
    TRAIN_MODE="ddp"
fi
echo "  Mode: ${TRAIN_MODE}"
echo ""
echo "Optimizations active:"
echo "  ✓ Mixed Precision (FP16)"
echo "  ✓ Gradient Accumulation (configured in YAML)"
echo "  ✓ NCCL P2P Disabled + PROTO=LL"
echo ""

cd "${PROJECT_DIR}"

if [ "${TRAIN_MODE}" == "dp" ]; then
    # DataParallel mode - single process, multi-GPU (legacy fallback)
    echo "⚠️  Using DataParallel (legacy mode)"
    echo "   For better performance, use --mode ddp"
    echo ""
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
    CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} python scripts/train_dp.py --config ${CONFIG_FILE}"
elif [ "${NUM_GPUS}" -eq 1 ]; then
    # Single GPU training
    echo "Single GPU training"
    echo ""
    CMD="python scripts/train.py --config ${CONFIG_FILE}"
else
    # DDP mode - multi-process with NCCL (optimized)
    echo "DDP training with ${NUM_GPUS} GPUs"
    echo ""
    CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} scripts/train.py --config ${CONFIG_FILE}"
fi

# Add resume if specified
if [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} --resume ${RESUME_CHECKPOINT}"
fi

# Run training
echo "Running command:"
echo "  ${CMD}"
echo ""
echo "=============================================="
echo "Starting training..."
echo "=============================================="
echo ""

eval ${CMD}
