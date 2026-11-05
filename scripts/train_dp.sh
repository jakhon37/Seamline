#!/bin/bash
# =============================================================================
# mySegLib Training Script (DataParallel - 4 GPU)
# =============================================================================
# Uses nn.DataParallel as a fallback when DDP is not stable
#
# Usage:
#   ./train_dp.sh                    # 4 GPU training (default)
#   ./train_dp.sh --gpus 0,1         # Specific GPUs
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/default.yaml"

# Default: use all 4 GPUs
GPU_IDS="0,1,2,3"

# NCCL settings for GPUs without P2P or NVLink
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus|-g)
            GPU_IDS="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpus, -g IDS      GPU IDs to use (e.g., '0,1,2,3')"
            echo "  --config, -c PATH   Config file path"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate conda
echo "=============================================="
echo "mySegLib Training (DataParallel)"
echo "=============================================="
# Activate a conda environment if CONDA_ENV is set
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

echo "Using GPUs: ${GPU_IDS}"
echo ""

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python scripts/train_dp.py --config "${CONFIG_FILE}"
