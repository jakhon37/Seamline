#!/bin/bash
# =============================================================================
# Semantic Segmentation Inference Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="base"

# Parse arguments
CHECKPOINT=""
INPUT=""
OUTPUT=""
USE_CHUNK=false
CHUNK_SIZE=224
OVERLAP=64

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --input|-i)
            INPUT="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --chunk)
            USE_CHUNK=true
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --checkpoint PATH --input PATH --output PATH [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint, -c PATH   Model checkpoint (required)"
            echo "  --input, -i PATH        Input image or directory (required)"
            echo "  --output, -o PATH       Output mask or directory (required)"
            echo "  --chunk                 Use chunking for large images"
            echo "  --chunk-size SIZE       Chunk size (default: 224)"
            echo "  --overlap SIZE          Overlap between chunks (default: 64)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${CHECKPOINT}" ] || [ -z "${INPUT}" ] || [ -z "${OUTPUT}" ]; then
    echo "Error: --checkpoint, --input, and --output are required"
    exit 1
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

cd "${PROJECT_DIR}"

# Build command
CMD="python scripts/inference.py --checkpoint ${CHECKPOINT} --input ${INPUT} --output ${OUTPUT}"
[ "${USE_CHUNK}" = true ] && CMD="${CMD} --chunk --chunk-size ${CHUNK_SIZE} --overlap ${OVERLAP}"

echo "Running: ${CMD}"
exec ${CMD}
