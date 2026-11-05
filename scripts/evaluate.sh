#!/bin/bash
# =============================================================================
# mySegLib Evaluation Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/default.yaml"

# Parse arguments
CHECKPOINT=""
OUTPUT=""
SAVE_PREDS=false
CONFUSION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --save-predictions|-s)
            SAVE_PREDS=true
            shift
            ;;
        --confusion-matrix|-m)
            CONFUSION=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --checkpoint PATH [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint, -c PATH   Model checkpoint (required)"
            echo "  --output, -o PATH       Output JSON file for results"
            echo "  --save-predictions, -s  Save prediction masks"
            echo "  --confusion-matrix, -m  Print confusion matrix"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${CHECKPOINT}" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Activate a conda environment if CONDA_ENV is set
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

cd "${PROJECT_DIR}"

# Build command
CMD="python scripts/evaluate.py --config ${CONFIG_FILE} --checkpoint ${CHECKPOINT}"
[ -n "${OUTPUT}" ] && CMD="${CMD} --output ${OUTPUT}"
[ "${SAVE_PREDS}" = true ] && CMD="${CMD} --save-predictions"
[ "${CONFUSION}" = true ] && CMD="${CMD} --confusion-matrix"

echo "Running: ${CMD}"
exec ${CMD}
