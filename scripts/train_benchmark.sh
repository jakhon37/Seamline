#!/bin/bash
# =============================================================================
# Multi-GPU DDP Training Benchmark with NCCL + P2P
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/default.yaml"
BENCHMARK_MODE="all"
NUM_GPUS=""
NUM_EPOCHS=3
MASTER_PORT=12355

# NCCL settings for local peer-to-peer
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=LOC  # Forces local P2P only
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode|-m) BENCHMARK_MODE="$2"; shift 2 ;;
        --gpus|-g) NUM_GPUS="$2"; BENCHMARK_MODE="specific"; shift 2 ;;
        --epochs|-e) NUM_EPOCHS="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 [--mode all|single|multi] [--gpus N] [--epochs N]"; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Activate a conda environment if CONDA_ENV is set
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

RESULTS_DIR="${PROJECT_DIR}/benchmark_results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/ddp_benchmark_${TIMESTAMP}.txt"

echo "=============================================="
echo "DDP Training Benchmark (Gloo Backend)"
echo "=============================================="
echo "Backend: Gloo (CPU-based, no NCCL needed)"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
echo "Results: ${RESULTS_FILE}"
echo ""

cd "${PROJECT_DIR}"

run_benchmark() {
    local gpu_count=$1
    local run_name=$2
    
    echo "=========================================="
    echo "${run_name}"
    echo "=========================================="
    
    local temp_config="${PROJECT_DIR}/configs/benchmark_temp.yaml"
    python -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
config['training']['epochs'] = ${NUM_EPOCHS}
config['early_stopping']['patience'] = ${NUM_EPOCHS} + 5
with open('${temp_config}', 'w') as f:
    yaml.dump(config, f)
"
    
    local start_time=$(date +%s)
    
    if [ "${gpu_count}" -eq 1 ]; then
        python scripts/train.py --config "${temp_config}" 2>&1 | tee -a "${RESULTS_FILE}"
    else
        torchrun --nproc_per_node="${gpu_count}" --master_port="${MASTER_PORT}" \
            scripts/train.py --config "${temp_config}" 2>&1 | tee -a "${RESULTS_FILE}"
    fi
    
    local elapsed=$(($(date +%s) - start_time))
    rm -f "${temp_config}"
    
    echo "" | tee -a "${RESULTS_FILE}"
    echo "=== ${run_name} DONE: $((elapsed/60))m $((elapsed%60))s ===" | tee -a "${RESULTS_FILE}"
    echo "" | tee -a "${RESULTS_FILE}"
    sleep 3
}

echo "=========================================" > "${RESULTS_FILE}"
echo "DDP Benchmark (NCCL+P2P) - $(date)"       >> "${RESULTS_FILE}"
echo "=========================================" >> "${RESULTS_FILE}"
echo ""                                          >> "${RESULTS_FILE}"

case "${BENCHMARK_MODE}" in
    all)
        run_benchmark 1 "1 GPU Baseline"
        run_benchmark 2 "2 GPU DDP"
        run_benchmark 4 "4 GPU DDP" ;;
    single) run_benchmark 1 "1 GPU Baseline" ;;
    multi)
        run_benchmark 2 "2 GPU DDP"
        run_benchmark 4 "4 GPU DDP" ;;
    specific)
        [ -z "${NUM_GPUS}" ] && echo "Error: --gpus required" && exit 1
        run_benchmark "${NUM_GPUS}" "${NUM_GPUS} GPU DDP" ;;
esac

echo "=========================================="
echo "Complete! Results: ${RESULTS_FILE}"
grep "DONE:" "${RESULTS_FILE}"
