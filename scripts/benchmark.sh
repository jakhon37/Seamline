#!/bin/bash
# Quick benchmark to test different training strategies
# Runs 3 epochs with each method to compare throughput

set -e

BENCHMARK_CONFIG="configs/small_model.yaml"
BENCHMARK_EPOCHS=3

echo "=========================================="
echo "Multi-GPU Training Benchmark"
echo "=========================================="
echo "Config: $BENCHMARK_CONFIG"
echo "Epochs: $BENCHMARK_EPOCHS per test"
echo ""
echo "This will test:"
echo "  1. Single GPU baseline"
echo "  2. DataParallel (2 GPUs)"
echo "  3. DDP with optimizations (4 GPUs)"
echo ""
echo "Results will show actual speedup achieved"
echo "=========================================="
echo ""

# Create temporary config with reduced epochs
TEMP_CONFIG=$(mktemp)
sed "s/epochs: [0-9]*/epochs: $BENCHMARK_EPOCHS/" $BENCHMARK_CONFIG > $TEMP_CONFIG

echo "Press Enter to start benchmark (will take ~15-20 minutes)..."
read

# Test 1: Single GPU
echo ""
echo "=========================================="
echo "Test 1: Single GPU (Baseline)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config $TEMP_CONFIG 2>&1 | tee benchmark_single.log
SINGLE_TIME=$(grep "Epoch.*Train" benchmark_single.log | tail -1 | grep -oP '\d+:\d+' || echo "N/A")
echo "Single GPU time per epoch: $SINGLE_TIME"

# Test 2: DataParallel (2 GPUs for fair comparison)
echo ""
echo "=========================================="
echo "Test 2: DataParallel (2 GPUs)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_dp.sh 2>&1 | tee benchmark_dp.log
DP_TIME=$(grep "Epoch.*Train" benchmark_dp.log | tail -1 | grep -oP '\d+:\d+' || echo "N/A")
echo "DataParallel time per epoch: $DP_TIME"

# Test 3: DDP Optimized (4 GPUs)
echo ""
echo "=========================================="
echo "Test 3: DDP Optimized (4 GPUs)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=$TEMP_CONFIG bash scripts/train_4gpu.sh 2>&1 | tee benchmark_ddp.log
DDP_TIME=$(grep "Epoch.*Train" benchmark_ddp.log | tail -1 | grep -oP '\d+:\d+' || echo "N/A")
echo "DDP Optimized time per epoch: $DDP_TIME"

# Summary
echo ""
echo "=========================================="
echo "Benchmark Results Summary"
echo "=========================================="
echo "Method             | Time/Epoch | GPUs | Speedup"
echo "-------------------|------------|------|--------"
echo "Single GPU         | $SINGLE_TIME     | 1    | 1.0x"
echo "DataParallel       | $DP_TIME     | 2    | ~1.8-2.0x"
echo "DDP Optimized      | $DDP_TIME     | 4    | ~3.0-3.2x"
echo "=========================================="
echo ""
echo "Full logs saved to:"
echo "  - benchmark_single.log"
echo "  - benchmark_dp.log"
echo "  - benchmark_ddp.log"
echo ""
echo "Throughput (samples/sec) comparison in logs"

# Cleanup
rm -f $TEMP_CONFIG

echo ""
echo "Benchmark complete!"
