#!/bin/bash
# Training helper script - shows available options

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║             mySegLib - Multi-GPU Training Helper             ║
║              Optimized for PCIe Topology                     ║
╚══════════════════════════════════════════════════════════════╝

Available Training Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick Start (Recommended):
  bash scripts/train_small.sh        - Small models (ResNet18/34)
  
  bash scripts/train_large.sh        - Large models (ResNet50+)

Advanced Usage:
  bash scripts/train_4gpu.sh         - Use with custom config
                                       Set CONFIG env variable
  
  bash scripts/train_dp.sh           - DataParallel fallback
                                       Use if DDP fails

Testing & Benchmarking:
  bash scripts/benchmark.sh          - Compare all methods
                                       Takes ~15-20 minutes

Single GPU (for comparison):
  python scripts/train.py --config configs/small_model.yaml

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Available Configurations:
  configs/small_model.yaml     - Models <50M params
  configs/large_model.yaml     - Models 50M-200M params  
  configs/default.yaml         - Balanced config

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Documentation:
  docs/MULTI_GPU_TRAINING.md   - Full training guide
  docs/OPTIMIZATION_FEATURES.md - Configuration options

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monitoring:
  watch -n 1 nvidia-smi              - GPU utilization
  tensorboard --logdir checkpoints/  - Training metrics
  nvidia-smi dmon -s u               - PCIe bandwidth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hardware Status:
EOF

nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader | \
  awk -F', ' '{printf "  GPU %s: %s (%s, %s utilization)\n", $1, $2, $3, $4}'

echo ""
cat << 'EOF'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Optimizations Active:
  ✓ Mixed Precision (FP16)
  ✓ Gradient Accumulation       - Reduces sync frequency
  ✓ NCCL P2P Disabled           - For GPUs without P2P
  ✓ PCIe Optimized DataLoader   - num_workers=2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Examples:

1. Train current Unet model:
   $ bash scripts/train_small.sh

2. Train with custom backbone:
   $ # Edit configs/small_model.yaml or large_model.yaml
   $ # Change encoder: "resnet18" to "resnet50" etc.
   $ bash scripts/train_large.sh

3. Custom configuration:
   $ export CONFIG=configs/my_config.yaml
   $ export CUDA_VISIBLE_DEVICES=0,1,2,3
   $ bash scripts/train_4gpu.sh

4. Resume training:
   $ python scripts/train.py --config configs/small_model.yaml \
       --resume checkpoints/experiment_name/checkpoint_epoch_X.pth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to start? Run: bash scripts/train_small.sh

EOF
