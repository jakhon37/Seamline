# Multi-GPU Training Guide

This guide covers DDP training on machines where GPUs communicate over PCIe, with no NVLink and no GPU peer-to-peer support.

## What the Scripts Set Up

### Mixed Precision (FP16)
- Halves the size of the gradients that GPUs have to synchronize
- Enabled in all configs

### Gradient Accumulation
- Synchronizes less often, which matters when inter-GPU communication is slow
- Small models: 2 steps
- Large models: 4 steps

### NCCL With P2P Disabled
- Falls back to Gloo automatically if NCCL fails
- Environment variables are set in the training scripts:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_PROTO=LL
```

If DDP with 3 or more GPUs fails with `CUDA error: an illegal memory access was encountered`, check that `NCCL_PROTO=LL` is set.

### Data Loading
- `num_workers` per GPU is configurable
- Pin memory is enabled
- Transfers are non-blocking

## Quick Start

### Small Models (ResNet18/34)
```bash
bash scripts/train_small.sh
```

### Large Models (ResNet50+)
```bash
bash scripts/train_large.sh
```

### Custom Configuration
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONFIG=configs/your_config.yaml
bash scripts/train_4gpu.sh
```

### Single GPU
```bash
python scripts/train.py --config configs/default.yaml
```

## Configuration Files

| File | Intended for | Batch size per GPU | Gradient accumulation |
|---|---|---|---|
| `configs/small_model.yaml` | Models under 50M parameters | 128 | 2 steps |
| `configs/large_model.yaml` | Models of 50M to 200M parameters | 64 | 4 steps |
| `configs/default.yaml` | Most models | 128 | 2 steps |

## Monitoring

```bash
tensorboard --logdir checkpoints/
watch -n 1 nvidia-smi
nvidia-smi dmon -s u
```

## Troubleshooting

### NCCL Errors
The script falls back to the Gloo backend, which is slower but more stable. To force it, set this in your config:

```yaml
distributed:
  backend: "gloo"
```

### Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps` to keep the effective batch size
- Try a smaller encoder

### Slow Training
- Check that all GPUs are busy in `nvidia-smi`
- Check that `mixed_precision` is enabled
- For large models, use at least 4 gradient accumulation steps

### DataParallel Fallback
```bash
bash scripts/train_dp.sh
```
