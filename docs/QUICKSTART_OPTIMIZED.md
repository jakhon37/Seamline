# Quick Start: Optimized Training

This guide shows you how to get maximum performance immediately.

## TL;DR - Just Run This

```bash
# Set your data path
export DATA_DIR=/path/to/your/dataset

# Single GPU (optimized)
python scripts/train.py --config configs/default.yaml

# Multi-GPU (4 GPUs, optimized)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
```

**That's it!** The default config is already optimized for maximum performance.

---

## What's Optimized by Default

The `configs/default.yaml` includes all optimizations:

✅ **Data Loading**
- `num_workers: 8`
- `prefetch_factor: 2`
- `persistent_workers: true`

✅ **Memory & Computation**
- `use_onehot_encoding: false` (optimized class indices)
- `metrics_on_cpu: true` (frees GPU memory)
- `cache_validation: true` (no repeated I/O)

✅ **Monitoring** (diagnostic)
- `track_memory: true` (GPU memory stats)

---

## Hardware-Specific Settings

### 24GB GPUs
```yaml
# configs/default.yaml starts from these values
training:
  batch_size: 128
  num_workers: 8
model:
  gradient_checkpointing: false  # Don't need it with 24GB
```

### Smaller GPUs (12-16GB)
```bash
# Use the small model config
python scripts/train.py --config configs/small_model.yaml
```

Or edit your config:
```yaml
training:
  batch_size: 64  # or 32 for 12GB
  gradient_accumulation_steps: 4  # Keep effective batch size large
model:
  encoder: "resnet18"  # Smaller encoder
  gradient_checkpointing: true  # Enable for memory savings
```

---

## Troubleshooting

### "Out of Memory" Error
1. Reduce `batch_size`: `64` → `32`
2. Enable `gradient_checkpointing: true` in model section
3. Check `metrics_on_cpu: true` is set

### "Data loading is slow"
1. Increase `num_workers`: `8` → `12`
2. Check storage speed (use SSD not HDD)
3. Verify `prefetch_factor: 2` is set

### "Training slower than expected"
1. Verify `use_onehot_encoding: false` (not true!)
2. Check `num_workers: 8` (not 2!)
3. Ensure `mixed_precision: true`

---

## Monitoring Performance

After each epoch, you'll see:
```
Epoch N: Train Loss=<value>, Val Loss=<value>, LR=<value>, F1=<value>, IoU=<value>
  GPU Memory - Allocated: <value>MB, Reserved: <value>MB, Peak: <value>MB
```

This tells you:
- Loss and metrics
- Current learning rate
- GPU memory usage (helps identify OOM issues)

---

## Advanced: Enable Profiling

Only for debugging bottlenecks:

```yaml
training:
  enable_profiling: true  # Adds overhead
```

Then check:
- `<experiment_dir>/profiler/profiler_trace.json`
- Open in Chrome at `chrome://tracing`

---

## Need More Details?

See [OPTIMIZATION_FEATURES.md](OPTIMIZATION_FEATURES.md) for:
- Complete list of all optimization options
- Hardware-specific recommendations
- Backward compatibility
