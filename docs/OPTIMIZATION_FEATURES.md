# Optimization Features Documentation

## Overview
This document describes the optimization features in mySegLib and how to configure them.

## Configuration Options

### Data Loading Optimizations

#### `data.use_onehot_encoding` (default: `false`)
**Impact:** Faster, lower memory use

When `false`, the dataset returns class indices `[B, H, W]` instead of one-hot encoded masks `[B, C, H, W]`. This is much more efficient for 2-class segmentation.

```yaml
data:
  use_onehot_encoding: false  # Optimized mode (recommended)
```

Set to `true` for backward compatibility with legacy code that expects one-hot encoding.

#### `data.cache_validation` (default: `true`)
**Impact:** Faster validation, no repeated file I/O

Caches the validation dataset in memory after first load. Since validation transforms are deterministic, this avoids repeated file I/O.

```yaml
data:
  cache_validation: true
```

#### `training.num_workers` (default: `8`)
**Impact:** Faster data loading

Number of parallel data loading workers. Tune it to your CPU core count.

```yaml
training:
  num_workers: 8  # Adjust based on your CPU cores
```

#### `training.prefetch_factor` (default: `2`)
**Impact:** Reduces data loading bottlenecks

Number of batches to prefetch per worker. Keeps the GPU fed with data.

```yaml
training:
  prefetch_factor: 2
```

#### `training.persistent_workers` (default: `true`)
**Impact:** Eliminates worker startup overhead between epochs

Keeps data loading workers alive between epochs instead of recreating them.

```yaml
training:
  persistent_workers: true
```

---

### Model Optimizations

#### `model.gradient_checkpointing` (default: `false`)
**Impact:** Saves GPU memory at the cost of slower training

Enables gradient checkpointing for large encoder backbones (ResNet50+, EfficientNet-B3+). Trades compute for memory by recomputing activations during backward pass.

```yaml
model:
  gradient_checkpointing: true  # Enable for large models or limited memory
```

**When to enable:**
- Large models (>50M parameters)
- Want larger batch sizes
- Limited GPU memory
- Training on 12GB or smaller GPUs

---

### Training Optimizations

#### `training.metrics_on_cpu` (default: `true`)
**Impact:** Frees some GPU memory

Computes metrics (F1, IoU, Precision, Recall) on CPU instead of GPU. Metrics computation is not a bottleneck, so moving it to CPU saves GPU memory for larger batch sizes.

```yaml
training:
  metrics_on_cpu: true
```

#### `training.track_memory` (default: `true`)
**Impact:** Diagnostic only, negligible performance cost

Tracks GPU memory usage and logs it after each epoch. Useful for identifying memory bottlenecks.

```yaml
training:
  track_memory: true
```

Output example:
```
GPU Memory - Allocated: <value>MB, Reserved: <value>MB, Peak: <value>MB
```

#### `training.enable_profiling` (default: `false`)
**Impact:** Only use for debugging, adds overhead

Enables PyTorch profiler to identify performance bottlenecks. Profiles first few epochs and saves Chrome trace.

```yaml
training:
  enable_profiling: true  # Only for debugging
```

Outputs to `<experiment_dir>/profiler/profiler_trace.json`. View in Chrome at `chrome://tracing`.

---

## Environment Variables

### `DATA_DIR`
Set the data directory via environment variable for portability:

```bash
export DATA_DIR=/path/to/your/data
python scripts/train.py --config configs/default.yaml
```

Config will use `$DATA_DIR` if set, otherwise falls back to the default path.

---

## Backward Compatibility

All optimizations are optional and configurable. To use legacy behavior:

```yaml
data:
  use_onehot_encoding: true  # Old one-hot encoding
  cache_validation: false
training:
  num_workers: 2  # Old value
  prefetch_factor: null  # Disable prefetch
  persistent_workers: false
  metrics_on_cpu: false  # Compute on GPU
  track_memory: false
model:
  gradient_checkpointing: false
```

---

## Starting Points by GPU Memory

These are starting points. Tune them for your model and image size.

### 24GB VRAM
```yaml
training:
  batch_size: 128
  num_workers: 8
  gradient_accumulation_steps: 2
  metrics_on_cpu: true
model:
  gradient_checkpointing: false
```

### 16GB VRAM
```yaml
training:
  batch_size: 64
  num_workers: 8
  gradient_accumulation_steps: 4
  metrics_on_cpu: true
model:
  gradient_checkpointing: true
```

### 12GB VRAM
```yaml
training:
  batch_size: 32
  num_workers: 6
  gradient_accumulation_steps: 8
  metrics_on_cpu: true
model:
  gradient_checkpointing: true
  encoder: "resnet18"
```

---

## Profiling and Debugging

### Enable Profiling
```yaml
training:
  enable_profiling: true
```

Run for 1-2 epochs, then check:
1. `<experiment_dir>/profiler/profiler_trace.json` - Load in Chrome at `chrome://tracing`
2. Console output shows top operations by CUDA time

### Common Bottlenecks
- **Data loading slow:** Increase `num_workers`, enable `prefetch_factor`
- **GPU memory full:** Enable `gradient_checkpointing`, reduce `batch_size`, enable `metrics_on_cpu`
- **Synchronization overhead:** Check distributed barriers (already optimized)

---

## Summary of All New Config Options

| Option | Default | Impact | When to Change |
|--------|---------|--------|----------------|
| `data.use_onehot_encoding` | `false` | Faster | Set `true` for legacy code |
| `data.cache_validation` | `true` | Faster validation | Set `false` if low RAM |
| `training.num_workers` | `8` | Faster loading | Adjust for CPU cores |
| `training.prefetch_factor` | `2` | Reduces bottlenecks | Increase for slow storage |
| `training.persistent_workers` | `true` | Faster epochs | Always keep enabled |
| `training.metrics_on_cpu` | `true` | Less GPU memory | Set `false` if CPU is bottleneck |
| `training.track_memory` | `true` | Diagnostic | Always keep enabled |
| `training.enable_profiling` | `false` | Debugging only | Enable to find bottlenecks |
| `model.gradient_checkpointing` | `false` | Less memory | Enable for large models |

---

## Troubleshooting

### Out of Memory Errors
1. Reduce `batch_size`
2. Enable `model.gradient_checkpointing`
3. Ensure `training.metrics_on_cpu: true`
4. Reduce `num_workers` if system RAM is low

### Data Loading Bottlenecks
1. Increase `num_workers` (try 12-16)
2. Increase `prefetch_factor` (try 4)
3. Check if storage is slow (use SSD not HDD)
4. Enable `data.cache_validation`

### Slower than Expected
1. Ensure `data.use_onehot_encoding: false`
2. Check `num_workers` is not 2 (should be 8+)
3. Verify `mixed_precision: true`
4. Disable `enable_profiling` if accidentally left on
