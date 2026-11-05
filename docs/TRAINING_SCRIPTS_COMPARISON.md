# Training Script Comparison

## Available Training Scripts

### 1. **train_small.sh** / **train_large.sh** ⭐ RECOMMENDED
```bash
bash scripts/train_small.sh   # For models <50M params
bash scripts/train_large.sh   # For models 50-200M params
```

**Pros:**
- ✅ Pre-configured for optimal performance
- ✅ Automatic config selection
- ✅ No parameters needed
- ✅ Fastest to use

**Use when:** You want the easiest, most optimized experience

---

### 2. **train_4gpu.sh** - General Purpose Launcher
```bash
export CONFIG=configs/your_config.yaml
bash scripts/train_4gpu.sh
```

**Pros:**
- ✅ Optimized NCCL settings pre-configured
- ✅ Works with any config via environment variable
- ✅ Clear output about optimizations

**Cons:**
- Need to set CONFIG environment variable

**Use when:** You have a custom config and want optimized settings

---

### 3. **train.sh** - Advanced/Legacy Script
```bash
bash scripts/train.sh --gpus 4 --config configs/default.yaml
```

**Pros:**
- ✅ Most flexible (many CLI options)
- ✅ Can specify GPU count
- ✅ Can force different modes (dp/ddp)

**Cons:**
- More complex
- Need to understand options

**Use when:** You need fine control over training parameters

---

### 4. **train_dp.sh** - Emergency Fallback
```bash
bash scripts/train_dp.sh
```

**Pros:**
- ✅ Simple DataParallel (no distributed setup)
- ✅ Most stable (no NCCL issues)

**Cons:**
- ❌ Slower than DDP
- ❌ Less efficient gradient sync

**Use when:** DDP fails or for debugging

---

## Decision Tree

```
Are you training for the first time?
└─ YES → Use train_small.sh or train_large.sh
└─ NO
   └─ Do you have a custom config?
      └─ YES → Use train_4gpu.sh (set CONFIG env)
      └─ NO
         └─ Need to change GPU count or advanced options?
            └─ YES → Use train.sh --gpus N --config PATH
            └─ NO → Use train_small.sh or train_large.sh
```

---

## Quick Reference

| Script | Complexity | Performance | Best For |
|--------|-----------|-------------|----------|
| **train_small.sh** | ⭐ Easy | ⭐⭐⭐⭐⭐ | **Most users** |
| **train_large.sh** | ⭐ Easy | ⭐⭐⭐⭐⭐ | **Large models** |
| train_4gpu.sh | ⭐⭐ Medium | ⭐⭐⭐⭐⭐ | Custom configs |
| train.sh | ⭐⭐⭐ Advanced | ⭐⭐⭐⭐⭐ | Power users |
| train_dp.sh | ⭐ Easy | ⭐⭐⭐ | Fallback only |

---

## Examples

### Example 1: Train current Unet (easiest)
```bash
bash scripts/train_small.sh
```

### Example 2: Train with large backbone
```bash
bash scripts/train_large.sh
```

### Example 3: Custom config with train_4gpu.sh
```bash
export CONFIG=configs/my_custom.yaml
bash scripts/train_4gpu.sh
```

### Example 4: Advanced usage with train.sh
```bash
# 2 GPUs only
bash scripts/train.sh --gpus 2

# Custom config and resume
bash scripts/train.sh \
  --gpus 4 \
  --config configs/large_model.yaml \
  --resume checkpoints/exp_123/checkpoint_epoch_10.pth

# Force DataParallel
bash scripts/train.sh --mode dp
```

---

## Recommendation

**For most users:** Use `train_small.sh` or `train_large.sh`

These scripts:
- Are pre-configured
- Work out of the box
- Provide the best performance
- Require zero configuration

Only use `train.sh` if you need fine control over parameters.
