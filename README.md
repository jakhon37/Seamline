# Semantic Segmentation (Multi-GPU Optimized)
Semantic segmentation built on PyTorch and segmentation_models_pytorch, tuned for PCIe-bound multi-GPU setups. Includes AMP, gradient accumulation, backend fallbacks, and production-friendly logging/checkpointing.

## Features
- Modular model factory (Unet/Unet++/DeepLabV3+/FPN/PSPNet/PAN/etc.) with swappable encoders
- Optimized multi-GPU training: AMP (FP16), gradient accumulation, NCCL→Gloo fallback, DataParallel fallback script
- YAML-driven configs for data, augmentation, model, loss, scheduler, logging, paths
- Robust metrics (F1, Precision, Recall, IoU), checkpoint manager (top-k), TensorBoard/W&B logging
- Inference with single image, directory batch, or chunked tiling for large images; ONNX export supported

## Project Layout
```
configs/            # YAML configs (default, small_model, large_model, etc.)
docs/               # Guides and requirements (docs/README.md, MULTI_GPU_TRAINING.md)
src/                # Data, models, losses, utils
scripts/            # Train/eval/inference/export entrypoints + helper shells
checkpoints/, training_outputs/, wandb/  # Outputs & runs
```

## Requirements & Setup
```bash
python -m venv venv && source venv/bin/activate
# Install PyTorch matching your CUDA (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Install project deps
pip install -r docs/requirements.txt
```

## Dataset Structure (example)
```
data/
├── dataset/
│   ├── train/{images/, masks/}
│   ├── val/{images/, masks/}
│   └── test/{images/, masks/}
```
Set `data.base_dir`, `dataset_dir`, `train/val/test` dirs, and `image_size/num_classes` in your config.

## Quick Start
Single GPU (baseline):
```bash
python scripts/train.py --config configs/default.yaml
```
Multi-GPU (DDP, optimized):
```bash
# Small models (<50M params)
bash scripts/train_small.sh
# Large models (50–200M params)
bash scripts/train_large.sh
# Custom GPUs/config
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONFIG=configs/your_config.yaml
bash scripts/train_4gpu.sh
```
DataParallel fallback (if NCCL fails):
```bash
bash scripts/train_dp.sh --config configs/default.yaml --gpus 0,1
```

## Configuration Highlights (`configs/*.yaml`)
- `data.*`: dataset paths, image size, num_classes, val split
- `model.*`: architecture, encoder, encoder_weights, activation
- `training.*`: epochs, batch_size (per GPU), learning_rate, weight_decay, num_workers, mixed_precision, gradient_accumulation_steps
- `loss.*`: focal/dice/combined/cross_entropy/tversky with class_weights
- `scheduler.*`: reduce_on_plateau or cosine; `early_stopping.*`
- `paths.*`: output/checkpoint/log dirs (defaults vary by config)

## Evaluation
```bash
python scripts/evaluate.py --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --confusion-matrix --save-predictions
```
Uses test set if present, else falls back to val split.

## Inference
Single image:
```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
  --input image.tif --output mask.tif
```
Directory (auto-saves *_mask.tif):
```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
  --input ./images/ --output ./predictions/
```
Large images (chunked): add `--chunk --chunk-size 224 --overlap 64`.

## ONNX Export
```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/default.yaml \
  --output models/segmentation.onnx
```

## Logging & Checkpoints
- Checkpoints + best model saved under `paths.checkpoint_dir` (e.g., `./checkpoints` or `./training_outputs/checkpoints`).
- TensorBoard and optional W&B logging via `logging.tensorboard` / `logging.wandb`.
- Metric plots/history written by `MetricLogger` in the experiment directory.

## Multi-GPU Notes
- Scripts export NCCL safety flags (`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_PROTO=LL`) for PCIe-connected GPU setups; torchrun auto-falls back to Gloo if NCCL fails.
- Gradient accumulation (`training.gradient_accumulation_steps`) keeps effective batch large while reducing communication.

## Troubleshooting & Docs
- See `docs/README.md` and `docs/MULTI_GPU_TRAINING.md` for detailed guidance and performance tips.
- Emergency fallback: `bash scripts/train_dp.sh`.

## License
MIT
