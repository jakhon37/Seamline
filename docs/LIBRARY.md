# mySegLib

The library under [Seamline](../README.md): semantic segmentation training, evaluation and inference on PyTorch and [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch). Everything is driven by YAML configs, and training scales from one GPU to several with DDP.

## Features

**Models and training**
- 9 architectures: Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, PSPNet, PAN, Linknet, MAnet
- Encoders from segmentation_models_pytorch: ResNet, EfficientNet, DenseNet, MobileNet and more
- 4 losses: focal, Dice, cross-entropy, and combined focal and Dice
- Multi-GPU training with DDP (NCCL or Gloo), and a DataParallel fallback
- Mixed precision, gradient accumulation, gradient checkpointing, early stopping

**Inference and export**
- Single-image and batch inference
- Chunked inference with overlap, for images too large for GPU memory
- ONNX export

**Tooling**
- 168 unit tests, of which 10 need a GPU and skip on CPU
- GitHub Actions, pre-commit hooks and a Makefile
- Docker with GPU pass-through
- TensorBoard and Weights & Biases logging

CI also runs the full pipeline on synthetic data: train, resume, evaluate, inference and ONNX export. It runs on CPU, so it does not exercise the GPU and multi-GPU code paths.

## Quick Start

### 1. Install

You need Python 3.10+ and PyTorch 2.3+.

```bash
git clone https://github.com/jakhon37/Seamline.git
cd mySegLib
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Or run the automated setup with `bash scripts/setup_dev.sh`.

### 2. Prepare your dataset

```
YOUR_DATA/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── masks/    # Masks hold class indices
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/         # Optional
        ├── images/
        └── masks/
```

Images and masks are paired by exact name, ignoring a `_mask` suffix. Images without a mask are skipped with a warning.

### 3. Configure

Edit `configs/default.yaml`, or set `DATA_DIR` instead of `base_dir`:

```yaml
data:
  base_dir: "/path/to/YOUR_DATA"
  dataset_dir: "dataset"
  image_size: 224
  num_classes: 2

model:
  architecture: "Unet"  # Unet, UnetPlusPlus, DeepLabV3Plus, etc.
  encoder: "resnet18"   # resnet50, efficientnet-b4, etc.
```

### 4. Train

```bash
# Single GPU
python scripts/train.py --config configs/default.yaml

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml

# Docker
docker-compose up -d training
```

### 5. Run inference

```bash
# Single image
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
  --input image.jpg --output prediction.png

# Directory of images
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
  --input ./images/ --output ./predictions/

# Large image, in overlapping chunks
python scripts/inference.py --checkpoint checkpoints/best_model.pth \
  --input large.tif --output mask.tif --chunk
```

## Project Structure

```
├── app/          Seamline, the desktop app
├── engine/       The app's link to the library
├── configs/      YAML configurations
├── src/
│   ├── data/     Dataset and data loading
│   ├── models/   Model creation
│   ├── losses/   Loss functions
│   └── utils/    Metrics, logging, checkpoints, profiling
├── scripts/      Training, evaluation, inference and export
├── tests/unit/   Unit tests
└── docs/         Documentation
```

## Development

```bash
make install-dev   # Development dependencies and pre-commit hooks
make test          # Run the tests
make test-cov      # Run the tests with a coverage report
make format        # Format with black and isort
make lint          # Run the linters
make ci            # Run the full local pipeline
```

## Documentation

- [Quick Start](QUICKSTART_OPTIMIZED.md)
- [Optimization Features](OPTIMIZATION_FEATURES.md)
- [Multi-GPU Training](MULTI_GPU_TRAINING.md)
- [Training Scripts](TRAINING_SCRIPTS_COMPARISON.md)
- [Architecture](ARCHITECTURE.md)
- [CI/CD](CI_CD.md)
- [Contributing](CONTRIBUTING.md)

## Acknowledgments

Built with [PyTorch](https://pytorch.org/), [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) and [Albumentations](https://albumentations.ai/).
