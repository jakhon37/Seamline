# System Architecture

This document provides a comprehensive overview of the mySegLib system architecture, including component interactions, data flows, and design decisions.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Training Pipeline](#training-pipeline)
- [Model Architecture](#model-architecture)
- [Inference Pipeline](#inference-pipeline)
- [Multi-GPU Architecture](#multi-gpu-architecture)
- [Configuration System](#configuration-system)
- [Logging and Monitoring](#logging-and-monitoring)
- [Design Decisions](#design-decisions)

---

## Overview

mySegLib is a modular deep learning framework for semantic segmentation. It is designed for scalability, maintainability, and high performance on both single-GPU and multi-GPU setups.

### Key Design Principles

1. **Modularity**: Components are loosely coupled and independently testable
2. **Configurability**: All parameters controlled via YAML configuration
3. **Scalability**: Efficient multi-GPU training with minimal code changes
4. **Performance**: Optimized data loading, mixed precision, and memory management
5. **Observability**: Comprehensive logging, metrics, and monitoring
6. **Reproducibility**: Deterministic training with seed control

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                         │
│                      (YAML Config Files)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  train.py    │  │ evaluate.py  │  │ inference.py │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Core Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Data    │  │  Models  │  │  Losses  │  │  Utils   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ PyTorch  │  │   CUDA   │  │ Storage  │  │ Logging  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Data Module (`src/data/`)

Responsible for data loading, preprocessing, and augmentation.

```
src/data/
├── __init__.py
├── dataset.py           # SegmentationDataset class
├── transforms.py        # Augmentation pipelines
└── utils.py            # Data utilities
```

**Key Classes:**

- **SegmentationDataset**: PyTorch Dataset for loading images and masks
  - Supports both one-hot and class-index encoding
  - Handles various image formats (TIF, PNG, JPG)
  - Applies augmentation transforms
  - Memory caching for validation data

- **TransformFactory**: Creates augmentation pipelines
  - Training transforms: rotation, flip, brightness, contrast, noise
  - Validation transforms: minimal (resize/normalize only)
  - Configurable via YAML

**Data Loading Flow:**

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│  Images  │───▶│   Dataset    │───▶│ DataLoader  │───▶│  Batches │
│  Masks   │    │              │    │             │    │  (GPU)   │
└──────────┘    └──────────────┘    └─────────────┘    └──────────┘
                      │                    │
                      │                    │
                      ▼                    ▼
                ┌─────────────┐      ┌──────────┐
                │ Transforms  │      │ Prefetch │
                │ (Augment)   │      │ Workers  │
                └─────────────┘      └──────────┘
```

**Optimizations:**

- Multi-worker prefetching (`num_workers=8`)
- Persistent workers between epochs
- Optimized class indices (vs one-hot encoding)
- Validation set caching in memory
- Pin memory for faster GPU transfer

### 2. Models Module (`src/models/`)

Provides model architectures and factory pattern for easy instantiation.

```
src/models/
├── __init__.py
├── factory.py          # ModelFactory
└── utils.py           # Model utilities
```

**Supported Architectures:**

- **U-Net**: Classic encoder-decoder with skip connections
- **U-Net++**: Nested U-Net with dense skip connections
- **DeepLabV3+**: Atrous convolutions with spatial pyramid pooling
- **FPN**: Feature Pyramid Network
- **PSPNet**: Pyramid Scene Parsing Network
- **PAN**: Pyramid Attention Network
- **LinkNet**: Efficient encoder-decoder
- **MANet**: Multi-scale attention network

**Encoder Backbones:**

- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- DenseNet (121, 169, 201)
- MobileNet V2
- VGG (11, 13, 16, 19)
- And 50+ more via `segmentation_models_pytorch`

**Model Creation Flow:**

```
┌───────────────┐
│ YAML Config   │
│ - architecture│
│ - encoder     │
│ - num_classes │
└───────┬───────┘
        │
        ▼
┌─────────────────┐
│ ModelFactory    │
│ .create_model() │
└───────┬─────────┘
        │
        ▼
┌─────────────────────────────────┐
│   Instantiated Model            │
│   - Encoder (pretrained)        │
│   - Decoder                     │
│   - Segmentation Head           │
└─────────────────────────────────┘
```

**Key Features:**

- Pretrained encoder weights (ImageNet)
- Gradient checkpointing for large models
- Automatic architecture selection
- Custom activation functions

### 3. Losses Module (`src/losses/`)

Implements various loss functions for semantic segmentation.

```
src/losses/
├── __init__.py
└── losses.py          # Loss implementations
```

**Available Losses:**

1. **Focal Loss**: Addresses class imbalance
   - `alpha`: Class weights
   - `gamma`: Focusing parameter (2.0 typical)

2. **Dice Loss**: Overlap-based metric
   - Smooth parameter for numerical stability
   - Works well for small objects

3. **Combined Loss**: Weighted combination
   - Typically: `alpha * Focal + (1-alpha) * Dice`
   - Balances pixel-wise and overlap objectives

4. **Cross-Entropy Loss**: Standard classification loss
   - Supports class weights
   - Fast and stable

5. **Tversky Loss**: Generalization of Dice
   - `alpha/beta`: Controls FP/FN trade-off
   - Useful for unbalanced classes

**Loss Selection Guide:**

```
Balanced dataset     ──▶ Cross-Entropy / Dice
Imbalanced dataset   ──▶ Focal Loss / Tversky Loss
Small objects        ──▶ Dice Loss / Combined Loss
Multi-class          ──▶ Combined Loss (Focal + Dice)
```

### 4. Utils Module (`src/utils/`)

Provides utilities for logging, metrics, checkpointing, and configuration.

```
src/utils/
├── __init__.py
├── config.py          # Configuration loader/validator
├── logger.py          # MetricLogger, TensorBoard, W&B
├── metrics.py         # IoU, F1, Precision, Recall
├── checkpoint.py      # CheckpointManager
└── helpers.py         # Utility functions
```

**Key Components:**

- **ConfigLoader**: YAML parsing and validation
- **MetricLogger**: Centralized logging (TB, W&B, file)
- **MetricsCalculator**: Compute IoU, F1, confusion matrix
- **CheckpointManager**: Save/load checkpoints, top-k retention
- **Helper Functions**: Seeding, device setup, visualization

---

## Data Flow

### Training Data Flow

```
┌─────────────┐
│   Dataset   │
│  (on disk)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Dataset         │
│ - Load image    │
│ - Load mask     │
│ - Apply augment │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  DataLoader     │
│ - Batch         │
│ - Shuffle       │
│ - Prefetch      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   GPU Memory    │
│ - Batch tensor  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Model Forward  │
│ - Predictions   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Loss Function  │
│ - Compute loss  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Backward Pass  │
│ - Gradients     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Optimizer Step │
│ - Update weights│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Metrics (CPU)  │
│ - IoU, F1, etc. │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    Logging      │
│ - TB, W&B, file │
└─────────────────┘
```

### Validation Data Flow

```
┌──────────────────┐
│ Validation Set   │
│ (cached in RAM)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Forward   │
│  (no grad)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Compute Metrics  │
│ (on CPU)         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Checkpoint Save? │
│ (if best model)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Log Results     │
└──────────────────┘
```

---

## Training Pipeline

### Single-GPU Training

```python
# Simplified training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass (with AMP)
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(images)
            loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics (on CPU)
        metrics = compute_metrics(predictions.cpu(), masks.cpu())

        # Logging
        logger.log(loss=loss.item(), **metrics)
```

### Multi-GPU Training (DDP)

```python
# Distributed training with DDP
def train_ddp(rank, world_size, config):
    # Setup process group
    setup(rank, world_size)

    # Create model and wrap with DDP
    model = create_model(config).to(rank)
    model = DDP(model, device_ids=[rank])

    # DistributedSampler ensures each GPU gets different data
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, ...)

    # Training loop (same as single-GPU)
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        train_epoch(model, dataloader, ...)

    cleanup()
```

### Training Pipeline Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     Training Pipeline                       │
└────────────────────────────────────────────────────────────┘

    [Load Config] ──▶ [Setup Environment] ──▶ [Create Model]
         │                   │                      │
         │                   │                      │
         ▼                   ▼                      ▼
    [Load Data]       [Init Logger]        [Load Checkpoint?]
         │                   │                      │
         │                   │                      │
         └───────────────────┴──────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Training Loop │
                    └────────┬───────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
         [Train Epoch] [Validate]  [Save Checkpoint]
                │            │            │
                │            │            │
                └────────────┴────────────┘
                             │
                             ▼
                    [Early Stopping?]
                             │
                    ┌────────┴────────┐
                    │                 │
                   YES               NO
                    │                 │
                    ▼                 │
              [Stop Training]         │
                                      │
                                      ▼
                              [Next Epoch]
```

---

## Model Architecture

### U-Net Architecture (Example)

```
Input Image (B, 3, H, W)
         │
         ▼
┌─────────────────────┐
│   Encoder Block 1   │  ──────┐
│   (Conv + Pool)     │        │ Skip Connection
└──────────┬──────────┘        │
           │                   │
           ▼                   │
┌─────────────────────┐        │
│   Encoder Block 2   │  ──────┤
│   (Conv + Pool)     │        │
└──────────┬──────────┘        │
           │                   │
           ▼                   │
┌─────────────────────┐        │
│   Encoder Block 3   │  ──────┤
│   (Conv + Pool)     │        │
└──────────┬──────────┘        │
           │                   │
           ▼                   │
┌─────────────────────┐        │
│     Bottleneck      │        │
│   (Conv + Conv)     │        │
└──────────┬──────────┘        │
           │                   │
           ▼                   │
┌─────────────────────┐        │
│   Decoder Block 1   │  ◀─────┘
│ (Upsample + Conv)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Decoder Block 2   │  ◀─────┘
│ (Upsample + Conv)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Decoder Block 3   │  ◀─────┘
│ (Upsample + Conv)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Segmentation Head  │
│   (1x1 Conv)        │
└──────────┬──────────┘
           │
           ▼
Output Mask (B, C, H, W)
```

### Key Architectural Features

1. **Encoder-Decoder Structure**: Downsample then upsample
2. **Skip Connections**: Preserve spatial information
3. **Pretrained Encoders**: Transfer learning from ImageNet
4. **Flexible Decoders**: Different strategies per architecture
5. **Segmentation Head**: Final classification layer

---

## Inference Pipeline

### Single Image Inference

```
┌──────────────┐
│ Load Image   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocess   │
│ - Resize     │
│ - Normalize  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Load Model   │
│ - Checkpoint │
│ - Config     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Forward Pass │
│ (no grad)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Postprocess  │
│ - Argmax     │
│ - Resize     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Save Mask    │
└──────────────┘
```

### Chunked Inference (Large Images)

For large images that don't fit in GPU memory:

```
┌──────────────────┐
│  Large Image     │
│  (4096x4096)     │
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  Split into Chunks     │
│  (224x224 with overlap)│
└────────┬───────────────┘
         │
         ▼
┌────────────────────┐
│  Process Each Chunk│  (Parallel)
│  - Forward pass    │
│  - Get prediction  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stitch Together   │
│  - Handle overlaps │
│  - Blend edges     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Full Prediction   │
│  (4096x4096)       │
└────────────────────┘
```

---

## Multi-GPU Architecture

### Distributed Data Parallel (DDP)

```
┌──────────────────────────────────────────────────────────┐
│                    Master Process (Rank 0)               │
│  - Coordinates training                                  │
│  - Saves checkpoints                                     │
│  - Aggregates metrics                                    │
└────────────┬─────────────────────────────────────────────┘
             │
   ┌─────────┴──────────────┬──────────────┬──────────────┐
   │                        │              │              │
   ▼                        ▼              ▼              ▼
┌──────┐                ┌──────┐       ┌──────┐       ┌──────┐
│ GPU0 │                │ GPU1 │       │ GPU2 │       │ GPU3 │
├──────┤                ├──────┤       ├──────┤       ├──────┤
│Model │                │Model │       │Model │       │Model │
│Copy  │                │Copy  │       │Copy  │       │Copy  │
└───┬──┘                └───┬──┘       └───┬──┘       └───┬──┘
    │                       │              │              │
    │  Gradient             │              │              │
    │  Synchronization      │              │              │
    │  (AllReduce)          │              │              │
    └───────────────────────┴──────────────┴──────────────┘
                            │
                            ▼
                   [Updated Weights]
                   (Synchronized across all GPUs)
```

### Communication Backend

1. **NCCL** (NVIDIA Collective Communications Library)
   - Optimized for NVIDIA GPUs
   - Best for NVLink-connected GPUs
   - Fallback to Gloo if fails

2. **Gloo**
   - CPU and GPU support
   - Works on PCIe-connected GPUs
   - More stable but slower

3. **DataParallel** (Emergency Fallback)
   - Single-process multi-GPU
   - Simpler but less efficient
   - Available via `train_dp.py`

### Gradient Accumulation

Enables training with larger effective batch sizes:

```
Effective Batch Size = Batch Size × GPUs × Accumulation Steps

Example:
- Batch size per GPU: 8
- Number of GPUs: 4
- Accumulation steps: 4
- Effective batch: 8 × 4 × 4 = 128

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Step 1   │  │ Step 2   │  │ Step 3   │  │ Step 4   │
│ Batch 8  │  │ Batch 8  │  │ Batch 8  │  │ Batch 8  │
│ (No opt) │  │ (No opt) │  │ (No opt) │  │ (Optimize)│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   │
                   ▼
          [Update Weights]
         (After 4 mini-batches)
```

---

## Configuration System

### Configuration Hierarchy

```yaml
# configs/default.yaml

data:
  base_dir: "/path/to/your/data"
  dataset_dir: "dataset"
  image_size: 224
  num_classes: 2
  use_onehot_encoding: false  # Optimized
  cache_validation: true

model:
  architecture: "unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  activation: "softmax"
  gradient_checkpointing: false

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 8              # Optimized
  prefetch_factor: 2          # Optimized
  persistent_workers: true    # Optimized
  mixed_precision: true
  gradient_accumulation_steps: 1
  metrics_on_cpu: true        # Optimized

loss:
  type: "combined"
  focal_alpha: 0.25
  focal_gamma: 2.0
  dice_smooth: 1.0
  combined_alpha: 0.5

optimizer:
  type: "adam"
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: "reduce_on_plateau"
  mode: "max"
  factor: 0.5
  patience: 5

logging:
  tensorboard: true
  wandb: false
  log_interval: 10

paths:
  output_dir: "./training_outputs"
  checkpoint_dir: "./checkpoints"
```

### Configuration Loading

```python
from src.utils.config import ConfigLoader

# Load and validate configuration
config = ConfigLoader.load_config("configs/default.yaml")

# Override with command-line args
config = ConfigLoader.merge_args(config, args)

# Validate
ConfigLoader.validate(config)
```

---

## Logging and Monitoring

### Logging Architecture

```
┌────────────────────────────────────────────────────────┐
│                   MetricLogger                         │
│  - Central logging hub                                 │
│  - Aggregates metrics from all sources                 │
└────────┬─────────────┬─────────────┬───────────────────┘
         │             │             │
         ▼             ▼             ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│TensorBoard │  │ Weights &  │  │   File     │
│  Logs      │  │   Biases   │  │  Logging   │
└────────────┘  └────────────┘  └────────────┘
    │               │               │
    ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Scalars  │  │Dashboard │  │ history  │
│ Images   │  │ Plots    │  │  .json   │
│ Histogr. │  │ Artifacts│  │ metrics  │
└──────────┘  └──────────┘  └──────────┘
```

### Metrics Tracked

**Training Metrics:**
- Loss (per batch and epoch)
- Learning rate
- GPU memory usage
- Training time per epoch

**Validation Metrics:**
- IoU (per class and mean)
- F1 Score (per class and mean)
- Precision (per class and mean)
- Recall (per class and mean)
- Confusion matrix

**System Metrics:**
- GPU utilization
- Memory usage (GPU and system)
- Data loading time
- Forward/backward pass time

---

## Design Decisions

### Why Modular Architecture?

**Pros:**
- Easy to test individual components
- Can swap implementations (e.g., different models)
- Clear separation of concerns
- Easier to maintain and extend

**Trade-offs:**
- More files and directories
- Slight overhead from abstraction
- Learning curve for new contributors

### Why YAML Configuration?

**Pros:**
- Human-readable and editable
- Version control friendly
- Easy to share experiments
- No code changes needed for new experiments

**Trade-offs:**
- Validation overhead
- Can be verbose for complex configs

### Why Optimized Class Indices?

**Pros:**
- Faster than one-hot encoding
- Less memory usage
- Better GPU cache utilization

**Trade-offs:**
- Requires careful loss function handling
- Need to ensure compatibility

### Why Multi-Worker Data Loading?

**Pros:**
- Faster training
- Better CPU utilization
- Reduces GPU idle time

**Trade-offs:**
- More memory usage
- Can cause issues on some systems

### Why Metrics on CPU?

**Pros:**
- Frees some GPU memory
- Allows larger batch sizes
- Reduces GPU memory fragmentation

**Trade-offs:**
- Small CPU overhead
- Requires careful tensor transfers

---

## Performance Characteristics

### Memory Usage

```
Component              | Memory Impact
-----------------------|---------------
Model (ResNet34-UNet)  | ~200MB
Batch (16×3×224×224)   | ~50MB
Gradients              | ~200MB
Optimizer State        | ~400MB
Activations (no ckpt)  | ~1GB
Activations (ckpt)     | ~300MB
-----------------------|---------------
Total (no ckpt)        | ~1.85GB
Total (ckpt)           | ~1.15GB
```

---

## Future Enhancements

1. **Model Optimization**
   - TensorRT integration
   - ONNX Runtime optimization
   - Model pruning and quantization

2. **Distributed Training**
   - Multi-node training support
   - Horovod integration
   - Kubernetes operators

3. **AutoML**
   - Neural architecture search
   - Hyperparameter optimization
   - Automated data augmentation

4. **Production Deployment**
   - Model serving (TorchServe, Triton)
   - API endpoints (FastAPI, gRPC)
   - Edge deployment (TFLite, ONNX)

5. **Advanced Features**
   - Active learning
   - Semi-supervised learning
   - Uncertainty quantification

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

**Version**: 2.0.0

For questions or suggestions about the architecture, please open an issue or discussion on GitHub.
