# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-05

First public release.

### Added
- YAML-driven training, evaluation and inference scripts
- Models from segmentation_models_pytorch, with focal, Dice, cross-entropy and combined losses
- Multi-GPU training with DDP, and a DataParallel fallback
- Mixed precision, gradient accumulation, gradient checkpointing and early stopping
- Chunked inference for large images, and ONNX export
- Unit tests, GitHub Actions workflows, pre-commit hooks and Docker support
