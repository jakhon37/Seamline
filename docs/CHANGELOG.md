# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Seamline, a desktop app for labelling and training, in `app/` and `engine/`
- `--events` option in `scripts/train.py` for per-epoch progress

### Changed
- Images and masks pair by exact name, so one missing file can't shift every later pair

### Fixed
- Training applied softmax twice: `activation` from the config reached the model, and the losses
  apply softmax themselves. A perfect prediction still cost 0.30, and small classes could collapse
  into background. `activation` now applies at inference and export only. Saved models still load,
  but retrain any model trained before this fix.
- Local training could die with exit code -4 on CPUs with AVX but no AVX2

## [0.1.0] - 2026-11-05

First public release.

### Added
- YAML-driven training, evaluation and inference scripts
- Models from segmentation_models_pytorch, with focal, Dice, cross-entropy and combined losses
- Multi-GPU training with DDP, and a DataParallel fallback
- Mixed precision, gradient accumulation, gradient checkpointing and early stopping
- Chunked inference for large images, and ONNX export
- Unit tests, GitHub Actions workflows, pre-commit hooks and Docker support
