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
- Tiling dropped any leftover strip narrower than half a tile, so 30% of a 1404×1070 photo never
  reached training, and 53% of a 1500×1500 image. Tiles now cover each side fully.
- Local training could die with exit code -4 on CPUs with AVX but no AVX2: a race in MKL's first
  vector-math call. One single-threaded call at import took real runs from 6 of 12 crashing to 0 of
  30, and a stress loop to 0 of 6000. `MKL_ENABLE_INSTRUCTIONS=AVX` only hides it: the same loop
  still crashed 578 times.

## [0.1.0] - 2026-11-05

First public release.

### Added
- YAML-driven training, evaluation and inference scripts
- Models from segmentation_models_pytorch, with focal, Dice, cross-entropy and combined losses
- Multi-GPU training with DDP, and a DataParallel fallback
- Mixed precision, gradient accumulation, gradient checkpointing and early stopping
- Chunked inference for large images, and ONNX export
- Unit tests, GitHub Actions workflows, pre-commit hooks and Docker support
