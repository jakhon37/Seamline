# Unit Tests Documentation

## Overview

This directory contains the unit tests for mySegLib. There are 143 tests, and 10 of them need a GPU and skip on CPU.

## Test Structure

```
tests/unit/
├── __init__.py
├── test_dataset.py      # Dataset and data loading tests (15 tests)
├── test_models.py       # Model creation and architecture tests (27 tests)
├── test_losses.py       # Loss function tests (48 tests)
├── test_metrics.py      # Metrics computation tests (27 tests)
└── test_utils.py        # Utility functions tests (24 tests)
```

## Test Files

### 1. test_dataset.py (17 tests)

Tests for `SegmentationDataset` class and data loading utilities:

**TestSegmentationDataset:**
- ✅ Dataset initialization with valid paths
- ✅ Filtering of invalid file extensions
- ✅ Error handling for mismatched image/mask counts
- ✅ `__getitem__` with class indices mode (default)
- ✅ `__getitem__` with one-hot encoding mode (legacy)
- ✅ Caching functionality
- ✅ Mask value clipping to valid range
- ✅ Transform/augmentation integration
- ✅ Empty dataset handling
- ✅ Multi-class segmentation support

**TestGetImageMaskPairs:**
- ✅ Basic image-mask pairing
- ✅ File sorting
- ✅ Filtering of _mask.tif files
- ✅ Multiple file extension support
- ✅ Empty directory handling

### 2. test_models.py (27 tests)

Tests for segmentation model creation and operations:

**TestCreateModel:**
- ✅ U-Net, U-Net++, DeepLabV3, DeepLabV3+, FPN, PSPNet creation
- ✅ Invalid architecture error handling
- ✅ Different encoder backbones (ResNet18/34/50)
- ✅ Different number of classes
- ✅ Activation function options
- ✅ Gradient checkpointing
- ✅ Model forward pass
- ✅ Different input sizes
- ✅ Trainability and backpropagation

**TestGetModelInfo:**
- ✅ Model parameter counting
- ✅ Trainable vs total parameters
- ✅ Frozen parameter handling

**TestModelIntegration:**
- ✅ Batch processing
- ✅ Memory efficiency
- ✅ CUDA compatibility (GPU tests)

### 3. test_losses.py (48 tests)

Tests for all loss functions:

**TestFocalLoss:**
- ✅ Initialization and configuration
- ✅ Class weight support
- ✅ Forward pass with class indices
- ✅ Forward pass with one-hot targets
- ✅ Different gamma values
- ✅ Backpropagation
- ✅ Reduction modes (mean, sum, none)

**TestDiceLoss:**
- ✅ Initialization
- ✅ Forward pass with class indices
- ✅ Forward pass with one-hot targets
- ✅ Backpropagation

**TestCombinedLoss:**
- ✅ Initialization
- ✅ Weighted combination of losses
- ✅ Different alpha values
- ✅ Backpropagation

**TestCreateLoss:**
- ✅ Factory creation of all loss types (focal, dice, cross_entropy, combined, tversky)
- ✅ Class weight integration
- ✅ Invalid loss type error handling
- ✅ Compatibility with one-hot targets
- ✅ Multi-class segmentation

**TestLossEdgeCases:**
- ✅ Perfect predictions
- ✅ Uniform (uncertain) predictions
- ✅ Single pixel inputs
- ✅ Gradient flow verification

### 4. test_metrics.py (27 tests)

Tests for segmentation metrics:

**TestSegmentationMetrics:**
- ✅ Initialization with CPU/GPU computation
- ✅ Update with class indices
- ✅ Update with logits
- ✅ Update with one-hot targets
- ✅ Computation of all metrics (F1, Precision, Recall, IoU)
- ✅ Reset functionality
- ✅ Perfect prediction metrics (should be 1.0)
- ✅ Worst prediction metrics
- ✅ Multiple batch accumulation
- ✅ Multi-class support
- ✅ CPU computation for GPU memory savings
- ✅ Edge cases (single class predictions)

**TestDiceCoefficient:**
- ✅ Perfect overlap
- ✅ No overlap
- ✅ Partial overlap
- ✅ Smoothing to prevent division by zero
- ✅ Different smoothing values
- ✅ Binary and multi-class segmentation
- ✅ Different input shapes

**TestMetricsIntegration:**
- ✅ Training loop simulation
- ✅ Model output compatibility
- ✅ Batch consistency
- ✅ GPU memory efficiency (GPU tests)

### 5. test_utils.py (24 tests)

Tests for utility functions:

**TestMemoryTracker:**
- ✅ CPU initialization (disabled when no GPU)
- ✅ GPU initialization (GPU tests)
- ✅ Disabled tracker behavior
- ✅ Memory statistics retrieval
- ✅ Formatted statistics
- ✅ Peak memory tracking
- ✅ Peak reset functionality
- ✅ GPU memory summary
- ✅ Invalid device ID handling

**TestConfigValidator:**
- ✅ Valid configuration validation
- ✅ Missing section detection
- ✅ Missing key detection
- ✅ Invalid value detection (image_size, num_classes, epochs, etc.)
- ✅ Invalid architecture detection
- ✅ Invalid loss type detection
- ✅ Combined loss alpha validation
- ✅ Environment variable expansion
- ✅ Individual config section validation
- ✅ Low num_workers warning

**TestAverageMeter:**
- ✅ Initialization
- ✅ Single value updates
- ✅ Updates with custom counts
- ✅ Reset functionality
- ✅ Multiple updates and averaging
- ✅ Weighted average computation
- ✅ String representation
- ✅ Zero division edge case

**TestDistributedUtils:**
- ✅ Main process detection (single and distributed)
- ✅ World size retrieval
- ✅ Rank retrieval
- ✅ Tensor reduction
- ✅ Mocked distributed behavior

**TestUtilsIntegration:**
- ✅ Memory tracker in training loop
- ✅ Average meter for metric tracking
- ✅ Config validation pipeline
- ✅ Distributed utilities consistency

## Running Tests

### Run all unit tests:
```bash
pytest tests/unit/ -v
```

### Run specific test file:
```bash
pytest tests/unit/test_dataset.py -v
```

### Run specific test class:
```bash
pytest tests/unit/test_dataset.py::TestSegmentationDataset -v
```

### Run specific test:
```bash
pytest tests/unit/test_dataset.py::TestSegmentationDataset::test_initialization -v
```

### Run tests with markers:
```bash
# Run only GPU tests
pytest tests/unit/ -v -m gpu

# Skip GPU tests
pytest tests/unit/ -v -m "not gpu"

# Run only slow tests
pytest tests/unit/ -v -m slow
```

### Run tests with coverage (requires pytest-cov):
```bash
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term
```

## Test Markers

- `unit`: Unit tests (all tests in this directory)
- `integration`: Integration tests
- `slow`: Slow-running tests
- `gpu`: Tests requiring GPU
- `distributed`: Tests requiring multi-GPU setup

## Fixtures

Tests use fixtures defined in `tests/conftest.py`:

- `project_root`: Project root directory path
- `test_data_dir`: Temporary test data directory
- `sample_image`: Random test image (224x224x3)
- `sample_mask`: Random test mask (224x224)
- `sample_config`: Complete configuration dictionary
- `mock_dataset`: Mock dataset with directory structure
- `device`: Available device (CUDA if available, else CPU)
- `set_random_seed`: Sets random seeds for reproducibility
- `cleanup_cuda`: Cleans up CUDA memory after tests

## Test Guidelines

Each test follows these principles:

1. **Clear naming**: `test_<functionality>` format
2. **Docstrings**: Every test has a descriptive docstring
3. **Assertions with messages**: Clear failure messages
4. **Independence**: Tests can run independently
5. **Edge cases**: Tests cover normal, edge, and error cases
6. **Deterministic**: Uses fixed seeds where needed

## Coverage

Run `make test-cov` for a current report.

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run unit tests
  run: |
    pytest tests/unit/ -v --tb=short \
      --junitxml=test-results/unit-tests.xml
```

## Performance

Test execution times:
- **All unit tests**: ~7 seconds
- **Dataset tests**: ~1 second
- **Model tests**: ~2 seconds
- **Loss tests**: ~1 second
- **Metrics tests**: ~2 seconds
- **Utils tests**: ~1 second

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Add docstrings to all test functions
3. Use appropriate fixtures from conftest.py
4. Mark GPU-specific tests with `@pytest.mark.gpu`
5. Ensure tests are independent and reproducible
6. Add assertions with clear failure messages
7. Test both success and failure cases
8. Update this README with new test information

## Troubleshooting

### Tests fail with "CUDA not available"
- GPU tests are automatically skipped when CUDA is unavailable
- Run with `-m "not gpu"` to skip GPU tests explicitly

### Tests fail with module import errors
- Ensure you're in the project root directory
- Verify `src` is in the Python path (handled by conftest.py)
- Check that all dependencies are installed

### Tests are slow
- Use `-k` to run specific tests: `pytest -k test_initialization`
- Skip slow tests: `pytest -m "not slow"`
- Use parallel execution: `pytest -n auto` (requires pytest-xdist)

### Random test failures
- Some tests use random data; check if seeds are set properly
- GPU memory issues: Use `cleanup_cuda` fixture or reduce batch sizes
- Timing-sensitive tests: May fail on slow systems

## License

These tests are part of the mySegLib project and follow the same license.
