# Contributing to mySegLib

First off, thank you for considering contributing to the mySegLib project! It's people like you that make this project such a great tool for the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA 12.1+ (for GPU support)
- Git
- Basic understanding of PyTorch and deep learning

### Areas for Contribution

We welcome contributions in several areas:

- **Bug fixes**: Fix identified issues or bugs
- **Features**: Implement new models, loss functions, or augmentations
- **Documentation**: Improve or expand documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize training or inference speed
- **Examples**: Add tutorials or example notebooks

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mySegLib.git
cd mySegLib

# Add upstream remote
git remote add upstream https://github.com/jakhon37/mySegLib.git
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting checks before each commit.

### 5. Verify Setup

```bash
# Run tests to verify everything is working
pytest tests/ -v

# Check code style
make lint
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes
- `perf/` - Performance improvements

### 2. Make Changes

- Write clear, readable code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Lint and Format

```bash
# Auto-format code
make format

# Or manually:
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Check linting
make lint

# Or manually:
flake8 src/ tests/ scripts/
mypy src/
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add focal loss implementation

- Implement focal loss with alpha and gamma parameters
- Add tests for focal loss computation
- Update documentation with usage examples"
```

Commit message format:
```
type: brief description (50 chars max)

- Detailed explanation of changes
- Why the change was made
- Any breaking changes or migration notes

Refs: #issue-number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `perf`, `chore`

## Code Style Guidelines

### Python Style

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized using `isort`
- **Formatting**: Use `black` for automatic formatting

### Code Quality Standards

1. **Type Hints**: Use type hints for all function signatures
   ```python
   def train_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
       """Train segmentation model."""
       pass
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
       """Compute segmentation loss.

       Args:
           pred: Model predictions of shape (B, C, H, W)
           target: Ground truth masks of shape (B, H, W)

       Returns:
           Computed loss value

       Raises:
           ValueError: If shapes are incompatible
       """
       pass
   ```

3. **Comments**: Write clear, concise comments for complex logic
   ```python
   # Use gradient checkpointing to reduce memory by 30%
   if config.get("gradient_checkpointing", False):
       model.encoder.set_gradient_checkpointing(True)
   ```

4. **Error Handling**: Use specific exceptions with clear messages
   ```python
   if num_classes < 2:
       raise ValueError(f"num_classes must be >= 2, got {num_classes}")
   ```

### File Organization

- Keep files focused and under 500 lines when possible
- Group related functionality together
- Use clear, descriptive names for files and classes
- Follow the existing project structure

## Testing Requirements

### Test Categories

All contributions must include appropriate tests:

1. **Unit Tests** (`@pytest.mark.unit`)
   - Test individual functions and classes
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (`@pytest.mark.integration`)
   - Test component interactions
   - Use real but minimal data
   - Moderate execution time

3. **GPU Tests** (`@pytest.mark.gpu`)
   - Only for GPU-specific functionality
   - Will be skipped in CPU-only environments

### Writing Tests

```python
import pytest
import torch
from src.models.factory import ModelFactory

class TestModelFactory:
    """Tests for model factory."""

    @pytest.mark.unit
    def test_create_unet(self):
        """Test U-Net model creation."""
        model = ModelFactory.create_model(
            architecture="unet",
            encoder="resnet34",
            num_classes=4
        )
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.unit
    def test_invalid_architecture(self):
        """Test error handling for invalid architecture."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ModelFactory.create_model(
                architecture="invalid",
                encoder="resnet34",
                num_classes=4
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_model_on_gpu(self):
        """Test model execution on GPU."""
        model = ModelFactory.create_model("unet", "resnet34", 4)
        model = model.cuda()
        x = torch.randn(2, 3, 224, 224).cuda()
        output = model(x)
        assert output.device.type == "cuda"
```

### Coverage Requirements

- Aim for **>80% code coverage** for new code
- All new features must have tests
- Bug fixes should include regression tests

## Pull Request Process

### 1. Pre-submission Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest tests/`)
- [ ] Code is formatted (`black` and `isort`)
- [ ] Linting passes (`flake8` and `mypy`)
- [ ] Security checks pass (`bandit`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commits are clean and well-described
- [ ] Branch is up-to-date with main

```bash
# Run full check
make ci
```

### 2. Create Pull Request

1. Push your branch to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template completely:
   - Clear title describing the change
   - Detailed description of what and why
   - Link related issues
   - Add screenshots/examples if relevant
   - Note any breaking changes

### 3. PR Title Format

```
type(scope): brief description

Examples:
feat(models): add DeepLabV3+ architecture
fix(training): resolve GPU memory leak in validation
docs(readme): update installation instructions
perf(dataloader): optimize image preprocessing pipeline
```

### 4. Code Review Process

1. **Automated Checks**: CI/CD pipeline must pass
2. **Peer Review**: At least one maintainer approval required
3. **Discussion**: Address reviewer feedback promptly
4. **Updates**: Push additional commits to your branch
5. **Merge**: Maintainer will merge once approved

### 5. After Merge

- Delete your feature branch
- Update your local main branch
- Close related issues if applicable

## Issue Reporting

### Bug Reports

Use the bug report template and include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - PyTorch version
   - CUDA version
   - GPU model
6. **Logs/Screenshots**: Error messages or relevant output

### Feature Requests

Use the feature request template and include:

1. **Problem**: What problem does this solve?
2. **Solution**: Proposed solution or API
3. **Alternatives**: Other approaches considered
4. **Use Case**: Real-world usage example
5. **Breaking Changes**: Any compatibility concerns

## Community

### Getting Help

- **Documentation**: Check [docs/](docs/) first
- **Issues**: Search existing issues before creating new ones

### Recognition

Contributors will be:
- Listed in CHANGELOG.md for their contributions
- Credited in release notes
- Added to CONTRIBUTORS.md (if it exists)

### Communication

- Be respectful and professional
- Provide constructive feedback
- Welcome newcomers
- Share knowledge generously

## Development Tips

### Debugging

```bash
# Run tests with verbose output
pytest tests/ -vv -s

# Debug specific test
pytest tests/test_models.py::TestUNet::test_forward -vv -s --pdb

# Profile code
python -m cProfile -o profile.stats scripts/train.py
```

### Performance Testing

```bash
# Benchmark training speed
python scripts/benchmark.sh

# Profile GPU memory
python scripts/train.py --profile-memory
```

### Documentation

```bash
# Build documentation (if Sphinx is set up)
cd docs/
make html

# View in browser
open _build/html/index.html
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to mySegLib! 🎉
