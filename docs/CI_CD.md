# CI/CD and Code Quality Infrastructure

This document describes the continuous integration, continuous deployment, and code quality infrastructure for the mySegLib project.

## 📋 Table of Contents

- [Overview](#overview)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Code Quality Tools](#code-quality-tools)
- [Development Workflow](#development-workflow)
- [Makefile Commands](#makefile-commands)

## 🎯 Overview

This project uses industry-standard tools and practices for maintaining code quality:

- **GitHub Actions** for automated testing and linting
- **Pre-commit hooks** for local quality checks
- **pytest** for comprehensive testing with coverage
- **black** for code formatting
- **flake8** for linting
- **isort** for import sorting
- **mypy** for type checking
- **bandit** for security scanning

## 🔄 GitHub Actions Workflows

### Test Workflow (`.github/workflows/tests.yml`)

Runs on every push and pull request to main/master/develop branches.

**Features:**
- Tests on Python 3.10, 3.11, and 3.12
- Parallel testing across versions
- Unit tests with coverage reporting
- Integration tests (separate job)
- Coverage upload to Codecov
- Test artifacts retention (30 days)

**Triggered by:**
- Push to main/master/develop branches
- Pull requests to main/master/develop branches
- Manual workflow dispatch

### Lint Workflow (`.github/workflows/lint.yml`)

Ensures code quality standards are met.

**Checks:**
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code linting (PEP 8 compliance)
- **mypy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking

## 🪝 Pre-commit Hooks

Pre-commit hooks run automatically before each commit to catch issues early.

### Setup

```bash
# Install pre-commit hooks
make install-dev

# Or manually
pip install pre-commit
pre-commit install
```

### Included Hooks

1. **File checks**: trailing whitespace, EOF, YAML/JSON validation
2. **Black**: Auto-format Python code
3. **isort**: Sort imports
4. **flake8**: Lint code
5. **Bandit**: Security checks
6. **nbQA**: Jupyter notebook formatting

### Running Manually

```bash
# Run on all files
make pre-commit

# Or
pre-commit run --all-files

# Update hooks
make pre-commit-update
```

## 🛠 Code Quality Tools

### Black (Code Formatter)

```bash
# Format code
make format

# Check formatting
make format-check

# Or directly
black src/ tests/ scripts/
```

**Configuration:** `pyproject.toml` → `[tool.black]`
- Line length: 100
- Target: Python 3.10, 3.11, 3.12

### isort (Import Sorter)

```bash
# Sort imports
isort src/ tests/ scripts/

# Check only
isort --check-only src/ tests/ scripts/
```

**Configuration:** `pyproject.toml` → `[tool.isort]`
- Profile: black-compatible
- Line length: 100

### flake8 (Linter)

```bash
# Run linter
flake8 src/ tests/ scripts/
```

**Configuration:** `.flake8`
- Max line length: 100
- Max complexity: 15
- Ignores: E203, E501, W503 (black conflicts)

### mypy (Type Checker)

```bash
# Type check
make type-check

# Or
mypy src/
```

**Configuration:** `pyproject.toml` → `[tool.mypy]`

### Bandit (Security Scanner)

```bash
# Security scan
make security

# Or
bandit -r src/
```

**Configuration:** `pyproject.toml` → `[tool.bandit]`

## 🔧 Development Workflow

### Initial Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd Seamline

# 2. Install development dependencies
make install-dev

# 3. Verify setup
make info
```

### Daily Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Format code
make format

# 4. Run tests
make test-unit

# 5. Run all quality checks
make check

# 6. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add new feature"

# 7. Push and create PR
git push origin feature/my-feature
```

### Before Creating PR

```bash
# Run full CI pipeline locally
make ci

# This runs:
# 1. clean
# 2. lint
# 3. test-cov
```

## 📦 Makefile Commands

### Installation

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make install-dev` | Install development dependencies + pre-commit hooks |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests |
| `make test-smoke` | Run quick smoke tests |
| `make test-fast` | Run tests in parallel |

### Code Quality

| Command | Description |
|---------|-------------|
| `make lint` | Run all linters |
| `make format` | Format code with black and isort |
| `make format-check` | Check formatting without changes |
| `make type-check` | Run mypy type checker |
| `make security` | Run security checks |
| `make check` | Quick quality check (format + lint) |
| `make ci` | Full CI pipeline locally |

### Pre-commit

| Command | Description |
|---------|-------------|
| `make pre-commit` | Run pre-commit on all files |
| `make pre-commit-update` | Update pre-commit hooks |

### Cleaning

| Command | Description |
|---------|-------------|
| `make clean` | Remove all artifacts |
| `make clean-pyc` | Remove Python artifacts |
| `make clean-test` | Remove test artifacts |
| `make clean-build` | Remove build artifacts |

### Other

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make info` | Show project information |
| `make train` | Run training script |
| `make docs` | Generate documentation |

## 🧪 Testing Best Practices

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_basic_function():
    assert True

@pytest.mark.integration
def test_model_training():
    pass

@pytest.mark.slow
def test_full_pipeline():
    pass

@pytest.mark.gpu
def test_gpu_training():
    pass
```

Run specific test types:

```bash
# Unit tests only
pytest -m unit

# Exclude slow tests
pytest -m "not slow"

# GPU tests only
pytest -m gpu
```

### Coverage

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html

# Coverage config in pyproject.toml
```

## 🔒 Security

### Bandit

Scans for common security issues:
- Hardcoded passwords
- SQL injection
- Shell injection
- Insecure random number generators

### Safety

Checks dependencies for known vulnerabilities:

```bash
pip install safety
safety check
```

## 📝 Configuration Files

| File | Purpose |
|------|---------|
| `.github/workflows/tests.yml` | Test automation |
| `.github/workflows/lint.yml` | Linting automation |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `pyproject.toml` | Modern Python tool configuration |
| `.flake8` | Flake8 linter configuration |
| `Makefile` | Development commands |
| `requirements-dev.txt` | Development dependencies |
| `pytest.ini` | Pytest configuration |

## 🚀 Quick Start Checklist

- [ ] Run `make install-dev` to set up environment
- [ ] Run `make test` to verify tests pass
- [ ] Run `make lint` to check code quality
- [ ] Run `make format` before committing
- [ ] Ensure pre-commit hooks are installed
- [ ] Create feature branch for changes
- [ ] Run `make ci` before creating PR

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [black documentation](https://black.readthedocs.io/)
- [flake8 documentation](https://flake8.pycqa.org/)
- [pre-commit documentation](https://pre-commit.com/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

---
