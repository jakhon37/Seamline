# Quick Reference: CI/CD Commands

## 🚀 Most Common Commands

```bash
# Setup (first time only)
make install-dev

# Before committing
make format          # Format code
make check          # Check quality (fast)

# Before pushing/PR
make test           # Run tests
make ci             # Full CI pipeline

# View all commands
make help
```

## 📋 Development Workflow

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes
# ... code ...

# 3. Format & test
make format && make test-unit

# 4. Commit (hooks run automatically)
git commit -am "feat: my feature"

# 5. Pre-push check
make ci

# 6. Push
git push
```

## 🔍 Troubleshooting

### Pre-commit hooks failing?

```bash
# Fix formatting issues
make format

# Check what's wrong
make lint

# Update hooks
make pre-commit-update
```

### Tests failing?

```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_file.py::test_function -v

# Run without coverage (faster)
pytest tests/ -v --no-cov
```

### Import errors?

```bash
# Reinstall in development mode
pip install -e .

# Or
make install-dev
```

## 📊 Coverage

```bash
# Generate coverage report
make test-cov

# View in browser
open htmlcov/index.html
```

## 🎯 Test Markers

```bash
# Unit tests only (fast)
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"

# Smoke tests (fastest)
make test-smoke
```

## 🔧 Configuration Files

- `.github/workflows/` - GitHub Actions
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pyproject.toml` - Tool configuration
- `.flake8` - Linting rules
- `Makefile` - Command shortcuts
- `requirements-dev.txt` - Dev dependencies
