# CI/CD Setup Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Development Environment

```bash
# Navigate to project
cd mySegLib

# Install all development dependencies and pre-commit hooks
make install-dev
```

This will:
- Install pytest, coverage tools
- Install black, flake8, isort, mypy
- Install pre-commit hooks
- Install security scanning tools

### Step 2: Verify Setup

```bash
# Check installation
make info

# View available commands
make help
```

### Step 3: Run Initial Tests

```bash
# Format existing code
make format

# Run tests
make test

# Check code quality
make lint
```

## 📋 What Was Installed

### GitHub Actions Workflows

**Location:** `.github/workflows/`

1. **tests.yml** - Automated testing
   - Runs on push/PR to main branches
   - Tests Python 3.8, 3.9, 3.10
   - Generates coverage reports
   - Uploads to Codecov

2. **lint.yml** - Code quality checks
   - Black formatting
   - isort import sorting
   - flake8 linting
   - mypy type checking
   - Bandit security scanning

### Pre-commit Hooks

**Location:** `.pre-commit-config.yaml`

Automatically run before each commit:
- ✓ Trailing whitespace removal
- ✓ End of file fixing
- ✓ YAML/JSON validation
- ✓ Black formatting
- ✓ isort import sorting
- ✓ flake8 linting
- ✓ Bandit security checks

### Configuration Files

1. **pyproject.toml** - Modern Python configuration
   - Project metadata
   - Tool configurations (black, isort, mypy, pytest)
   - Coverage settings
   - Build system

2. **.flake8** - Linting rules
   - Line length: 100
   - Complexity: 15
   - Ignore patterns for black compatibility

3. **Makefile** - Development shortcuts
   - 25+ commands for common tasks
   - Colored output
   - Cross-platform support

4. **requirements-dev.txt** - Development dependencies
   - Testing: pytest, pytest-cov, pytest-xdist
   - Formatting: black, isort
   - Linting: flake8 + plugins
   - Type checking: mypy
   - Security: bandit, safety

## 🔧 Using Pre-commit Hooks

### First Time Setup

```bash
# Install hooks (done automatically with make install-dev)
pre-commit install

# Test on all files
make pre-commit
```

### Daily Use

Pre-commit hooks run automatically when you commit:

```bash
git add .
git commit -m "feat: add new feature"
# Hooks run automatically here
```

If hooks fail:

```bash
# Fix formatting issues automatically
make format

# Check what's wrong
make lint

# Try commit again
git commit -m "feat: add new feature"
```

### Bypass Hooks (Not Recommended)

```bash
# Only use in emergencies
git commit --no-verify -m "urgent fix"
```

## 🧪 Testing Strategy

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component integration tests
└── e2e/            # End-to-end tests
```

### Test Markers

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.slow          # Slow test (>1s)
@pytest.mark.gpu           # Requires GPU
@pytest.mark.smoke         # Quick smoke test
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Unit tests only (fast)
make test-unit

# Integration tests
make test-integration

# Smoke tests (fastest)
make test-smoke

# Parallel execution
make test-fast

# Specific test
pytest tests/test_models.py::test_unet -v

# With markers
pytest -m "unit and not slow"
```

## 📊 Code Coverage

### Generating Reports

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html
# or
firefox htmlcov/index.html
```

### Coverage Goals

- Overall: >80%
- Critical modules: >90%
- Exclude: visualization, scripts

### CI/CD Integration

Coverage reports are automatically:
- Generated in CI
- Uploaded to Codecov
- Displayed in PR comments

## 🎨 Code Formatting

### Black (Code Formatter)

```bash
# Format all code
make format

# Check without modifying
make format-check

# Or directly
black src/ tests/ scripts/
```

**Settings:**
- Line length: 100
- Target: Python 3.8+
- String quotes: Double

### isort (Import Sorter)

```bash
# Sort imports
isort src/ tests/ scripts/

# Check only
isort --check-only src/
```

**Settings:**
- Profile: black-compatible
- Section order: STDLIB → THIRDPARTY → FIRSTPARTY → LOCALFOLDER

## 🔍 Linting

### flake8

```bash
# Run linter
flake8 src/ tests/ scripts/

# Or use make
make lint
```

**Configuration:**
- Max line length: 100
- Max complexity: 15
- Ignores: E203, E501, W503 (black conflicts)

### Common Issues

```bash
# Fix most issues automatically
make format

# Remaining issues usually require manual fixes
# Check .flake8 for ignored rules
```

## 🔒 Security Scanning

### Bandit (Code Security)

```bash
# Run security scan
make security

# Or directly
bandit -r src/
```

### Safety (Dependency Security)

```bash
# Check dependencies
safety check

# Update vulnerable packages
pip list --outdated
pip install --upgrade <package>
```

## 🔄 Complete Development Workflow

### Starting New Feature

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit code ...

# 3. Format code
make format

# 4. Run tests
make test-unit

# 5. Check quality
make check

# 6. Commit (hooks run automatically)
git add .
git commit -m "feat: add my feature"

# 7. Run full CI locally
make ci

# 8. Push
git push origin feature/my-feature
```

### Before Creating PR

```bash
# Full quality check
make ci

# This runs:
# 1. clean - Remove artifacts
# 2. lint - All linters
# 3. test-cov - Tests with coverage
```

### Fixing PR Comments

```bash
# Address formatting issues
make format

# Fix linting issues
make lint

# Re-run tests
make test

# Commit fixes
git add .
git commit -m "fix: address PR comments"
git push
```

## 🎯 Common Tasks

### Update Dependencies

```bash
# Update pre-commit hooks
make pre-commit-update

# Update Python packages
pip list --outdated
pip install --upgrade <package>

# Update requirements
pip freeze > requirements-dev.txt
```

### Clean Project

```bash
# Remove all artifacts
make clean

# Remove only test artifacts
make clean-test

# Remove only Python cache
make clean-pyc
```

### View Project Info

```bash
# Show project information
make info

# Show Git status
git status

# Show test results
pytest --collect-only
```

## 🐛 Troubleshooting

### "Module not found" errors

```bash
# Reinstall in development mode
pip install -e .

# Or
make install-dev
```

### Pre-commit hooks too slow

```bash
# Skip slow hooks temporarily
SKIP=mypy git commit -m "message"

# Or disable specific hooks in .pre-commit-config.yaml
```

### Tests failing in CI but passing locally

```bash
# Ensure clean environment
make clean
make install-dev
make test

# Check Python version matches CI
python --version
```

### Linting conflicts

```bash
# Format first (fixes most issues)
make format

# Check remaining issues
make lint

# Update configuration if needed
# Edit .flake8 or pyproject.toml
```

## 📚 Additional Resources

- [Full CI/CD Documentation](./CI_CD.md)
- [Quick Reference](./QUICK_REFERENCE.md)
- [pytest Documentation](https://docs.pytest.org/)
- [black Documentation](https://black.readthedocs.io/)
- [pre-commit Documentation](https://pre-commit.com/)

## ✅ Checklist

- [ ] Run `make install-dev`
- [ ] Verify with `make help`
- [ ] Run initial tests: `make test`
- [ ] Check code quality: `make lint`
- [ ] Format code: `make format`
- [ ] Test pre-commit hooks: make a commit
- [ ] Run full CI: `make ci`
- [ ] Read documentation: `docs/CI_CD.md`

---

**Need Help?**
- Check `make help` for all commands
- Read detailed docs in `docs/CI_CD.md`
- Review configuration files for customization
