.PHONY: help install install-dev test test-cov test-unit test-integration lint format format-check type-check security clean clean-pyc clean-test clean-build pre-commit docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Semantic Segmentation Framework - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml --cov-branch

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -v -m "unit" --maxfail=5

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m "integration" --maxfail=3

test-smoke: ## Run quick smoke tests
	@echo "$(BLUE)Running smoke tests...$(NC)"
	pytest tests/ -v -m "smoke" --maxfail=1

test-fast: ## Run tests in parallel (fast)
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest tests/ -v -n auto --maxfail=5

# Code quality targets
lint: ## Run all linters (flake8, black check, isort check, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "$(GREEN)→ Running flake8...$(NC)"
	flake8 src/ tests/ scripts/
	@echo "$(GREEN)→ Checking code format with black...$(NC)"
	black --check src/ tests/ scripts/
	@echo "$(GREEN)→ Checking import order with isort...$(NC)"
	isort --check-only src/ tests/ scripts/
	@echo "$(GREEN)→ Running type checker...$(NC)"
	mypy src/ || true
	@echo "$(GREEN)✓ All linters passed!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check --diff src/ tests/ scripts/
	isort --check-only --diff src/ tests/ scripts/

type-check: ## Run mypy type checker
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy src/

security: ## Run security checks (bandit)
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/ -f json -o bandit-report.json
	bandit -r src/
	@echo "$(GREEN)✓ Security check complete!$(NC)"

# Pre-commit targets
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

# Cleaning targets
clean: clean-pyc clean-test clean-build ## Remove all build, test, and Python artifacts

clean-pyc: ## Remove Python file artifacts
	@echo "$(BLUE)Cleaning Python artifacts...$(NC)"
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '*.egg' -exec rm -rf {} +

clean-test: ## Remove test and coverage artifacts
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .tox/
	rm -rf .mypy_cache/

clean-build: ## Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/

# Training and experiments
train: ## Run training script
	@echo "$(BLUE)Starting training...$(NC)"
	python scripts/train.py

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if [ -d "docs" ]; then \
		cd docs && make html; \
		echo "$(GREEN)✓ Documentation generated in docs/_build/html/$(NC)"; \
	else \
		echo "$(RED)No docs directory found$(NC)"; \
	fi

# CI/CD simulation
ci: clean lint test-cov ## Run full CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline completed successfully!$(NC)"

# Quick quality check
check: format-check lint ## Quick code quality check (formatting + linting)
	@echo "$(GREEN)✓ Code quality check passed!$(NC)"

# Show project info
info: ## Show project information
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  Python version:  $$(python --version)"
	@echo "  Pip version:     $$(pip --version | cut -d' ' -f2)"
	@echo "  Location:        $$(pwd)"
	@echo "  Git branch:      $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "  Git status:      $$(git status -s 2>/dev/null | wc -l) files changed"
