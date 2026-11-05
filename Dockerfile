# ========================================
# Segmentation - Multi-Stage Dockerfile
# ========================================
# Production-ready Docker image with CUDA support
# Optimized for deployment and development workflows

# ========================================
# Base Stage - CUDA Runtime
# ========================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install core packages
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# ========================================
# Builder Stage - Install Dependencies
# ========================================
FROM base AS builder

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ========================================
# Development Stage
# ========================================
FROM builder AS development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy entire project
COPY . /app/

# Install package in editable mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/checkpoints \
    /app/training_outputs \
    /app/wandb \
    /data

# Set volumes
VOLUME ["/app/checkpoints", "/app/training_outputs", "/app/wandb", "/data"]

# Expose TensorBoard port
EXPOSE 6006

# Default command (can be overridden)
CMD ["bash"]

# ========================================
# Production Stage
# ========================================
FROM base AS production

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy only necessary application files
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY pyproject.toml README.md ./

# Install package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 seguser && \
    chown -R seguser:seguser /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/checkpoints \
    /app/training_outputs \
    /data && \
    chown -R seguser:seguser /app /data

# Switch to non-root user
USER seguser

# Set volumes
VOLUME ["/app/checkpoints", "/app/training_outputs", "/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU Available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]

# ========================================
# Testing Stage (for CI/CD)
# ========================================
FROM development AS testing

# Run tests
RUN pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

# Run linting
RUN black --check src/ tests/ scripts/ && \
    isort --check-only src/ tests/ scripts/ && \
    flake8 src/ tests/ scripts/ && \
    mypy src/

# Run security checks
RUN bandit -r src/ -f json -o bandit-report.json || true

# Labels
LABEL maintainer="Segmentation Framework Team"
LABEL version="0.1.0"
LABEL description="Semantic segmentation framework with CUDA support"
