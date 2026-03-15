#!/bin/bash
# =============================================================================
# Semantic Segmentation - Development Environment Setup Script
# =============================================================================
# This script automates the setup of a development environment for the
# semantic segmentation framework.
#
# Usage:
#   bash scripts/setup_dev.sh [OPTIONS]
#
# Options:
#   --cuda-version VERSION    Specify CUDA version (default: 12.1)
#   --python-version VERSION  Specify Python version (default: 3.10)
#   --skip-cuda               Skip CUDA/PyTorch installation
#   --skip-tests              Skip running tests after setup
#   --no-pre-commit           Skip pre-commit hooks installation
#   --help                    Show this help message
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CUDA_VERSION="121"
PYTHON_VERSION="3.10"
SKIP_CUDA=false
SKIP_TESTS=false
NO_PRE_COMMIT=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-cuda)
            SKIP_CUDA=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-pre-commit)
            NO_PRE_COMMIT=true
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //g' | sed 's/^#//g'
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Setup
# =============================================================================

print_header "Semantic Segmentation - Development Setup"

# Change to project root
cd "$PROJECT_ROOT"
print_info "Project root: $PROJECT_ROOT"

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================

print_header "Step 1: Checking Prerequisites"

# Check Python
if check_command python$PYTHON_VERSION; then
    PYTHON_CMD="python$PYTHON_VERSION"
    print_success "Found Python $PYTHON_VERSION: $(which $PYTHON_CMD)"
elif check_command python3; then
    PYTHON_CMD="python3"
    PYTHON_VERSION_FOUND=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_warning "Python $PYTHON_VERSION not found, using $PYTHON_VERSION_FOUND"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check pip
if check_command pip3; then
    PIP_CMD="pip3"
    print_success "Found pip3: $(which pip3)"
else
    print_error "pip3 not found. Please install pip."
    exit 1
fi

# Check git
if check_command git; then
    print_success "Found git: $(which git)"
else
    print_warning "git not found. Version control features will be limited."
fi

# Check CUDA (if not skipped)
if [ "$SKIP_CUDA" = false ]; then
    if check_command nvcc; then
        CUDA_VERSION_FOUND=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "Found CUDA: $CUDA_VERSION_FOUND"
    else
        print_warning "CUDA not found. GPU training will not be available."
        print_info "Install CUDA from: https://developer.nvidia.com/cuda-downloads"
    fi
    
    if check_command nvidia-smi; then
        print_success "NVIDIA driver detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "nvidia-smi not found. GPU may not be available."
    fi
fi

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================

print_header "Step 2: Setting Up Virtual Environment"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip, setuptools, wheel
print_info "Upgrading pip, setuptools, and wheel..."
pip install --quiet --upgrade pip setuptools wheel
print_success "Package installers upgraded"

# =============================================================================
# Step 3: Install PyTorch with CUDA Support
# =============================================================================

if [ "$SKIP_CUDA" = false ]; then
    print_header "Step 3: Installing PyTorch with CUDA $CUDA_VERSION"
    
    print_info "Installing PyTorch and torchvision..."
    pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/cu$CUDA_VERSION"
    
    # Verify PyTorch installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        print_success "PyTorch installed with CUDA support"
    else
        print_warning "PyTorch installed but CUDA is not available"
    fi
else
    print_header "Step 3: Installing PyTorch (CPU-only)"
    pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
    print_success "PyTorch (CPU) installed"
fi

# =============================================================================
# Step 4: Install Project Dependencies
# =============================================================================

print_header "Step 4: Installing Project Dependencies"

# Install main dependencies
if [ -f "docs/requirements.txt" ]; then
    print_info "Installing main dependencies from docs/requirements.txt..."
    pip install --quiet -r docs/requirements.txt
    print_success "Main dependencies installed"
else
    print_warning "docs/requirements.txt not found, skipping main dependencies"
fi

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    print_info "Installing development dependencies from requirements-dev.txt..."
    pip install --quiet -r requirements-dev.txt
    print_success "Development dependencies installed"
else
    print_warning "requirements-dev.txt not found, skipping dev dependencies"
fi

# Install package in editable mode
print_info "Installing package in editable mode..."
pip install --quiet -e .
print_success "Package installed in editable mode"

# =============================================================================
# Step 5: Setup Pre-commit Hooks
# =============================================================================

if [ "$NO_PRE_COMMIT" = false ]; then
    print_header "Step 5: Setting Up Pre-commit Hooks"
    
    if [ -f ".pre-commit-config.yaml" ]; then
        print_info "Installing pre-commit hooks..."
        pre-commit install
        print_success "Pre-commit hooks installed"
        
        print_info "Running pre-commit on all files (first time may be slow)..."
        pre-commit run --all-files || print_warning "Some pre-commit checks failed (this is normal on first run)"
    else
        print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
else
    print_info "Skipping pre-commit hooks installation (--no-pre-commit)"
fi

# =============================================================================
# Step 6: Create Directory Structure
# =============================================================================

print_header "Step 6: Creating Directory Structure"

DIRS=("checkpoints" "training_outputs" "data" "notebooks")

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    else
        print_info "Directory already exists: $dir"
    fi
done

# =============================================================================
# Step 7: Verify Installation
# =============================================================================

print_header "Step 7: Verifying Installation"

# Test imports
print_info "Testing Python imports..."

python -c "
import sys
import torch
import torchvision
import segmentation_models_pytorch as smp
import albumentations
import cv2
import numpy as np
from src.models.factory import ModelFactory
from src.data.dataset import SegmentationDataset
from src.losses.losses import FocalLoss, DiceLoss
from src.utils.config import ConfigLoader

print('✓ All imports successful')
print(f'✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
"

if [ $? -eq 0 ]; then
    print_success "All Python imports verified"
else
    print_error "Some imports failed. Please check the error messages above."
    exit 1
fi

# =============================================================================
# Step 8: Run Tests
# =============================================================================

if [ "$SKIP_TESTS" = false ]; then
    print_header "Step 8: Running Tests"
    
    if [ -d "tests" ]; then
        print_info "Running test suite..."
        pytest tests/ -v --tb=short -k "not slow" || print_warning "Some tests failed"
        print_success "Test suite completed"
    else
        print_warning "tests/ directory not found, skipping tests"
    fi
else
    print_info "Skipping tests (--skip-tests)"
fi

# =============================================================================
# Step 9: Display Next Steps
# =============================================================================

print_header "Setup Complete! 🎉"

echo ""
echo -e "${GREEN}Your development environment is ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "2. Prepare your dataset:"
echo -e "   - Place your data in ${BLUE}./data/${NC}"
echo "   - Update data paths in config files"
echo ""
echo "3. Run a test training:"
echo -e "   ${BLUE}python scripts/train.py --config configs/default.yaml${NC}"
echo ""
echo "4. For multi-GPU training:"
echo -e "   ${BLUE}bash scripts/train_4gpu.sh${NC}"
echo ""
echo "5. View TensorBoard logs:"
echo -e "   ${BLUE}tensorboard --logdir training_outputs${NC}"
echo ""
echo "6. Run tests:"
echo -e "   ${BLUE}pytest tests/ -v${NC}"
echo ""
echo "7. Format code:"
echo -e "   ${BLUE}make format${NC}"
echo ""
echo "8. Run linters:"
echo -e "   ${BLUE}make lint${NC}"
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - docs/ARCHITECTURE.md"
echo "  - docs/MULTI_GPU_TRAINING.md"
echo "  - CONTRIBUTING.md"
echo ""

# Create activation reminder file
cat > activate_venv.sh << 'EOF'
#!/bin/bash
# Quick activation script
source venv/bin/activate
echo "Virtual environment activated!"
echo "Run 'python scripts/train.py --help' to see training options"
EOF

chmod +x activate_venv.sh
print_info "Created quick activation script: ./activate_venv.sh"

print_success "Setup completed successfully!"

# =============================================================================
# Optional: Show System Information
# =============================================================================

echo ""
print_info "System Information:"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Python: $($PYTHON_CMD --version)"
echo "  Pip: $(pip --version | cut -d' ' -f1,2)"

if command -v nvidia-smi &> /dev/null; then
    echo "  GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
fi

echo ""
print_info "Installed Package Versions:"
pip list | grep -E "(torch|numpy|opencv|albumentations|segmentation-models-pytorch)" || true

echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
