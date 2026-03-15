#!/bin/bash
# CI/CD Infrastructure Validation Script

set -e

echo "============================================"
echo "CI/CD Infrastructure Validation"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

# Check if directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/"
        return 1
    fi
}

echo "📁 Checking files..."
echo "-------------------------------------------"

check_file ".github/workflows/tests.yml"
check_file ".github/workflows/lint.yml"
check_file ".pre-commit-config.yaml"
check_file "pyproject.toml"
check_file ".flake8"
check_file "Makefile"
check_file "requirements-dev.txt"
check_file "docs/CI_CD.md"
check_file "docs/QUICK_REFERENCE.md"

echo ""
echo "📋 Checking configurations..."
echo "-------------------------------------------"

# Validate YAML files
if command -v python &> /dev/null; then
    python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))" && \
        echo -e "${GREEN}✓${NC} .pre-commit-config.yaml syntax valid" || \
        echo -e "${RED}✗${NC} .pre-commit-config.yaml syntax invalid"
    
    python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))" && \
        echo -e "${GREEN}✓${NC} tests.yml syntax valid" || \
        echo -e "${RED}✗${NC} tests.yml syntax invalid"
    
    python -c "import yaml; yaml.safe_load(open('.github/workflows/lint.yml'))" && \
        echo -e "${GREEN}✓${NC} lint.yml syntax valid" || \
        echo -e "${RED}✗${NC} lint.yml syntax invalid"
else
    echo -e "${YELLOW}⚠${NC}  Python not found, skipping YAML validation"
fi

# Check Makefile
if make -n help &> /dev/null; then
    echo -e "${GREEN}✓${NC} Makefile syntax valid"
else
    echo -e "${RED}✗${NC} Makefile syntax invalid"
fi

echo ""
echo "🎯 Testing Makefile commands..."
echo "-------------------------------------------"

# Test help command
if make help &> /dev/null; then
    echo -e "${GREEN}✓${NC} make help works"
else
    echo -e "${RED}✗${NC} make help failed"
fi

# Test info command
if make info &> /dev/null; then
    echo -e "${GREEN}✓${NC} make info works"
else
    echo -e "${RED}✗${NC} make info failed"
fi

echo ""
echo "📊 Summary..."
echo "-------------------------------------------"

TOTAL_FILES=9
echo "Total files created: $TOTAL_FILES"
echo ""
echo -e "${GREEN}✅ CI/CD infrastructure validation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run 'make install-dev' to setup environment"
echo "  2. Run 'make help' to see all commands"
echo "  3. Run 'make test' to run tests"
echo "  4. Read docs/CI_CD.md for full documentation"
echo ""
echo "============================================"
