# Assets & Data — tickers, intervals, sessions

All market data comes from **yfinance** (free). Use the correct ticker format or
the data call returns empty.

## Ticker formats

| Class | Example tickers | Notes |
|-------|-----------------|-------|
| **Stocks / ETFs** | `AAPL`, `MSFT`, `SPY`, `NVDA` | US cash session 09:30–16:00 ET |
| **Crypto** | `BTC-USD`, `ETH-USD`, `SOL-USD` | 24/7, use `-USD` suffix |
| **FX** | `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` | 24/5, use `=X` suffix |

## Candle interval by horizon

| Horizon | Interval | Lookback window |
|---------|----------|-----------------|
| Intraday (default) | `15m` | last 5 days |
| Scalp | `5m` | last 1–2 days |
| Swing | `1h` / `1d` | last 1–3 months |

Default desk mode is **intraday on 15m candles**.

## Sessions (for phase interpretation)

- **Stocks:** phases map to the US cash session (pre_market → open → mid_day →
  close).
- **Crypto:** no official session — map phases to UTC blocks (pre_market = Asia,
  open = London open, mid_day = London/NY overlap, close = NY close).
- **FX:** 24/5 — same UTC-block mapping as crypto; watch session overlaps for
  volume.

## Data hygiene

- Always check the data is non-empty and recent before trusting a signal.
- If data is missing/stale → report `degraded` and default to **HOLD**.
- Never invent prices. If you don't have a candle value, say so.
