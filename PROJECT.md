# ALPHA·DESK — Multi-Agent Paper Trading Bot

A multi-agent trading bot that scans the market, decides trades with a desk of
AI agents, and paper-executes them against real prices. Built on CrewAI with a
Streamlit "pro-trader terminal" dashboard. **No real money moves** — fills
resolve via yfinance simulation, Binance Spot Testnet, or Alpaca paper.

---

## Quick start

```bash
# Use the project venv (its name is `v/`). The streamlit shebang is stale,
# so always launch via `python -m`.
v/bin/python -m streamlit run dashboard.py        # dashboard at http://localhost:8501

# CLI trade runs
v/bin/python main.py --asset AAPL --phase open                  # single ticker
v/bin/python main.py --asset AAPL --asset BTC-USD --phase open  # multiple
v/bin/python main.py --discover --phase open                    # scout + trade movers
v/bin/python main.py --summary                                  # stats only
v/bin/python main.py --reconcile                                # settle open broker brackets
```

Default `.env` runs the **free local stack**: `LLM_PROVIDER=ollama`
(`llama3.1:8b`), `RESEARCH_SOURCE=candles`, `FILL_SOURCE=yfinance`. Requires a
local ollama server (`http://localhost:11434`) with the model pulled.

---

## Architecture

Two agent crews feed one execution engine. Everything flows through strict
Pydantic contracts defined in `config.py` so a field rename can't silently
desync the pipeline.

### Decision desk (per asset)
```
Researcher → Analyst → Risk Manager → ExecutionTicket → execute_ticket()
```
- **Researcher** (`build_researcher`) → `MarketScanReport`: headlines, macro
  catalysts, sentiment score, key levels.
- **Analyst** (`build_analyst`) → `TradeSignal`: direction (BUY/SELL/HOLD),
  confidence, suggested entry/stop/target.
- **Risk Manager** (`build_risk_officer`) → `ExecutionTicket`: the ONLY object
  the broker accepts. Enforces the hard dollar-risk cap and stop-side geometry.

### Discovery crew (finds what to trade)
```
Macro Scout ─┐
Stock Scanner ┼─→ Opportunity Ranker → OpportunityShortlist
Crypto Scanner┘
```
4 scout agents scan live movers + macro, rank a shortlist; the desk then trades
the top `MAX_CANDIDATES`.

---

## Files

| File | Role |
|------|------|
| `config.py` | Settings (env, capital, risk) + all Pydantic schemas (`MarketScanReport`, `TradeSignal`, `ExecutionTicket`, `TradeIdea`, `OpportunityShortlist`). `cap_quantity` clamps notional ≤ capital (no leverage). |
| `agents.py` | CrewAI agent builders (desk + scouts). |
| `tasks.py` | CrewAI task builders wiring agents to prompts + context. |
| `tools.py` | Agent tools (market scanner, candle scan, etc.). |
| `scanners.py` | Live mover/coin scanners feeding the discovery crew. |
| `candles.py` | yfinance candle data + `phase_data(phase)` → (period, interval); `HOLDOUT_BARS` = out-of-sample window. |
| `main.py` | Orchestrator: `run_cycle`, `run_discovery`, circuit breaker, CLI. |
| `execution.py` | Paper broker: `simulate_fill`, `execute_ticket`, `reconcile_open`, `performance_summary`. Writes `trade_log.json`. |
| `alpaca_broker.py` | Alpaca paper stock brackets (BUY + SHORT). |
| `binance_broker.py` | Binance Spot Testnet crypto brackets (BUY only — spot can't short). |
| `dashboard.py` | Streamlit pro-trader terminal UI. |

---

## Fill routing (`execution.simulate_fill`)

Dispatches on `FILL_SOURCE`:
- `yfinance` — resolve against REAL price bars, **no lookahead** (walks the
  `HOLDOUT_BARS` window the scanner dropped; entry = open of first unseen bar;
  whichever of stop/target touches first wins; else mark-to-close = "markout").
- `coinflip` — random TP/SL using ticket geometry (fast, no network).
- `binance` / `alpaca` / `live` — place a REAL bracket (market entry + OCO
  exit). Records the trade as `"open"`; `reconcile_open()` polls later to
  realize P&L. Anything a broker can't route falls back to yfinance → coinflip.

Routing: `binance` → crypto only · `alpaca` → stocks only · `live` → both.
`is_crypto(asset)` = ticker ends `-USD` / `-USDT` / `USDT`.

---

## Risk & safety

- **Hard risk cap**: `MAX_RISK_DOLLARS = STARTING_CAPITAL × MAX_RISK_PCT`
  (default $20 on $1000 @ 2%). Enforced in `ExecutionTicket._risk_cap`.
- **Geometry invariant**: BUY needs `stop < entry < target`; SELL the reverse.
- **No leverage**: `cap_quantity` clamps notional ≤ capital.
- **Daily circuit breaker**: bot stands down if day drawdown ≥
  `DAILY_MAX_LOSS_PCT` (default 5%).
- **PAPER everywhere**: Binance defaults to `BINANCE_TESTNET=true`; Alpaca uses
  the paper endpoint. Only `BINANCE_TESTNET=false` touches real funds.

---

## Dashboard (`dashboard.py`)

Dark OLED "trading terminal" (Fira Code numerics / Fira Sans body, blue+amber
accents, green/red P&L). Layout:
- **Top status bar**: `ALPHA·DESK` brand · PAPER/LIVE chip · LLM/FILL chips ·
  UTC clock.
- **KPI strip**: equity · net P&L · win rate · trades (+open) · max drawdown ·
  risk cap.
- **Left — ORDER DESK**: auto-discover vs pick-ticker, session pills
  (pre/open/mid/close), EXECUTE button + live **AGENT FEED** that streams each
  agent's reasoning in real time (via CrewAI `task_callback`).
- **Right**: Altair equity curve (green/red gradient + dashed cost-basis rule),
  OPEN POSITIONS (live brackets awaiting reconcile), color-coded TRADE BLOTTER.
- Bottom expander: risk summary + "Flatten book" reset.

The dashboard forces the free local stack on run (`ollama` / `candles` /
`yfinance`) regardless of `.env`. Refresh the page after a run to roll new fills
into the KPI strip / curve / blotter.

Verify UI without a browser:
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("dashboard.py", default_timeout=30); at.run()
assert not at.exception
```

---

## trade_log.json

Single JSON doc: `{meta:{starting_capital, equity, created_utc}, trades:[...]}`.
Each trade record carries entry/stop/target, result (`take_profit` /
`stop_loss` / `markout` / `open` / `no_trade`), pnl, equity before/after,
fill_source, rationale, and broker order refs. `trade_log.bak.json` is a manual
backup. Delete `trade_log.json` (or use the dashboard reset) to start fresh.

---

## Gotchas

- Project venv is `v/` (not `.venv`). Its `streamlit`/`crewai` console scripts
  have a **stale shebang** → always run `v/bin/python -m <module>`.
- System `python3` lacks the deps; only `v/bin/python` has them.
- Local `llama3.1:8b` is slow: a full 3-agent cycle takes minutes; discovery
  (4 scouts + N cycles) takes much longer.
- The Researcher task text mentions "Perplexity" but the tool reads from the
  configured `RESEARCH_SOURCE` (candles in the free stack).
