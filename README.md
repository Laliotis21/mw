# Multi-Agent Paper Trading Bot

CrewAI + Claude + Perplexity. A **discovery crew** of 4 scouts scans the live
market + the world to find what to trade; a **decision desk** of 3 agents then
debates each candidate and a hard 2% risk gate sizes it. **Paper trading only** —
no real orders are sent.

> ⚠️ Educational simulation. Not financial advice. Live trading with real
> capital carries substantial risk of loss; wire a real broker only after you
> understand and accept that.

## Architecture

```
DISCOVERY CREW (--discover, no preset ticker)
   World Macro Scout   ──┐  global risk tilt + themes
   Stock Scanner       ──┼─▶ Opportunity Ranker ──▶ OpportunityShortlist (top-N names)
   Crypto Scanner      ──┘
                                     │  per shortlisted asset ▼
DECISION DESK
   Researcher (Data Scout)   ──▶ MarketScanReport
   Analyst (Quant Signal)    ──▶ TradeSignal
   Risk Officer (2% cap)     ──▶ ExecutionTicket (JSON)
                                 └▶ execution.py (paper) ─▶ trade_log.json
```

The desk also runs standalone on a hand-picked `--asset`, skipping discovery.

| File           | Role |
|----------------|------|
| `config.py`    | Env loading + Pydantic schemas (the data contracts) |
| `scanners.py`  | Live movers engine — yfinance stock screeners + CoinGecko crypto |
| `tools.py`     | Perplexity/world scanners + the 4 scout tools |
| `agents.py`    | The 4 scout + 3 desk Claude-backed CrewAI agents |
| `tasks.py`     | Discovery tasks + sequential desk dataflow |
| `execution.py` | Paper broker + performance/drawdown tracking |
| `main.py`      | Orchestrator, discovery, daily-loss circuit breaker, CLI |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in ANTHROPIC_API_KEY + PERPLEXITY_API_KEY
```

## Run

```bash
python main.py --discover --phase open                  # auto-find + trade movers
python main.py --asset AAPL --phase pre_market           # hand-picked
python main.py --asset AAPL --asset MSFT --phase open    # batch
python main.py --summary                                 # stats only
```

Phases: `pre_market`, `open`, `mid_day`, `close`.

`--discover` runs the 4 scouts, ranks the live movers into a shortlist of
`MAX_CANDIDATES` (default 5), then runs the desk on each. Tune `SCAN_PER_SOURCE`
(movers pulled per source) and `MAX_CANDIDATES` in `.env`. Live world/macro intel
needs `RESEARCH_SOURCE=perplexity`; in `mock`/`candles` mode the macro backdrop is
neutral and ideas rank on technicals/movement alone.

## Risk model

- **Hard cap:** `MAX_RISK_PCT` (default 2% = $20 on $1,000) per trade. Enforced
  twice — in the Risk Officer's prompt *and* in `ExecutionTicket` validators, so
  a hallucinated oversized ticket fails schema validation before execution.
- **Position size:** `risk_dollars / abs(entry − stop)`.
- **Min reward:risk:** 1.5:1.
- **Circuit breaker:** run halts if intraday loss ≥ `DAILY_MAX_LOSS_PCT` (5%).

## Notes / gotchas

- `sonar-medium` is **deprecated** by Perplexity. Default here is `sonar`;
  override with `PERPLEXITY_MODEL` (`sonar` / `sonar-pro`).
- Model id `claude-3-5-sonnet-20241022` works; newer Claude models are stronger —
  swap via `ANTHROPIC_MODEL`.
- Fills are **simulated** (`execution.simulate_fill`). Replace that one function
  with an Alpaca/Binance call to go live — the `ExecutionTicket` contract stays
  identical, so nothing upstream changes.
```
