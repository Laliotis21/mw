"""
execution.py
============
Simulated broker. Receives an ExecutionTicket and paper-trades it, logging every
fill to a local JSON file for performance tracking.

This is deliberately a PAPER engine: no real orders are sent. It models a trade
deterministically — from the entry, the position either hits the take-profit or
the stop-loss — and records realized P&L plus a running equity curve so you can
measure win rate and max drawdown over many cycles.

Swap `simulate_fill` for a real Alpaca/Binance call when you go live; the ticket
contract (config.ExecutionTicket) stays identical, so nothing upstream changes.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import Action, ExecutionTicket, cap_quantity, logger, settings
from usage import cost_usd

TRADE_LOG = Path(__file__).parent / "trade_log.json"


# --------------------------------------------------------------------------- #
# Token-cost ledger — accumulates LLM spend in the log's meta so the dashboard
# can show real $/trade and total API spend.
# --------------------------------------------------------------------------- #
def _bump_meta_usage(log: dict, usage: Optional[dict]) -> float:
    """Roll one usage bundle into meta totals; return this bundle's dollar cost."""
    if not usage:
        return 0.0
    cost = cost_usd(
        usage.get("model", ""),
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        int(usage.get("cached_tokens", 0) or 0),
    )
    m = log["meta"].setdefault(
        "usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
         "requests": 0, "cost_usd": 0.0, "model": usage.get("model", "")},
    )
    m["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    m["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    m["cached_tokens"] += int(usage.get("cached_tokens", 0) or 0)
    m["requests"] += int(usage.get("requests", 0) or 0)
    m["cost_usd"] = round(m["cost_usd"] + cost, 6)
    if usage.get("model"):
        m["model"] = usage["model"]
    return cost


def log_usage(usage: Optional[dict]) -> None:
    """Record token usage not tied to a trade (e.g. the discovery crew)."""
    if not usage:
        return
    log = _load_log()
    _bump_meta_usage(log, usage)
    _save_log(log)


def set_last_run(summary: dict) -> None:
    """Persist a one-line summary of the most recent run for the dashboard."""
    log = _load_log()
    log["meta"]["last_run"] = summary
    _save_log(log)


# --------------------------------------------------------------------------- #
# Persistence helpers — the log is a single JSON doc: {meta, trades:[...]}.
# --------------------------------------------------------------------------- #
def _load_log() -> dict:
    if TRADE_LOG.exists():
        try:
            return json.loads(TRADE_LOG.read_text())
        except json.JSONDecodeError:
            logger.warning("trade_log.json corrupt — starting fresh.")
    return {
        "meta": {
            "starting_capital": settings.STARTING_CAPITAL,
            "equity": settings.STARTING_CAPITAL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "trades": [],
    }


def _save_log(log: dict) -> None:
    TRADE_LOG.write_text(json.dumps(log, indent=2))


def current_equity() -> float:
    """Latest equity from the log — used by the daily-loss circuit breaker."""
    return _load_log()["meta"]["equity"]


# --------------------------------------------------------------------------- #
# Fill simulation.
# --------------------------------------------------------------------------- #
def simulate_fill(
    ticket: ExecutionTicket,
    outcome: Optional[str] = None,
    market_phase: str = "mid_day",
) -> dict:
    """
    Model the trade's resolution.

    Dispatches on settings.FILL_SOURCE:
      - 'yfinance': resolve against REAL price bars (free, realistic).
      - 'coinflip': random TP/SL draw (fast, no network).
    `outcome` forces 'tp'/'sl' for deterministic tests (coinflip path only).
    """
    if ticket.action == Action.HOLD or ticket.quantity == 0:
        return {"result": "no_trade", "pnl": 0.0}

    # Live paper brokers: route per asset to a real bracket order. Returns an
    # 'open' record; reconcile_open() realizes P&L later. Anything a broker can't
    # route (wrong asset class / qty<min / API down) returns None → fall through
    # to the local sim below.
    #   binance -> crypto only | alpaca -> stocks only | live -> both
    if settings.FILL_SOURCE in ("binance", "alpaca", "live") and outcome is None:
        from binance_broker import is_crypto

        real = None
        if is_crypto(ticket.asset) and settings.FILL_SOURCE in ("binance", "live"):
            from binance_broker import place_bracket as _binance_bracket

            real = _binance_bracket(ticket)
        elif not is_crypto(ticket.asset) and settings.FILL_SOURCE in ("alpaca", "live"):
            from alpaca_broker import place_bracket as _alpaca_bracket

            real = _alpaca_bracket(ticket)
        if real is not None:
            return real
        logger.warning("Broker can't route %s — falling back to yfinance.", ticket.asset)

    if settings.FILL_SOURCE in ("yfinance", "binance", "alpaca", "live") and outcome is None:
        try:
            real = _yfinance_fill(ticket, market_phase)
            if real is not None:
                return real
            logger.warning(
                "yfinance: no data for %s — falling back to coinflip.", ticket.asset
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance fill failed for %s: %s — coinflip.", ticket.asset, exc)

    return _coinflip_fill(ticket, outcome)


def _coinflip_fill(ticket: ExecutionTicket, outcome: Optional[str]) -> dict:
    """Random TP/SL resolution using the ticket's own price geometry."""
    if outcome is None:
        rr = max(ticket.reward_risk_ratio, 0.01)
        p_win = min(0.7, max(0.25, 1.0 / (1.0 + rr)))
        outcome = "tp" if random.random() < p_win else "sl"

    per_unit = abs(ticket.entry_price - ticket.stop_loss)
    if outcome == "tp":
        gain_per_unit = abs(ticket.take_profit - ticket.entry_price)
        pnl = round(gain_per_unit * ticket.quantity, 2)
        result = "take_profit"
    else:
        pnl = round(-per_unit * ticket.quantity, 2)
        result = "stop_loss"

    return {"result": result, "pnl": pnl, "fill_source": "coinflip"}


def _yfinance_fill(ticket: ExecutionTicket, market_phase: str = "mid_day") -> Optional[dict]:
    """
    Resolve the trade against real market bars — WITHOUT lookahead.

    Uses the SAME period/interval the Researcher scanned, then resolves the trade
    only over candles.HOLDOUT_BARS — the most recent bars that candle_scan
    explicitly DROPS before computing its signals. So the agent decides on history
    strictly before the entry and we walk that out-of-sample window forward: entry
    is the open of the first held-out bar, then whichever of stop / target touches
    first decides the outcome; if neither, mark to the last close (a "markout").

    The agent's entry/stop/target may be approximate, so we keep only their
    RELATIVE geometry (risk % and reward %) and re-anchor to the real entry.
    Position size is recomputed so dollar risk respects the hard cap.
    Returns None if no usable price data (caller falls back to coinflip).
    """
    import yfinance as yf

    from candles import HOLDOUT_BARS, phase_data

    period, interval = phase_data(market_phase)
    df = yf.Ticker(ticket.asset).history(period=period, interval=interval)
    if df is None or df.empty or len(df) < HOLDOUT_BARS + 2:
        return None

    # Out-of-sample window: exactly the bars candle_scan held out of analysis.
    window = df.iloc[-HOLDOUT_BARS:]
    opens, closes, highs, lows = window["Open"], window["Close"], window["High"], window["Low"]
    entry = float(opens.iloc[0])  # enter at the open of the first unseen bar

    # Relative geometry from the agent's ticket (fractions of entry price).
    risk_pct = max(abs(ticket.entry_price - ticket.stop_loss) / ticket.entry_price, 0.001)
    rew_pct = max(abs(ticket.take_profit - ticket.entry_price) / ticket.entry_price, 0.001)

    is_long = ticket.action == Action.BUY
    if is_long:
        sl, tp = entry * (1 - risk_pct), entry * (1 + rew_pct)
    else:
        sl, tp = entry * (1 + risk_pct), entry * (1 - rew_pct)

    per_unit_risk = abs(entry - sl)
    qty = settings.MAX_RISK_DOLLARS / per_unit_risk if per_unit_risk > 0 else 0.0
    qty = cap_quantity(qty, entry, ticket.capital_at_open)  # no leverage: notional <= capital

    # Walk the window forward from the entry bar. On a bar that straddles both,
    # assume the stop hits first (conservative — never flatter the result).
    result, exit_price = "markout", float(closes.iloc[-1])
    for i in range(len(window)):
        hi, lo = float(highs.iloc[i]), float(lows.iloc[i])
        if is_long:
            if lo <= sl:
                result, exit_price = "stop_loss", sl
                break
            if hi >= tp:
                result, exit_price = "take_profit", tp
                break
        else:
            if hi >= sl:
                result, exit_price = "stop_loss", sl
                break
            if lo <= tp:
                result, exit_price = "take_profit", tp
                break

    pnl = (exit_price - entry) * qty if is_long else (entry - exit_price) * qty

    return {
        "result": result,
        "pnl": round(pnl, 2),
        "entry_price": round(entry, 2),
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "quantity": round(qty, 4),
        "exit_price": round(exit_price, 2),
        "risk_dollars": round(per_unit_risk * qty, 2),  # real risk on re-anchored qty
        "fill_source": "yfinance",
    }


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def execute_ticket(
    ticket: ExecutionTicket,
    outcome: Optional[str] = None,
    market_phase: str = "mid_day",
    usage: Optional[dict] = None,
) -> dict:
    """
    Paper-execute a ticket: simulate the fill, update equity, persist the record.

    `market_phase` is forwarded to the fill so it uses the same candles the
    Researcher scanned. `usage` is the CrewAI token usage for the cycle that
    produced this ticket — its dollar cost is stored on the record and rolled
    into the log's running API-spend total. Returns the trade record (also
    appended to trade_log.json).
    """
    log = _load_log()
    equity_before = log["meta"]["equity"]

    fill = simulate_fill(ticket, outcome=outcome, market_phase=market_phase)
    equity_after = round(equity_before + fill["pnl"], 2)

    # Attribute this cycle's LLM token cost to the trade + the run ledger.
    cycle_cost = _bump_meta_usage(log, usage)

    # Prefer real fill values (yfinance re-anchors price/qty) over the ticket's.
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "asset": ticket.asset,
        "action": ticket.action.value,
        "quantity": fill.get("quantity", ticket.quantity),
        "entry_price": fill.get("entry_price", ticket.entry_price),
        "stop_loss": fill.get("stop_loss", ticket.stop_loss),
        "take_profit": fill.get("take_profit", ticket.take_profit),
        "exit_price": fill.get("exit_price"),
        # Real-fill paths re-anchor qty to live price, so risk_dollars must come
        # from the fill too — else logged risk != quantity * stop distance.
        "risk_dollars": fill.get("risk_dollars", ticket.risk_dollars),
        "reward_risk_ratio": ticket.reward_risk_ratio,
        "result": fill["result"],
        "pnl": fill["pnl"],
        "fill_source": fill.get("fill_source", "coinflip"),
        "equity_before": equity_before,
        "equity_after": equity_after,
        "rationale": ticket.rationale,
        # Open-bracket broker refs (present only for live/testnet orders).
        "binance_symbol": fill.get("binance_symbol"),
        "binance_order_list_id": fill.get("binance_order_list_id"),
        "alpaca_order_id": fill.get("alpaca_order_id"),
        # LLM cost to produce this ticket (0.0 for free local models).
        "llm_cost_usd": round(cycle_cost, 6),
        "llm_tokens": (usage or {}).get("total_tokens", 0),
    }

    log["trades"].append(record)
    log["meta"]["equity"] = equity_after
    _save_log(log)

    logger.info(
        "EXECUTED %s %s qty=%s -> %s pnl=$%.2f equity=$%.2f",
        ticket.action.value,
        ticket.asset,
        ticket.quantity,
        fill["result"],
        fill["pnl"],
        equity_after,
    )
    return record


def open_positions_count() -> int:
    """How many broker brackets are still open (awaiting TP/SL). Cheap, no API."""
    return sum(1 for t in _load_log()["trades"] if t.get("result") == "open")


def reconcile_open() -> list[dict]:
    """
    Resolve any open broker brackets whose OCO has finished. Updates each trade's
    result/exit/pnl in place and rolls realized P&L into equity. Returns a list
    of the newly-CLOSED trades [{asset, result, pnl}] so callers can notify the
    user. Safe to call repeatedly (no-op when nothing has resolved).
    """
    import binance_broker
    import alpaca_broker

    log = _load_log()
    closed: list[dict] = []
    for t in log["trades"]:
        if t.get("result") != "open":
            continue
        src = t.get("fill_source", "")
        if src.startswith("binance"):
            out = binance_broker.reconcile_bracket(t)
        elif src.startswith("alpaca"):
            out = alpaca_broker.reconcile_bracket(t)
        else:
            out = None
        if out is None:
            continue
        result, exit_price, pnl = out
        t["result"] = result
        t["exit_price"] = exit_price
        t["pnl"] = pnl
        new_equity = round(log["meta"]["equity"] + pnl, 2)
        t["equity_after"] = new_equity
        log["meta"]["equity"] = new_equity
        closed.append({"asset": t["asset"], "result": result, "pnl": pnl})
        logger.info("RECONCILED %s %s pnl=$%.2f equity=$%.2f",
                    t["asset"], result, pnl, new_equity)
    if closed:
        _save_log(log)
    return closed


def performance_summary() -> dict:
    """Aggregate stats over the whole log: win rate, P&L, max drawdown."""
    log = _load_log()
    # 'open' = Binance bracket not yet resolved; not a settled trade yet.
    trades = [t for t in log["trades"] if t["result"] not in ("no_trade", "open")]
    start = log["meta"]["starting_capital"]
    equity = log["meta"]["equity"]

    # Win rate only over trades that actually hit TP or SL. "markout" trades
    # (mark-to-close, neither level touched) are not a win/loss signal — counting
    # a fractionally-positive markout as a win inflates the rate.
    resolved = [t for t in trades if t["result"] in ("take_profit", "stop_loss")]
    wins = [t for t in resolved if t["pnl"] > 0]
    # Max drawdown across the equity curve.
    peak = start
    max_dd = 0.0
    running = start
    for t in log["trades"]:
        running = t["equity_after"]
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        "starting_capital": start,
        "current_equity": equity,
        "net_pnl": round(equity - start, 2),
        "return_pct": round((equity - start) / start * 100, 2) if start else 0.0,
        "total_trades": len(trades),
        "resolved_trades": len(resolved),
        "win_rate_pct": round(len(wins) / len(resolved) * 100, 2) if resolved else 0.0,
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / peak * 100, 2) if peak else 0.0,
    }
