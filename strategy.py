"""
strategy.py
===========
Deterministic, LLM-free decision engine. Reads real candles and turns plain
technical rules into a TradeSignal -> ExecutionTicket, plus a rules-based
discovery shortlist. No API calls, no tokens, no HOLD-bias, fully reproducible.

Used when DECISION_ENGINE=rules (the default). The CrewAI/Claude path is kept
for DECISION_ENGINE=llm — worth it only once live news/sentiment feeds the desk
(candles alone are deterministic and don't need a model to interpret).

Trade logic (swing/intraday on daily bars):
    BUY  : uptrend (close>SMA20>SMA50) + healthy RSI, OR a 20-bar breakout.
    SELL : downtrend (close<SMA20<SMA50) + weak RSI, OR a 20-bar breakdown.
    HOLD : no clean edge.
Stops are ATR-based (1.5x), targets are 2R (reward:risk 2.0).
"""

from __future__ import annotations

from typing import Optional

from config import (
    Action,
    AssetClass,
    ExecutionTicket,
    MarketPhase,
    OpportunityShortlist,
    Sentiment,
    TradeIdea,
    TradeSignal,
    cap_quantity,
    logger,
    settings,
)

ATR_STOP_MULT = 1.5
REWARD_RISK = 2.0


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def _indicators(asset: str) -> Optional[dict]:
    """
    Technical snapshot from real daily candles; None if there's too little data
    even for momentum. Works on as few as 6 bars (fresh IPOs) — trend rules need
    >=25 bars (`bars` is exposed so the caller can gate on it), but the momentum
    + catalyst path runs on thin history too.
    """
    import yfinance as yf

    df = yf.Ticker(asset).history(period="3mo", interval="1d")
    if df is None or df.empty or len(df) < 6:
        return None
    close, high, low = df["Close"], df["High"], df["Low"]
    bars = len(close)
    last = float(close.iloc[-1])
    sma20 = float(close.tail(min(20, bars)).mean())
    sma50 = float(close.tail(min(50, bars)).mean())
    atr = float((high - low).tail(min(14, bars)).mean())
    hi20 = float(high.tail(min(20, bars)).max())
    lo20 = float(low.tail(min(20, bars)).min())
    if bars >= 15:
        delta = close.diff()
        gain = float(delta.clip(lower=0).tail(14).mean())
        loss = float((-delta.clip(upper=0)).tail(14).mean())
        rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    else:
        rsi = 50.0  # neutral until there's enough history
    mom = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
    stop_dist = max(ATR_STOP_MULT * atr, last * 0.005)
    return {
        "last": last, "sma20": sma20, "sma50": sma50, "atr": atr, "rsi": rsi,
        "hi20": hi20, "lo20": lo20, "mom": mom, "stop_dist": stop_dist, "bars": bars,
    }


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def rules_signal(asset: str, market_phase: str) -> TradeSignal:
    """Deterministic BUY/SELL/HOLD with entry/stop/target from candle rules."""
    ind = _indicators(asset)
    if ind is None:
        return TradeSignal(
            asset=asset, action=Action.HOLD, confidence=0.0,
            rationale="No usable candle data — stand down.", time_horizon="swing",
        )

    last, sma20, sma50, rsi, mom = ind["last"], ind["sma20"], ind["sma50"], ind["rsi"], ind["mom"]
    hi20, lo20, sd = ind["hi20"], ind["lo20"], ind["stop_dist"]

    up_trend = last > sma20 > sma50
    down_trend = last < sma20 < sma50
    breakout = last >= hi20 * 0.999
    breakdown = last <= lo20 * 1.001
    rsi_long_ok = 45.0 <= rsi <= 72.0
    rsi_short_ok = 28.0 <= rsi <= 55.0

    buy = (up_trend and rsi_long_ok) or (breakout and rsi < 75 and mom > 0)
    sell = (down_trend and rsi_short_ok) or (breakdown and rsi > 25 and mom < 0)

    if buy and not sell:
        action = Action.BUY
        entry, stop, target = last, last - sd, last + REWARD_RISK * sd
        conf = 0.5 + 0.2 * up_trend + 0.15 * breakout + 0.1 * rsi_long_ok + 0.05 * (mom > 0)
        why = (f"BUY: close {last:.2f} {'>' if up_trend else 'vs'} SMA20 {sma20:.2f} "
               f"> SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%"
               f"{', 20-bar breakout' if breakout else ''}.")
    elif sell and not buy:
        action = Action.SELL
        entry, stop, target = last, last + sd, last - REWARD_RISK * sd
        conf = 0.5 + 0.2 * down_trend + 0.15 * breakdown + 0.1 * rsi_short_ok + 0.05 * (mom < 0)
        why = (f"SELL: close {last:.2f} {'<' if down_trend else 'vs'} SMA20 {sma20:.2f} "
               f"< SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%"
               f"{', 20-bar breakdown' if breakdown else ''}.")
    else:
        return TradeSignal(
            asset=asset, action=Action.HOLD, confidence=0.3,
            rationale=(f"HOLD: no clean edge (close {last:.2f}, SMA20 {sma20:.2f}, "
                       f"SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%)."),
            time_horizon="swing",
        )

    return TradeSignal(
        asset=asset, action=action, confidence=round(min(conf, 1.0), 2),
        rationale=why, suggested_entry=round(entry, 4),
        suggested_stop=round(stop, 4), suggested_target=round(target, 4),
        time_horizon="swing",
    )


# --------------------------------------------------------------------------- #
# Ticket (signal + deterministic sizing) — drop-in for run_cycle.
# --------------------------------------------------------------------------- #
def rules_ticket(asset: str, market_phase: str) -> Optional[ExecutionTicket]:
    """Full rules cycle: signal -> risk-capped ExecutionTicket. Never None for a
    real asset (HOLD returns a stand-down ticket); None only on a build error."""
    from execution import current_equity

    signal = rules_signal(asset, market_phase)
    capital = current_equity()

    if signal.action == Action.HOLD:
        logger.info("RULES HOLD %s — %s", asset, signal.rationale)
        return ExecutionTicket(
            asset=asset, action=Action.HOLD,
            entry_price=signal.suggested_entry or 0.0,
            stop_loss=signal.suggested_stop or 0.0,
            take_profit=signal.suggested_target or 0.0,
            quantity=0.0, risk_dollars=0.0, risk_pct=0.0,
            reward_risk_ratio=0.0, capital_at_open=capital,
            rationale=signal.rationale,
        )

    entry = float(signal.suggested_entry)
    stop = float(signal.suggested_stop)
    target = float(signal.suggested_target)
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        logger.warning("RULES %s: zero stop distance — HOLD.", asset)
        return rules_ticket_hold(asset, capital)
    qty = cap_quantity(settings.MAX_RISK_DOLLARS / per_unit, entry, capital)
    risk_dollars = round(qty * per_unit, 2)
    rr = round(abs(target - entry) / per_unit, 2)
    try:
        ticket = ExecutionTicket(
            asset=asset, action=signal.action,
            entry_price=round(entry, 4), stop_loss=round(stop, 4),
            take_profit=round(target, 4), quantity=round(qty, 4),
            risk_dollars=risk_dollars,
            risk_pct=round(risk_dollars / capital, 4) if capital else 0.0,
            reward_risk_ratio=rr, capital_at_open=round(capital, 2),
            rationale=signal.rationale,
        )
    except Exception as exc:  # noqa: BLE001 — geometry/cap violation
        logger.error("RULES ticket build failed for %s: %s", asset, exc)
        return None
    logger.info("RULES TICKET | %s %s qty=%s entry=%s sl=%s tp=%s risk=$%.2f",
                ticket.action.value, asset, ticket.quantity, ticket.entry_price,
                ticket.stop_loss, ticket.take_profit, ticket.risk_dollars)
    return ticket


def rules_ticket_hold(asset: str, capital: float) -> ExecutionTicket:
    return ExecutionTicket(
        asset=asset, action=Action.HOLD, entry_price=0.0, stop_loss=0.0,
        take_profit=0.0, quantity=0.0, risk_dollars=0.0, risk_pct=0.0,
        reward_risk_ratio=0.0, capital_at_open=capital, rationale="HOLD — no edge.",
    )


# --------------------------------------------------------------------------- #
# Discovery (deterministic scan + rank) — drop-in for run_discovery.
# --------------------------------------------------------------------------- #
def rules_discovery(market_phase: str) -> OpportunityShortlist:
    """Scan live movers and rank them deterministically — no LLM ranker."""
    from scanners import fetch_crypto_movers, fetch_stock_movers

    try:
        phase = MarketPhase(market_phase)
    except ValueError:
        phase = MarketPhase.MID_DAY

    raw = (fetch_stock_movers(phase.value) or []) + (fetch_crypto_movers(phase.value) or [])
    # Rank by conviction: bigger |move| and real volume first.
    raw.sort(key=lambda d: (abs(d.get("raw_score", 0.0)), d.get("volume") or 0), reverse=True)

    ideas = []
    for d in raw[: max(settings.MAX_CANDIDATES * 2, 8)]:
        try:
            ideas.append(TradeIdea(**{k: d.get(k) for k in
                         ("asset", "asset_class", "raw_score", "change_pct",
                          "volume", "reason", "source")}))
        except Exception:  # noqa: BLE001 — skip malformed scanner rows
            continue

    logger.info("RULES SHORTLIST | phase=%s ideas=%s", phase.value,
                ", ".join(i.asset for i in ideas[: settings.MAX_CANDIDATES]) or "(none)")
    return OpportunityShortlist(
        market_phase=phase, macro_bias=Sentiment.NEUTRAL, macro_score=0.0,
        themes=[], ideas=ideas,
    )
