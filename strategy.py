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

import numpy as np
import pandas as pd

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
PARTIAL_FRACTION = 0.5  # fraction banked at +1R under the 'be_partial' exit


# --------------------------------------------------------------------------- #
# Exit simulation — shared by the paper fill (execution) and the backtest, so
# both measure the SAME exit logic. Pure: walks forward bars, returns the
# realized R-multiple (in units of initial risk) plus a coarse result label.
# --------------------------------------------------------------------------- #
def simulate_exit(
    highs, lows, closes, entry: float, stop: float, target: float,
    is_long: bool, style: str = "target",
) -> tuple[str, float, int]:
    """Walk the forward bars and resolve the trade; return (result, R, exit_idx).

    `highs/lows/closes` are the bars AFTER entry (any sequence). R is the realized
    profit in units of the initial stop distance (a −1R loss, +2R target win).
    `exit_idx` is the 0-based index of the bar that closed the trade (len-1 on a
    markout) so a caller can resume scanning right after it.

    style:
      'target'     — original behaviour: first of stop / target hits; if a single
                     bar straddles both, the STOP is assumed first (conservative);
                     if neither is touched, mark to the last close.
      'be_partial' — bank PARTIAL_FRACTION at +1R and move the stop to breakeven;
                     the remainder runs to the 2R target (stop = breakeven).
                     Same conservative stop-first rule on straddling bars.
    """
    risk = abs(entry - stop)
    n = len(closes)
    if risk <= 0 or n == 0:
        return ("markout", 0.0, max(n - 1, 0))
    last = n - 1

    def r_of(price: float) -> float:  # signed R at a price for this direction
        return (price - entry) / risk if is_long else (entry - price) / risk

    if style != "be_partial":
        for i in range(n):
            hi, lo = float(highs[i]), float(lows[i])
            hit_stop = lo <= stop if is_long else hi >= stop
            hit_tp = hi >= target if is_long else lo <= target
            if hit_stop:
                return ("stop_loss", -1.0, i)
            if hit_tp:
                return ("take_profit", r_of(target), i)
        return ("markout", r_of(float(closes[last])), last)

    # be_partial: half off at +1R, stop to breakeven, remainder to target.
    one_r = entry + risk if is_long else entry - risk
    stop_px, remaining, banked_r, partial_done = stop, 1.0, 0.0, False
    for i in range(n):
        hi, lo = float(highs[i]), float(lows[i])
        # Conservative: the active stop is checked before any up-level this bar.
        hit_stop = lo <= stop_px if is_long else hi >= stop_px
        if hit_stop:
            return _label(banked_r + r_of(stop_px) * remaining, i)
        reached_1r = (hi >= one_r) if is_long else (lo <= one_r)
        if not partial_done and reached_1r:
            banked_r += 1.0 * PARTIAL_FRACTION
            remaining -= PARTIAL_FRACTION
            partial_done, stop_px = True, entry  # breakeven on the rest
        hit_tp = hi >= target if is_long else lo <= target
        if hit_tp:
            return _label(banked_r + r_of(target) * remaining, i)
    return _label(banked_r + r_of(float(closes[last])) * remaining, last)


def _label(total_r: float, idx: int) -> tuple[str, float, int]:
    """Map a blended R outcome to a coarse result label for the trade log."""
    total_r = round(total_r, 4)
    if total_r > 0.01:
        return ("take_profit", total_r, idx)
    if total_r < -0.01:
        return ("stop_loss", total_r, idx)
    return ("markout", total_r, idx)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def _wilder_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI with Wilder's smoothing (the textbook definition) — the latest value.

    A plain rolling mean over `period` reacts too sharply; Wilder smooths gains
    and losses with an EMA of alpha = 1/period, which is what every charting
    package reports, so our 45/72 thresholds line up with what a trader sees.
    Returns a neutral 50.0 when there isn't enough history.
    """
    delta = close.diff().dropna()
    if len(delta) < period:
        return 50.0
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = float(gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    avg_loss = float(loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def efficiency_ratio(values, period: int = 20) -> float:
    """Kaufman Efficiency Ratio over the last `period` bars: net directional move
    divided by the total path travelled. 1.0 = a perfectly clean trend, ~0 = pure
    chop. Used to gate out directionless, whipsaw-prone names. Accepts any
    sequence (list / np.ndarray / pd.Series); returns 0.0 on thin history.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < period + 1:
        return 0.0
    seg = arr[-(period + 1):]
    net = abs(seg[-1] - seg[0])
    noise = float(np.abs(np.diff(seg)).sum())
    return float(net / noise) if noise > 0 else 0.0


def _true_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range (latest), the real definition that accounts for gaps.

    True range = max(H-L, |H-prevClose|, |L-prevClose|), so an overnight gap
    widens the stop instead of being ignored. A naive high-low mean under-sizes
    stops on gappy names and triggers needless stop-outs.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr = tr.dropna()
    if tr.empty:
        return float((high - low).tail(min(period, len(high))).mean())
    return float(tr.tail(min(period, len(tr))).mean())


def _indicators(asset: str) -> Optional[dict]:
    """
    Technical snapshot from real daily candles; None if there's too little data
    even for momentum. Works on as few as 6 bars (fresh IPOs) — trend rules need
    >=25 bars (`bars` is exposed so the caller can gate on it), but the momentum
    + catalyst path runs on thin history too.
    """
    from candles import fetch_history

    df = fetch_history(asset, "3mo", "1d")
    if df is None or df.empty or len(df) < 6:
        return None
    close, high, low = df["Close"], df["High"], df["Low"]
    bars = len(close)
    last = float(close.iloc[-1])
    sma20 = float(close.tail(min(20, bars)).mean())
    sma50 = float(close.tail(min(50, bars)).mean())
    atr = _true_atr(high, low, close, period=min(14, bars))
    hi20 = float(high.tail(min(20, bars)).max())
    lo20 = float(low.tail(min(20, bars)).min())
    # Wilder RSI needs ~15 bars to be meaningful; neutral 50 until then.
    rsi = _wilder_rsi(close, period=14) if bars >= 15 else 50.0
    mom = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
    stop_dist = max(ATR_STOP_MULT * atr, last * 0.005)
    vol = df["Volume"]
    avg_vol = float(vol.tail(min(20, bars)).mean())
    vol_ok = avg_vol <= 0 or float(vol.iloc[-1]) >= 0.7 * avg_vol  # participation
    er = efficiency_ratio(close, period=min(settings.TREND_ER_WINDOW, bars - 1))
    # SMA50 slope for trend alignment: compare the current SMA50 to where it sat
    # ~10 bars ago. None on thin history (we don't gate what we can't measure).
    sma50_rising = float(close.iloc[-60:-10].mean()) < sma50 if bars >= 60 else None
    return {
        "last": last, "sma20": sma20, "sma50": sma50, "atr": atr, "rsi": rsi,
        "hi20": hi20, "lo20": lo20, "mom": mom, "stop_dist": stop_dist,
        "bars": bars, "vol_ok": vol_ok, "er": er, "sma50_rising": sma50_rising,
    }


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def _news(asset: str) -> dict:
    """News/catalyst signal, or a neutral stub when the news layer is off."""
    if not settings.USE_NEWS:
        return {"score": 0.0, "n": 0, "catalyst": False, "top": "", "fresh_hours": None}
    from news import news_signal
    return news_signal(asset)


def _mk(asset, action, entry, stop, sd, conf, why, news) -> TradeSignal:
    """Assemble a directional TradeSignal, folding the news note into rationale."""
    target = entry + REWARD_RISK * sd if action == Action.BUY else entry - REWARD_RISK * sd
    note = ""
    if news.get("n"):
        note = f" | news {news['score']:+.2f} ({news['n']}h{'·CATALYST' if news.get('catalyst') else ''}): {news.get('top','')}"
    return TradeSignal(
        asset=asset, action=action, confidence=round(min(max(conf, 0.0), 1.0), 2),
        rationale=why + note, suggested_entry=round(entry, 4),
        suggested_stop=round(stop, 4), suggested_target=round(target, 4),
        time_horizon="swing",
    )


def rules_signal(asset: str, market_phase: str) -> TradeSignal:
    """
    Deterministic BUY/SELL/HOLD from candle rules, with a news/catalyst layer:
    news confirms or vetoes a technical trade, and a fresh strong catalyst can
    trigger a momentum entry on thin history (new listings the SMAs can't see).
    """
    ind = _indicators(asset)
    if ind is None:
        # No candle data → can't size anyway. Skip the news call (saves a fetch/
        # LLM hit on dead tickers).
        return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.0,
                           rationale="No usable candle data — stand down.",
                           time_horizon="swing")

    news = _news(asset)
    ns, catalyst = news.get("score", 0.0), news.get("catalyst", False)

    last, sma20, sma50, rsi = ind["last"], ind["sma20"], ind["sma50"], ind["rsi"]
    hi20, lo20, mom, bars = ind["hi20"], ind["lo20"], ind["mom"], ind["bars"]
    sd = ind["stop_dist"]
    sd_cat = max(sd, last * 0.02)  # wider stop for volatile catalyst moves
    # Spot crypto can't be shorted — treat crypto as long-only (no SELL signals).
    crypto = asset.upper().endswith(("-USD", "-USDT", "USDT"))

    # Catalyst-momentum: fresh strong news aligned with price — works on any
    # history depth, so it catches IPOs/news plays before the SMAs are valid.
    if catalyst and ns >= 0.5 and mom >= 1.0:
        why = f"BUY (catalyst momentum): fresh news {ns:+.2f}, mom {mom:+.1f}%, close {last:.2f}, {bars} bars"
        return _mk(asset, Action.BUY, last, last - sd_cat, sd_cat, 0.6 + 0.3 * ns, why, news)
    if catalyst and ns <= -0.5 and mom <= -1.0 and not crypto:
        why = f"SELL (catalyst momentum): fresh news {ns:+.2f}, mom {mom:+.1f}%, close {last:.2f}, {bars} bars"
        return _mk(asset, Action.SELL, last, last + sd_cat, sd_cat, 0.6 + 0.3 * abs(ns), why, news)

    # Thin history: trend rules unreliable; without a catalyst we stand down.
    if bars < 25:
        return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.2,
                           rationale=(f"HOLD: only {bars} bars (too thin for trend rules) "
                                      f"and no catalyst. news {ns:+.2f}."),
                           time_horizon="swing")

    up_trend = last > sma20 > sma50
    down_trend = last < sma20 < sma50
    breakout = last >= hi20 * 0.999
    breakdown = last <= lo20 * 1.001
    rsi_long_ok = 45.0 <= rsi <= 72.0
    rsi_short_ok = 28.0 <= rsi <= 55.0
    # Entry confirmation: need real volume (participation) and don't chase a
    # price already stretched >8% above SMA20.
    vol_ok = ind.get("vol_ok", True)
    extended = last > sma20 * 1.08
    buy = ((up_trend and rsi_long_ok) or (breakout and rsi < 75 and mom > 0)) and vol_ok and not extended
    sell = ((down_trend and rsi_short_ok) or (breakdown and rsi > 25 and mom < 0)) and vol_ok

    # Trend-alignment gate (default): don't fight the primary trend. Drop a long
    # when the SMA50 is falling and a short when it's rising — that's where the
    # strategy gets run over (counter-trend whipsaws). A live catalyst already
    # passed above, so news plays aren't gated. None slope = thin history, skip.
    rising = ind.get("sma50_rising")
    if settings.TREND_FILTER and rising is not None:
        if buy and not rising:
            return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.25,
                               rationale=(f"HOLD: BUY against a falling SMA50 (counter-trend) "
                                          f"— stand down. close {last:.2f}, SMA50 {sma50:.2f}."),
                               time_horizon="swing")
        if sell and rising:
            return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.25,
                               rationale=(f"HOLD: SELL against a rising SMA50 (counter-trend) "
                                          f"— stand down. close {last:.2f}, SMA50 {sma50:.2f}."),
                               time_horizon="swing")

    # Optional chop gate (opt-in via TREND_MIN_ER>0): refuse trades in choppy,
    # directionless price (low Efficiency Ratio). Off by default — backtests show
    # it removes profitable moderate-trend trades on a trending basket.
    er = ind.get("er", 1.0)
    if settings.TREND_MIN_ER > 0 and (buy or sell) and er < settings.TREND_MIN_ER:
        return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.25,
                           rationale=(f"HOLD: choppy regime (ER {er:.2f} < {settings.TREND_MIN_ER:.2f}) "
                                      f"— no clean trend to trade. close {last:.2f}, RSI {rsi:.0f}."),
                           time_horizon="swing")

    if crypto and sell and not buy:  # spot crypto is long-only — never short
        return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.3,
                           rationale=(f"HOLD: bearish technicals on {asset} but spot crypto "
                                      f"is long-only (no short). close {last:.2f}, RSI {rsi:.0f}."),
                           time_horizon="swing")

    if buy and not sell:
        if ns <= -0.4:  # bad fresh news on a long — veto
            return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.3,
                               rationale=(f"HOLD: technical BUY vetoed by negative news "
                                          f"{ns:+.2f}: {news.get('top','')}"), time_horizon="swing")
        conf = 0.5 + 0.2 * up_trend + 0.15 * breakout + 0.1 * rsi_long_ok + 0.1 * max(ns, 0)
        why = (f"BUY: close {last:.2f} {'>' if up_trend else 'vs'} SMA20 {sma20:.2f} > "
               f"SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%"
               f"{', 20-bar breakout' if breakout else ''}")
        return _mk(asset, Action.BUY, last, last - sd, sd, conf, why, news)

    if sell and not buy:
        if ns >= 0.4:  # good fresh news against a short — veto
            return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.3,
                               rationale=(f"HOLD: technical SELL vetoed by positive news "
                                          f"{ns:+.2f}: {news.get('top','')}"), time_horizon="swing")
        conf = 0.5 + 0.2 * down_trend + 0.15 * breakdown + 0.1 * rsi_short_ok + 0.1 * max(-ns, 0)
        why = (f"SELL: close {last:.2f} {'<' if down_trend else 'vs'} SMA20 {sma20:.2f} < "
               f"SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%"
               f"{', 20-bar breakdown' if breakdown else ''}")
        return _mk(asset, Action.SELL, last, last + sd, sd, conf, why, news)

    return TradeSignal(asset=asset, action=Action.HOLD, confidence=0.3,
                       rationale=(f"HOLD: no clean edge (close {last:.2f}, SMA20 {sma20:.2f}, "
                                  f"SMA50 {sma50:.2f}, RSI {rsi:.0f}, mom {mom:+.1f}%, "
                                  f"news {ns:+.2f})."), time_horizon="swing")


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
    # Confidence-weighted sizing: scale 0.5x–1.0x of the per-trade cap by signal
    # confidence (smaller bets on weak setups). The cap itself is 2% of CURRENT
    # equity (settings.risk_budget), so the book compounds — winners enlarge the
    # next bet, drawdowns shrink it.
    risk_budget = settings.risk_budget(capital) * min(1.0, 0.5 + 0.5 * signal.confidence)
    qty = cap_quantity(risk_budget / per_unit, entry, capital)
    # Affordability: a stock that can't make 1 whole share within budget can't be
    # a real Alpaca bracket — HOLD instead of faking a sim fill.
    crypto = asset.upper().endswith(("-USD", "-USDT", "USDT"))
    if not crypto and settings.FILL_SOURCE in ("alpaca", "live") and int(qty) < 1:
        logger.info("RULES %s: 1 share risks > budget on a $%.0f book — HOLD.", asset, capital)
        return rules_ticket_hold(asset, capital)
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

    key = lambda d: (abs(d.get("raw_score", 0.0)), d.get("volume") or 0)  # noqa: E731
    stocks = sorted(fetch_stock_movers(phase.value) or [], key=key, reverse=True)
    crypto = sorted(fetch_crypto_movers(phase.value) or [], key=key, reverse=True)

    # Balance the shortlist: crypto has bigger % swings and would crowd out
    # stocks on a pure |move| sort, yet stocks trade real on Alpaca and can be
    # shorted. Round-robin stock,crypto,... (stocks first) to guarantee both.
    n = max(settings.MAX_CANDIDATES * 2, 8)
    balanced: list[dict] = []
    si = ci = 0
    while len(balanced) < n and (si < len(stocks) or ci < len(crypto)):
        if si < len(stocks):
            balanced.append(stocks[si]); si += 1
        if ci < len(crypto) and len(balanced) < n:
            balanced.append(crypto[ci]); ci += 1

    ideas = []
    for d in balanced:
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
