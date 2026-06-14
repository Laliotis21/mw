"""
candles.py
==========
Real market-data + technical-analysis engine. Pulls candles from yfinance and
computes the signal set described in specs/signals.md, so the Researcher agent
reads the ACTUAL market instead of mock text.

Public entry point:
    candle_scan(asset, market_phase) -> dict
      {asset, market_phase, interval, content (human summary), signals (dict),
       sources, degraded}

No extra dependencies — indicators are computed with pandas/numpy that ship with
yfinance. FX tickers (`*=X`) usually have no volume; we detect that and skip the
volume/VWAP signals rather than emit garbage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import MarketPhase, logger

# Phase -> (yfinance period, interval). Faster intervals near the open/close
# where intraday structure matters; wider window mid-session and pre-market.
PHASE_DATA = {
    MarketPhase.PRE_MARKET: ("5d", "15m"),
    MarketPhase.OPEN: ("1d", "5m"),
    MarketPhase.MID_DAY: ("5d", "15m"),
    MarketPhase.CLOSE: ("1d", "5m"),
}


def phase_data(market_phase: str) -> tuple[str, str]:
    """(period, interval) for a phase — shared by research scan and fill sim."""
    try:
        phase = MarketPhase(market_phase)
    except ValueError:
        phase = MarketPhase.MID_DAY
    return PHASE_DATA[phase]


# --------------------------------------------------------------------------- #
# Indicator helpers
# --------------------------------------------------------------------------- #
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _vwap_last_day(df: pd.DataFrame) -> Optional[float]:
    """VWAP over the most recent calendar day's bars (intraday convention)."""
    if df["Volume"].sum() == 0:
        return None  # FX / no-volume instrument
    last_day = df.index[-1].date()
    day = df[df.index.map(lambda ts: ts.date() == last_day)]
    if day.empty or day["Volume"].sum() == 0:
        return None
    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    return float((typical * day["Volume"]).sum() / day["Volume"].sum())


def _candle_pattern(df: pd.DataFrame) -> str:
    """Cheap last-bar pattern read: engulfing / hammer / shooting star / none."""
    if len(df) < 2:
        return "none"
    o, h, l, c = (float(df[col].iloc[-1]) for col in ("Open", "High", "Low", "Close"))
    po, pc = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if c > o and pc < po and c >= po and o <= pc:
        return "bullish engulfing"
    if c < o and pc > po and c <= po and o >= pc:
        return "bearish engulfing"
    if lower_wick > 2 * body and upper_wick < body:
        return "hammer (bullish)"
    if upper_wick > 2 * body and lower_wick < body:
        return "shooting star (bearish)"
    return "none"


# --------------------------------------------------------------------------- #
# Main scan
# --------------------------------------------------------------------------- #
def candle_scan(asset: str, market_phase: str) -> dict:
    """Fetch candles + compute signals for the asset at the given phase."""
    try:
        phase = MarketPhase(market_phase)
    except ValueError:
        phase = MarketPhase.MID_DAY

    period, interval = PHASE_DATA[phase]

    try:
        import yfinance as yf

        df = yf.Ticker(asset).history(period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        logger.error("candle_scan fetch failed for %s: %s", asset, exc)
        df = None

    if df is None or df.empty or len(df) < 25:
        return {
            "asset": asset,
            "market_phase": phase.value,
            "interval": interval,
            "content": (
                f"DATA UNAVAILABLE for {asset} ({interval}). No reliable candles. "
                f"Treat as no edge → HOLD."
            ),
            "signals": {},
            "sources": [f"yfinance:{asset}:{interval}"],
            "degraded": True,
        }

    close = df["Close"]
    last = float(close.iloc[-1])

    ema9 = float(_ema(close, 9).iloc[-1])
    ema21 = float(_ema(close, 21).iloc[-1])
    ema9_prev = float(_ema(close, 9).iloc[-2])
    ema21_prev = float(_ema(close, 21).iloc[-2])
    rsi = float(_rsi(close).iloc[-1])
    vwap = _vwap_last_day(df)

    # Recent structure (last ~20 bars) for support/resistance.
    window = df.tail(20)
    support = float(window["Low"].min())
    resistance = float(window["High"].max())

    # Volume conviction.
    has_volume = df["Volume"].sum() > 0
    if has_volume:
        avg_vol = float(df["Volume"].tail(20).mean())
        last_vol = float(df["Volume"].iloc[-1])
        vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0.0
    else:
        vol_ratio = None

    # Gap vs prior session close.
    last_day = df.index[-1].date()
    prior = df[df.index.map(lambda ts: ts.date() != last_day)]
    today = df[df.index.map(lambda ts: ts.date() == last_day)]
    if not prior.empty and not today.empty:
        prev_close = float(prior["Close"].iloc[-1])
        today_open = float(today["Open"].iloc[0])
        gap_pct = round((today_open - prev_close) / prev_close * 100, 2)
    else:
        gap_pct = None

    pattern = _candle_pattern(df)

    # --- Derive a bullish/bearish bias score in -1..+1 from confluence. ---
    score = 0.0
    trend_up = ema9 > ema21
    score += 0.25 if trend_up else -0.25
    if ema9_prev <= ema21_prev and ema9 > ema21:
        score += 0.2  # fresh bullish cross
    if ema9_prev >= ema21_prev and ema9 < ema21:
        score -= 0.2  # fresh bearish cross
    if rsi >= 55:
        score += 0.15
    elif rsi <= 45:
        score -= 0.15
    if vwap is not None:
        score += 0.15 if last > vwap else -0.15
    if "bullish" in pattern or "hammer" in pattern:
        score += 0.15
    elif "bearish" in pattern or "shooting" in pattern:
        score -= 0.15
    score = round(max(-1.0, min(1.0, score)), 2)
    sentiment = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"

    signals = {
        "last_price": round(last, 4),
        "ema9": round(ema9, 4),
        "ema21": round(ema21, 4),
        "trend": "up" if trend_up else "down",
        "rsi14": round(rsi, 1),
        "vwap": round(vwap, 4) if vwap is not None else None,
        "price_vs_vwap": (None if vwap is None else ("above" if last > vwap else "below")),
        "volume_x_avg": vol_ratio,
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "gap_pct": gap_pct,
        "candle_pattern": pattern,
        "sentiment_score": score,
    }

    # Human-readable brief the LLM reads directly.
    vol_txt = f"{vol_ratio}x avg" if vol_ratio is not None else "n/a (no volume)"
    vwap_txt = (
        f"{signals['price_vs_vwap']} VWAP ({signals['vwap']})"
        if vwap is not None
        else "VWAP n/a"
    )
    gap_txt = f"{gap_pct:+}% gap" if gap_pct is not None else "no gap data"
    content = (
        f"{asset} [{interval}] @ {round(last, 4)} | phase {phase.value}\n"
        f"Trend: EMA9 {round(ema9, 4)} {'>' if trend_up else '<'} EMA21 "
        f"{round(ema21, 4)} ({signals['trend']}). "
        f"RSI(14) {round(rsi, 1)}. {vwap_txt}. Volume {vol_txt}. {gap_txt}.\n"
        f"Support {round(support, 4)} | Resistance {round(resistance, 4)}. "
        f"Last candle: {pattern}.\n"
        f"Bias: {sentiment} (score {score})."
    )

    return {
        "asset": asset,
        "market_phase": phase.value,
        "interval": interval,
        "content": content,
        "signals": signals,
        "sources": [f"yfinance:{asset}:{interval}:{period}"],
        "degraded": False,
    }
