"""
scanners.py
===========
Universe discovery engine. Finds *candidate* trades by scanning live market
movers across two asset classes, so the desk no longer needs a human to pre-pick
a ticker:

    fetch_stock_movers(phase)  -> [TradeIdea-shaped dict]   (yfinance screeners)
    fetch_crypto_movers(phase) -> [TradeIdea-shaped dict]   (CoinGecko markets)

Design: the scanners are deliberately *cheap*. They rank purely off the screener
payload (intraday % change + volume) — no per-ticker candle fetch, no LLM. That
is the funnel's wide mouth. The expensive deep analysis (real candles, technical
confluence, risk sizing) only runs later on the handful of shortlisted names.

Every fetch degrades gracefully: if the live source is down, we fall back to a
small seed universe scored with `candles.candle_scan`, and flag the result so the
ranker agent knows the intel is thin.
"""

from __future__ import annotations

from typing import Dict, List

import requests

from config import logger, settings

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
REQUEST_TIMEOUT = 30  # seconds

# Stablecoins never make tradable movers — their |change| is ~0 by design but a
# volume sort can still surface them. Drop by symbol.
STABLE_SKIP = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD", "PYUSD"}

# Seed fallbacks — used only when the live screener/API fails. Liquid, always-on.
SEED_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "SPY", "QQQ"]
SEED_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]

# yfinance predefined screens we union for stock movers.
STOCK_SCREENS = ["day_gainers", "day_losers", "most_actives"]


def _score_from_change(change_pct: float) -> float:
    """
    Map an intraday % change to a signed -1..+1 bias score.

    A ±10% move saturates to ±1. This is a coarse pre-screen only — the desk's
    candle analysis re-derives the real technical score on the shortlist, so we
    just need a sane ranking signal here.
    """
    if change_pct is None:
        return 0.0
    return round(max(-1.0, min(1.0, change_pct / 10.0)), 3)


# --------------------------------------------------------------------------- #
# Stocks / ETFs — yfinance predefined screeners
# --------------------------------------------------------------------------- #
def fetch_stock_movers(market_phase: str, per_source: int | None = None) -> List[Dict]:
    """
    Union the day's gainers, losers, and most-active stocks/ETFs into ranked
    TradeIdea-shaped dicts. Sorted by |raw_score| descending (biggest movers
    first). Falls back to the seed universe on total screener failure.
    """
    per_source = per_source or settings.SCAN_PER_SOURCE
    try:
        import yfinance as yf

        merged: Dict[str, Dict] = {}
        for screen in STOCK_SCREENS:
            try:
                quotes = yf.screen(screen, count=per_source).get("quotes", [])
            except Exception as exc:  # noqa: BLE001 — one bad screen != whole failure
                logger.warning("stock screen '%s' failed: %s", screen, exc)
                continue
            for q in quotes:
                sym = q.get("symbol")
                if not sym:
                    continue
                change = q.get("regularMarketChangePercent")
                qtype = (q.get("quoteType") or "EQUITY").upper()
                merged[sym] = {  # dedupe by symbol; last screen wins, data is same
                    "asset": sym,
                    "asset_class": "etf" if qtype == "ETF" else "stock",
                    "raw_score": _score_from_change(change),
                    "change_pct": round(change, 2) if change is not None else None,
                    "volume": q.get("regularMarketVolume"),
                    "reason": f"{screen.replace('_', ' ')}: {round(change or 0, 2)}% intraday",
                    "source": f"yfinance:{screen}",
                }
        if merged:
            ideas = sorted(merged.values(), key=lambda d: abs(d["raw_score"]), reverse=True)
            logger.info("Stock scanner: %d unique movers from live screeners.", len(ideas))
            return ideas
        logger.warning("Stock screeners returned nothing — using seed universe.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Stock scanner failed hard: %s — using seed universe.", exc)

    return _seed_movers(SEED_STOCKS, "stock", market_phase)


# --------------------------------------------------------------------------- #
# Crypto — CoinGecko free markets endpoint
# --------------------------------------------------------------------------- #
def fetch_crypto_movers(market_phase: str, per_source: int | None = None) -> List[Dict]:
    """
    Pull top-cap coins with 24h change from CoinGecko, drop stablecoins, and
    rank by |24h change|. Symbols are mapped to yfinance tickers (`BTC-USD`) so
    the same downstream candle/fill path works. Falls back to seed crypto.
    """
    per_source = per_source or settings.SCAN_PER_SOURCE
    try:
        resp = requests.get(
            COINGECKO_URL,
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": max(per_source * 4, 40),  # over-fetch; we filter + trim
                "page": 1,
                "price_change_percentage": "24h",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        ideas: List[Dict] = []
        for c in resp.json():
            sym = (c.get("symbol") or "").upper()
            if not sym or sym in STABLE_SKIP:
                continue
            change = c.get("price_change_percentage_24h")
            ideas.append(
                {
                    "asset": f"{sym}-USD",
                    "asset_class": "crypto",
                    "raw_score": _score_from_change(change),
                    "change_pct": round(change, 2) if change is not None else None,
                    "volume": c.get("total_volume"),
                    "reason": f"24h move {round(change or 0, 2)}% on ${c.get('total_volume', 0):,} vol",
                    "source": "coingecko:markets",
                }
            )
        ideas.sort(key=lambda d: abs(d["raw_score"]), reverse=True)
        if ideas:
            logger.info("Crypto scanner: %d movers from CoinGecko.", len(ideas))
            return ideas[: per_source * 2]
        logger.warning("CoinGecko returned no usable coins — using seed crypto.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Crypto scanner failed: %s — using seed crypto.", exc)

    return _seed_movers(SEED_CRYPTO, "crypto", market_phase)


# --------------------------------------------------------------------------- #
# Degraded fallback — score a fixed seed list with real candles
# --------------------------------------------------------------------------- #
def _seed_movers(symbols: List[str], asset_class: str, market_phase: str) -> List[Dict]:
    """
    Last resort: compute a bias for each seed symbol via candle_scan so the
    pipeline still produces ranked ideas when live movers are unavailable.
    """
    from candles import candle_scan

    ideas: List[Dict] = []
    for sym in symbols:
        try:
            scan = candle_scan(sym, market_phase)
            sig = scan.get("signals", {}) or {}
            score = float(sig.get("sentiment_score", 0.0) or 0.0)
            ideas.append(
                {
                    "asset": sym,
                    "asset_class": asset_class,
                    "raw_score": round(score, 3),
                    "change_pct": sig.get("gap_pct"),
                    "volume": None,
                    "reason": f"seed fallback: candle bias {score}",
                    "source": "seed+candles",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Seed scan failed for %s: %s", sym, exc)
    ideas.sort(key=lambda d: abs(d["raw_score"]), reverse=True)
    return ideas
