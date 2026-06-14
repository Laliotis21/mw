"""
tools.py
========
Perplexity live-web scanning, exposed as a CrewAI tool.

One function does the HTTP work (`perplexity_scan`). One CrewAI `@tool` wraps it
so the Researcher agent can call it. The prompt is selected by market phase —
each phase needs different intelligence:

    pre_market -> overnight news + gap catalysts
    open       -> opening-drive order-flow / volume spikes
    mid_day    -> macro catalysts + trend continuation/reversal
    close      -> closing sentiment + positioning into the bell

Robustness: retries with backoff (tenacity), hard timeout, and a graceful
degraded payload if the API is down so the crew never hard-crashes mid-cycle.
"""

from __future__ import annotations

import json
from typing import Dict

import requests
from crewai.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import MarketPhase, logger, settings

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
REQUEST_TIMEOUT = 45  # seconds


# --------------------------------------------------------------------------- #
# Phase-specialized prompts. Each one is tuned for what actually matters at
# that point in the session — generic "give me news" wastes the model.
# --------------------------------------------------------------------------- #
def _build_prompt(asset: str, phase: MarketPhase) -> str:
    prompts: Dict[MarketPhase, str] = {
        MarketPhase.PRE_MARKET: (
            f"You are a pre-market intelligence scout. For {asset}, report ONLY "
            f"facts from the last 18 hours: overnight headlines, earnings, "
            f"analyst rating changes, pre-market gap and its cause, and any "
            f"scheduled macro event today that could move it. Cite sources. "
            f"State a one-line directional bias and a -1..+1 sentiment score."
        ),
        MarketPhase.OPEN: (
            f"You are an opening-bell order-flow scout. For {asset}, report the "
            f"opening-drive behaviour right now: unusual volume or options "
            f"activity, notable block/dark-pool prints if reported, the opening "
            f"range, and which side (buyers/sellers) is in control. Cite sources. "
            f"Give a -1..+1 sentiment score."
        ),
        MarketPhase.MID_DAY: (
            f"You are a macro catalyst scout. For {asset} at mid-session, report "
            f"any macroeconomic releases, Fed/central-bank commentary, sector "
            f"rotation, or breaking news in the last 3 hours, and whether the "
            f"intraday trend is continuing or reversing. Cite sources. Give a "
            f"-1..+1 sentiment score."
        ),
        MarketPhase.CLOSE: (
            f"You are a closing-sentiment scout. For {asset} into the close, "
            f"report end-of-day positioning: was it bought or sold into the "
            f"bell, after-hours catalysts pending, and tomorrow's setup. Cite "
            f"sources. Give a -1..+1 sentiment score."
        ),
    }
    return prompts[phase]


# --------------------------------------------------------------------------- #
# Mock research — free, deterministic, no API key. Lets us exercise the full
# agent pipeline without spending Perplexity credits. Returns plausible canned
# intel that varies by phase so the analyst has something real to chew on.
# --------------------------------------------------------------------------- #
def _mock_payload(asset: str, phase: MarketPhase) -> dict:
    by_phase = {
        MarketPhase.PRE_MARKET: (
            f"{asset} indicated up ~1.8% pre-market after an overnight earnings "
            f"beat (EPS $2.10 vs $1.95 est, revenue +9% YoY). Two analysts raised "
            f"targets. Gap-up driven by guidance, not a one-off. Macro: CPI print "
            f"due 13:30 UTC — possible volatility. Implied support 182, resistance "
            f"191. Bias: bullish. Sentiment score: +0.55."
        ),
        MarketPhase.OPEN: (
            f"{asset} opened strong on ~1.7x average volume. Opening range 184-187, "
            f"buyers controlling the tape, no large sell prints reported. Call "
            f"option volume elevated at the 190 strike. Bias: bullish but watch for "
            f"a pullback to VWAP near 185. Sentiment score: +0.40."
        ),
        MarketPhase.MID_DAY: (
            f"{asset} consolidating mid-session. CPI came in line, muted macro "
            f"reaction. Sector (tech) rotating slightly risk-off into the afternoon. "
            f"Intraday trend flattening; range 185-189. No fresh catalyst. Bias: "
            f"neutral-to-slightly-bullish. Sentiment score: +0.15."
        ),
        MarketPhase.CLOSE: (
            f"{asset} sold into the close, giving back ~0.6% from highs on profit "
            f"taking. No after-hours catalysts pending. Closing near 186, mid-range. "
            f"Tomorrow's setup neutral pending broad market. Sentiment score: -0.10."
        ),
    }
    return {
        "asset": asset,
        "market_phase": phase.value,
        "content": by_phase[phase],
        "sources": ["MOCK: deterministic test data (no live web)"],
        "degraded": False,
    }


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_perplexity(prompt: str) -> dict:
    """Single POST to Perplexity with retry/backoff on transient network errors."""
    headers = {
        "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise financial research assistant. Return "
                    "concise, source-backed facts only. Never invent prices."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,  # low — we want facts, not creativity
    }
    resp = requests.post(
        PERPLEXITY_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def perplexity_scan(asset: str, market_phase: str) -> str:
    """
    Run a phase-specific live web scan for `asset`.

    Returns a JSON string (so it survives the CrewAI tool string boundary):
        {"asset", "market_phase", "content", "sources", "degraded"}

    On total failure it returns a `degraded=True` payload instead of raising,
    so the analyst can still HOLD rather than the whole crew aborting.
    """
    try:
        phase = MarketPhase(market_phase)
    except ValueError:
        valid = ", ".join(p.value for p in MarketPhase)
        return json.dumps(
            {
                "asset": asset,
                "market_phase": market_phase,
                "content": f"Invalid market_phase. Use one of: {valid}.",
                "sources": [],
                "degraded": True,
            }
        )

    # Real candle path — yfinance + technical indicators (free, real market).
    if settings.RESEARCH_SOURCE == "candles":
        from candles import candle_scan

        logger.info("CANDLE scan | asset=%s phase=%s (real data)", asset, phase.value)
        return json.dumps(candle_scan(asset, phase.value))

    # Free mock path — skip the paid API entirely.
    if settings.RESEARCH_SOURCE == "mock":
        logger.info("MOCK scan | asset=%s phase=%s (no credits used)", asset, phase.value)
        return json.dumps(_mock_payload(asset, phase))

    settings.require("PERPLEXITY_API_KEY")
    prompt = _build_prompt(asset, phase)
    logger.info("Perplexity scan | asset=%s phase=%s", asset, phase.value)

    try:
        data = _call_perplexity(prompt)
        content = data["choices"][0]["message"]["content"]
        # Perplexity returns citations top-level when available.
        sources = data.get("citations", []) or data.get("search_results", [])
        return json.dumps(
            {
                "asset": asset,
                "market_phase": phase.value,
                "content": content,
                "sources": sources,
                "degraded": False,
            }
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, log loudly
        logger.error("Perplexity scan failed for %s: %s", asset, exc)
        return json.dumps(
            {
                "asset": asset,
                "market_phase": phase.value,
                "content": (
                    "RESEARCH UNAVAILABLE — live scan failed. Treat as no edge; "
                    "default to HOLD unless technicals are decisive."
                ),
                "sources": [],
                "degraded": True,
            }
        )


# --------------------------------------------------------------------------- #
# World / macro scan — broad backdrop, NOT a single asset. Drives the macro tilt
# the Opportunity Ranker uses to weight candidate ideas.
# --------------------------------------------------------------------------- #
def _world_prompt(phase: MarketPhase) -> str:
    when = {
        MarketPhase.PRE_MARKET: "overnight and pre-open",
        MarketPhase.OPEN: "right now at the opening bell",
        MarketPhase.MID_DAY: "this afternoon mid-session",
        MarketPhase.CLOSE: "into the closing bell",
    }[phase]
    return (
        f"You are a global macro desk scout. Report ONLY material, source-backed "
        f"developments {when} that move broad risk assets: index futures (ES/NQ), "
        f"major macro data (CPI/jobs/Fed), central-bank action, geopolitics, oil, "
        f"the dollar (DXY), and crypto-wide moves. List the 3-5 dominant themes. "
        f"End with an overall risk tilt: a one-word bias (bullish/bearish/neutral) "
        f"and a -1..+1 risk score (+1 full risk-on, -1 full risk-off). Cite sources."
    )


def world_scan(market_phase: str) -> str:
    """
    Broad world/macro intelligence for the current phase (no single asset).

    Returns a JSON string: {market_phase, content, sources, degraded}. Live web
    only makes sense with Perplexity; mock/candles sources return a neutral,
    degraded backdrop so the pipeline still runs offline.
    """
    try:
        phase = MarketPhase(market_phase)
    except ValueError:
        phase = MarketPhase.MID_DAY

    if settings.RESEARCH_SOURCE in ("mock", "candles"):
        logger.info("World scan | phase=%s (offline backdrop)", phase.value)
        return json.dumps(
            {
                "market_phase": phase.value,
                "content": (
                    "OFFLINE BACKDROP — no live macro feed in this mode. Assume a "
                    "neutral, range-bound tape with no dominant catalyst. Risk tilt "
                    "neutral, score 0.0. Weight ideas on their own technicals."
                ),
                "sources": ["offline: no live macro source"],
                "degraded": True,
            }
        )

    settings.require("PERPLEXITY_API_KEY")
    logger.info("World/macro scan | phase=%s", phase.value)
    try:
        data = _call_perplexity(_world_prompt(phase))
        content = data["choices"][0]["message"]["content"]
        sources = data.get("citations", []) or data.get("search_results", [])
        return json.dumps(
            {
                "market_phase": phase.value,
                "content": content,
                "sources": sources,
                "degraded": False,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("World scan failed: %s", exc)
        return json.dumps(
            {
                "market_phase": phase.value,
                "content": "MACRO UNAVAILABLE — treat backdrop as neutral, score 0.0.",
                "sources": [],
                "degraded": True,
            }
        )


# --------------------------------------------------------------------------- #
# CrewAI tool wrappers — what the agents actually invoke.
# --------------------------------------------------------------------------- #
@tool("Perplexity Market Scanner")
def market_scanner_tool(asset: str, market_phase: str) -> str:
    """
    Scan live financial web intelligence for a given asset and market phase.

    Args:
        asset: Ticker or instrument, e.g. 'AAPL', 'BTC-USD'.
        market_phase: One of 'pre_market', 'open', 'mid_day', 'close'.

    Returns:
        JSON string with keys: asset, market_phase, content, sources, degraded.
    """
    return perplexity_scan(asset=asset, market_phase=market_phase)


@tool("World Macro Scanner")
def world_events_tool(market_phase: str) -> str:
    """
    Scan broad world/macro developments for the current market phase (no single
    asset). Use to read the global risk backdrop driving all markets.

    Args:
        market_phase: One of 'pre_market', 'open', 'mid_day', 'close'.

    Returns:
        JSON string with keys: market_phase, content, sources, degraded.
    """
    return world_scan(market_phase=market_phase)


@tool("Stock Movers Scanner")
def stock_scanner_tool(market_phase: str) -> str:
    """
    Scan the live stock/ETF universe for today's biggest movers (gainers,
    losers, most active). Cheap pre-screen — ranks by intraday move, no deep
    analysis.

    Args:
        market_phase: One of 'pre_market', 'open', 'mid_day', 'close'.

    Returns:
        JSON string: a list of candidate dicts, each with asset, asset_class,
        raw_score, change_pct, volume, reason, source.
    """
    from scanners import fetch_stock_movers

    return json.dumps(fetch_stock_movers(market_phase))


@tool("Crypto Movers Scanner")
def crypto_scanner_tool(market_phase: str) -> str:
    """
    Scan the live crypto universe for the biggest 24h movers by absolute price
    change (stablecoins excluded). Symbols come back as yfinance tickers
    (e.g. 'BTC-USD').

    Args:
        market_phase: One of 'pre_market', 'open', 'mid_day', 'close'.

    Returns:
        JSON string: a list of candidate dicts, each with asset, asset_class,
        raw_score, change_pct, volume, reason, source.
    """
    from scanners import fetch_crypto_movers

    return json.dumps(fetch_crypto_movers(market_phase))
