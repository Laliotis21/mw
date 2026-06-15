"""
news.py
=======
Free news/catalyst signal from the Alpaca News API (uses the Alpaca keys you
already have — no Perplexity, no extra cost). Turns recent headlines into a
sentiment score (-1..+1) plus a "catalyst" flag the rules engine uses to
confirm/veto technical trades and to catch fresh-listing momentum (e.g. an IPO
ripping 150 -> 200 with no SMA history yet).

Sentiment scoring (NEWS_SENTIMENT):
    keyword (default) — finance word lists, recency-weighted. Free, instant.
    llm               — one cheap Haiku call over the headlines (~$0.001/ticker),
                        only when an Anthropic key is set. More nuanced.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests

from config import logger, settings

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
REQUEST_TIMEOUT = 15

# Catalyst thresholds.
CATALYST_MIN_HEADLINES = 2
CATALYST_MIN_SCORE = 0.4
CATALYST_FRESH_HOURS = 48

_POS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally",
    "rallies", "upgrade", "upgraded", "raises", "raised", "record", "profit",
    "growth", "wins", "win", "awarded", "approval", "approved", "breakthrough",
    "partnership", "acquire", "acquisition", "outperform", "bullish", "gain",
    "gains", "tops", "exceeds", "strong", "soaring", "rockets", "skyrocket",
    "buyback", "beats expectations", "all-time high",
}
_NEG = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "slump", "downgrade", "downgraded", "cuts", "cut", "lawsuit", "probe",
    "investigation", "fraud", "recall", "halt", "halts", "bankruptcy", "layoffs",
    "warning", "warns", "weak", "loss", "losses", "decline", "sinks", "tumble",
    "tumbles", "bearish", "default", "delays", "delay", "scandal", "slashes",
    "plummet", "plummets", "sell-off", "selloff",
}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_news(symbol: str, limit: int = 10) -> list[dict]:
    """Recent Alpaca headlines for one symbol. [] on any failure (degrade safe)."""
    if not (settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY):
        return []
    headers = {
        "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
    }
    # Alpaca news symbols are bare equities ('TSLA'); strip our crypto suffix.
    sym = symbol.upper().replace("-USD", "").replace("-USDT", "").replace("USDT", "")
    try:
        r = requests.get(ALPACA_NEWS_URL, headers=headers,
                         params={"symbols": sym, "limit": limit}, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            logger.warning("Alpaca news %s: %s %s", sym, r.status_code, r.text[:120])
            return []
        return r.json().get("news", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alpaca news fetch failed for %s: %s", sym, exc)
        return []


def _clean(items: list[dict]) -> list[dict]:
    """Precision filter: drop stale (>72h), generic round-ups (>5 symbols), and
    duplicate headlines — so the score reflects real, relevant, fresh news."""
    out, seen = [], set()
    now = datetime.now(timezone.utc)
    for i in items:
        head = (i.get("headline") or "").strip()
        if not head or head.lower() in seen:
            continue
        if len(i.get("symbols") or []) > 5:  # market round-up, not asset-specific
            continue
        ts = i.get("created_at")
        if ts:
            try:
                age = (now - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
                if age > 72:
                    continue
            except Exception:  # noqa: BLE001
                pass
        seen.add(head.lower())
        out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
def _keyword_sentiment(texts: list[str]) -> float:
    """(-1..+1) from finance word hits across headlines."""
    pos = neg = 0
    for t in texts:
        low = f" {t.lower()} "
        pos += sum(1 for w in _POS if w in low)
        neg += sum(1 for w in _NEG if w in low)
    total = pos + neg
    return round((pos - neg) / total, 3) if total else 0.0


def _llm_sentiment(symbol: str, texts: list[str]) -> Optional[float]:
    """One cheap Anthropic call -> (-1..+1). None if unavailable (caller falls back)."""
    if not settings.ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        joined = "\n".join(f"- {t}" for t in texts[:10])
        msg = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=16,
            messages=[{
                "role": "user",
                "content": (
                    f"Headlines for {symbol}:\n{joined}\n\n"
                    "Reply with ONLY a number from -1 (very bearish) to 1 (very "
                    "bullish) summarizing the net trading sentiment."
                ),
            }],
        )
        # Roll this call's cost into the trade-log ledger so news sentiment spend
        # isn't invisible (the dashboard API-spend total must stay honest).
        try:
            from execution import log_usage
            u = msg.usage
            log_usage({
                "model": settings.ANTHROPIC_MODEL,
                "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
                "cached_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
                "requests": 1,
            })
        except Exception:  # noqa: BLE001 — never let accounting break a signal
            pass
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return max(-1.0, min(1.0, float(txt.strip().split()[0])))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM news sentiment failed for %s: %s — keyword fallback.", symbol, exc)
        return None


# --------------------------------------------------------------------------- #
# Public: combined news signal
# --------------------------------------------------------------------------- #
def news_signal(symbol: str) -> dict:
    """
    {score (-1..+1), n, catalyst (bool), fresh_hours, top, source}. A 'catalyst'
    is fresh (<48h), with enough headlines and a decisive score — what flags an
    IPO/news-driven momentum move the technicals can't see yet.
    """
    items = fetch_news(symbol)
    items = _clean(items)
    if not items:
        return {"score": 0.0, "n": 0, "catalyst": False, "fresh_hours": None,
                "top": "", "source": "none"}

    texts = [f"{i.get('headline','')} {i.get('summary','')}".strip() for i in items]
    if settings.NEWS_SENTIMENT == "llm":
        score = _llm_sentiment(symbol, texts)
        source = "alpaca+llm"
        if score is None:
            score, source = _keyword_sentiment(texts), "alpaca+keyword"
    else:
        score, source = _keyword_sentiment(texts), "alpaca+keyword"

    # Freshness of the most recent headline.
    fresh_hours = None
    try:
        newest = max(
            datetime.fromisoformat(i["created_at"].replace("Z", "+00:00"))
            for i in items if i.get("created_at")
        )
        fresh_hours = round((datetime.now(timezone.utc) - newest).total_seconds() / 3600, 1)
    except Exception:  # noqa: BLE001
        pass

    catalyst = (
        len(items) >= CATALYST_MIN_HEADLINES
        and abs(score) >= CATALYST_MIN_SCORE
        and (fresh_hours is not None and fresh_hours <= CATALYST_FRESH_HOURS)
    )
    return {
        "score": round(score, 3), "n": len(items), "catalyst": catalyst,
        "fresh_hours": fresh_hours, "top": items[0].get("headline", "")[:90],
        "source": source,
    }
