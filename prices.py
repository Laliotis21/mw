"""
prices.py
=========
Background price poller. A single daemon thread keeps a live price map fresh
(~2s) for whatever symbols are currently open, using FAST batched endpoints:

    stocks  -> Alpaca latest-trade  (one request for all symbols)
    crypto  -> Binance ticker price  (one request for all symbols)

The dashboard reads the map INSTANTLY (no network in the render path), so the
auto-refresh fragment never blocks/dims and prices feel live. Decouples
price-fetch latency from Streamlit reruns.

Usage:
    import prices
    prices.watch(["QBTS", "BTC-USD"])   # set the symbols to track (starts thread)
    prices.get("QBTS")                   # latest price (or None), instant
    prices.get_all()                     # {sym: price}, instant
"""

from __future__ import annotations

import threading
import time

import requests

from config import logger, settings

POLL_SEC = max(1, int(getattr(settings, "PRICE_POLL_SEC", 2)))
REQUEST_TIMEOUT = 8

_LOCK = threading.Lock()
_PRICES: dict[str, float] = {}
_SYMBOLS: set[str] = set()
_THREAD: threading.Thread | None = None


def _is_crypto(sym: str) -> bool:
    s = sym.upper()
    return s.endswith("-USD") or s.endswith("-USDT") or s.endswith("USDT")


def _to_binance(sym: str) -> str:
    a = sym.upper().replace("-", "")
    return a if a.endswith("USDT") else (a[:-3] + "USDT" if a.endswith("USD") else a + "USDT")


def _fetch_stocks(syms: list[str]) -> dict[str, float]:
    if not syms or not (settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY):
        return {}
    try:
        h = {"APCA-API-KEY-ID": settings.ALPACA_API_KEY,
             "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY}
        r = requests.get("https://data.alpaca.markets/v2/stocks/trades/latest",
                         headers=h, params={"symbols": ",".join(syms)}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        trades = r.json().get("trades", {})
        return {s: float(t["p"]) for s, t in trades.items() if t.get("p")}
    except Exception:  # noqa: BLE001
        return {}


def _fetch_crypto(syms: list[str]) -> dict[str, float]:
    if not syms:
        return {}
    pairs = {_to_binance(s): s for s in syms}
    try:
        import json as _json
        # Binance rejects spaces in the symbols array — compact JSON, no spaces.
        sym_param = _json.dumps(list(pairs), separators=(",", ":"))
        r = requests.get(settings.BINANCE_BASE_URL.rstrip("/") + "/api/v3/ticker/price",
                         params={"symbols": sym_param}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return {}
        return {pairs[d["symbol"]]: float(d["price"]) for d in r.json()
                if d.get("symbol") in pairs and d.get("price")}
    except Exception:  # noqa: BLE001
        return {}


def _poll_once() -> None:
    with _LOCK:
        syms = set(_SYMBOLS)
    if not syms:
        return
    stocks = [s for s in syms if not _is_crypto(s)]
    crypto = [s for s in syms if _is_crypto(s)]
    fresh = {**_fetch_stocks(stocks), **_fetch_crypto(crypto)}
    if fresh:
        with _LOCK:
            _PRICES.update(fresh)


def _loop() -> None:
    while True:
        try:
            _poll_once()
        except Exception as exc:  # noqa: BLE001
            logger.warning("price poller: %s", exc)
        time.sleep(POLL_SEC)


def _ensure_thread() -> None:
    global _THREAD
    if _THREAD is None or not _THREAD.is_alive():
        _THREAD = threading.Thread(target=_loop, name="price-poller", daemon=True)
        _THREAD.start()


def watch(symbols: list[str]) -> None:
    """Set the symbol set to track and ensure the poller thread is running."""
    with _LOCK:
        _SYMBOLS.clear()
        _SYMBOLS.update(symbols)
    _ensure_thread()


def get(sym: str) -> float | None:
    with _LOCK:
        return _PRICES.get(sym)


def get_all() -> dict[str, float]:
    with _LOCK:
        return dict(_PRICES)
