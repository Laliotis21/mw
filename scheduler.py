"""
scheduler.py
============
Autonomous run loop — trades on its own instead of you pressing the button.
Runs the discovery → desk → execute pipeline on a fixed interval, with a smart
gate: stock candidates trade only while the US market is open; crypto trades
24/7 (Binance). Reconciles open positions every tick and respects the daily
loss circuit breaker. The Streamlit dashboard just monitors the trade log.

Run:  v/bin/python scheduler.py
Stop: Ctrl-C (or kill the process).

Config (.env):
    AUTORUN_INTERVAL_MIN  minutes between cycles (default 30)
"""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from datetime import timedelta, timezone

from binance_broker import is_crypto
from config import current_market_phase, logger, settings
from execution import (
    current_equity,
    execute_ticket,
    log_usage,
    reconcile_open,
    set_autorun_status,
)
from main import _circuit_breaker_tripped, pop_last_usage, run_cycle, run_discovery


def _publish_status(cycle: int, note: str, traded: int, held: int, mkt_open: bool) -> None:
    """Write the scheduler heartbeat so the dashboard can show what it's doing."""
    now = datetime.now(timezone.utc)
    set_autorun_status({
        "cycle": cycle,
        "interval_min": settings.AUTORUN_INTERVAL_MIN,
        "last_run_utc": now.isoformat(),
        "next_run_utc": (now + timedelta(minutes=settings.AUTORUN_INTERVAL_MIN)).isoformat(),
        "market_open": mkt_open,
        "traded": traded,
        "held": held,
        "note": note,
        "equity": current_equity(),
    })


def us_market_open() -> bool:
    """Authoritative US-equity open check via Alpaca clock; ET-hours fallback."""
    try:
        import requests
        h = {"APCA-API-KEY-ID": settings.ALPACA_API_KEY,
             "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY}
        r = requests.get(settings.ALPACA_BASE_URL.rstrip("/") + "/v2/clock",
                         headers=h, timeout=10)
        if r.status_code == 200:
            return bool(r.json().get("is_open"))
    except Exception:  # noqa: BLE001
        pass
    et = datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    return (et.hour, et.minute) >= (9, 30) and et.hour < 16


def run_once(day_open_equity: float, cycle: int) -> None:
    """One autonomous cycle: discover, gate per asset class, trade, reconcile."""
    mkt_open = us_market_open()
    logger.info("AUTORUN cycle #%d | phase=%s market_open=%s", cycle,
                current_market_phase().value, mkt_open)
    reconcile_open()  # settle anything that closed since last tick

    if _circuit_breaker_tripped(day_open_equity):
        logger.info("AUTORUN #%d: circuit breaker tripped — standing down.", cycle)
        _publish_status(cycle, "circuit-breaker — stood down", 0, 0, mkt_open)
        return

    phase = current_market_phase().value
    shortlist = run_discovery(phase)
    log_usage(pop_last_usage())
    if shortlist is None or not shortlist.ideas:
        logger.info("AUTORUN #%d: no ideas this cycle.", cycle)
        _publish_status(cycle, "no tradable ideas", 0, 0, mkt_open)
        return

    traded = held = 0
    for idea in shortlist.ideas[: settings.MAX_CANDIDATES]:
        if _circuit_breaker_tripped(day_open_equity):
            break
        # Gate: stocks only during market hours; crypto always.
        if not is_crypto(idea.asset) and not mkt_open:
            logger.info("AUTORUN #%d: skip stock %s — US market closed.", cycle, idea.asset)
            continue
        ticket = run_cycle(idea.asset, phase)
        usage = pop_last_usage()
        if ticket is None:
            log_usage(usage)
            continue
        rec = execute_ticket(ticket, market_phase=phase, usage=usage)
        if rec.get("result") == "no_trade":
            held += 1
        else:
            traded += 1
    logger.info("AUTORUN #%d done | %d traded / %d hold | equity $%.2f",
                cycle, traded, held, current_equity())
    _publish_status(cycle, "ok", traded, held, mkt_open)


def main() -> None:
    interval = int(settings.AUTORUN_INTERVAL_MIN) * 60
    logger.info("AUTORUN started | every %dmin | engine=%s fill=%s | stocks gated to "
                "market hours, crypto 24/7.", settings.AUTORUN_INTERVAL_MIN,
                settings.DECISION_ENGINE, settings.FILL_SOURCE)
    day = date.today()
    day_open_equity = current_equity()
    cycle = 0
    while True:
        if date.today() != day:  # new trading day → report + reset the loss baseline
            eq = current_equity()
            logger.info("AUTORUN DAILY REPORT %s | P&L $%.2f | equity $%.2f",
                        day, eq - day_open_equity, eq)
            day, day_open_equity = date.today(), eq
        cycle += 1
        try:
            run_once(day_open_equity, cycle)
        except Exception as exc:  # noqa: BLE001 — never let one cycle kill the loop
            logger.error("AUTORUN cycle #%d error: %s", cycle, exc)
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("AUTORUN stopped.")
