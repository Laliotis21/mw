"""
alpaca_broker.py
================
Alpaca **paper** stock broker. Talks to the paper-trading REST API
(paper-api.alpaca.markets) — real order engine, fake money. Mirrors
binance_broker so execution.py can route per asset:

    crypto -> Binance Spot Testnet   (binance_broker)
    stocks -> Alpaca paper           (this module)

Trade model = a native Alpaca **bracket** order (order_class="bracket"): one
market entry plus a take-profit limit child and a stop-loss stop child, OCO. The
trade is OPEN after placement; reconcile_bracket() polls the order (nested legs)
and realizes P&L once a leg fills. Unlike spot crypto, Alpaca paper supports
SHORTS, so SELL signals route here too.

No new dependency: raw REST over `requests`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import requests

from config import Action, ExecutionTicket, cap_quantity, logger, settings

REQUEST_TIMEOUT = 15


def _headers() -> dict:
    settings.require("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    return {
        "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
    }


def _url(path: str) -> str:
    return f"{settings.ALPACA_BASE_URL.rstrip('/')}{path}"


def is_stock(asset: str) -> bool:
    """True for equities/ETFs — i.e. NOT our crypto '-USD' tickers."""
    a = asset.upper()
    return not (a.endswith("-USD") or a.endswith("-USDT") or a.endswith("USDT"))


# --------------------------------------------------------------------------- #
# Place a bracket: market entry + TP/SL children (OCO).
# --------------------------------------------------------------------------- #
def place_bracket(ticket: ExecutionTicket) -> Optional[dict]:
    """
    Place an Alpaca paper bracket for a stock BUY or SELL(short). Returns an
    'open' fill record (entry submitted, TP/SL resting) or None if it can't route
    (not a stock / qty < 1 share / API error) so the caller falls back to sim.
    """
    if ticket.action not in (Action.BUY, Action.SELL) or not is_stock(ticket.asset):
        return None

    # Cap notional to capital (no leverage), then floor to WHOLE shares —
    # bracket/advanced order classes reject fractional qty.
    capped = cap_quantity(ticket.quantity, float(ticket.entry_price), ticket.capital_at_open)
    qty = int(math.floor(capped))
    if qty < 1:
        logger.warning("Alpaca: qty %.4f < 1 share for %s — fall back.", ticket.quantity, ticket.asset)
        return None

    is_long = ticket.action == Action.BUY
    entry = float(ticket.entry_price)
    risk_pct = abs(ticket.entry_price - ticket.stop_loss) / ticket.entry_price
    rew_pct = abs(ticket.take_profit - ticket.entry_price) / ticket.entry_price
    if is_long:
        tp, sl = entry * (1 + rew_pct), entry * (1 - risk_pct)
    else:
        tp, sl = entry * (1 - rew_pct), entry * (1 + risk_pct)

    body = {
        "symbol": ticket.asset,
        "qty": str(qty),
        "side": "buy" if is_long else "sell",
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": round(tp, 2)},
        "stop_loss": {"stop_price": round(sl, 2)},
    }
    try:
        r = requests.post(_url("/v2/orders"), headers=_headers(), json=body, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            logger.error("Alpaca bracket rejected %s: %s %s", ticket.asset, r.status_code, r.text[:200])
            return None
        o = r.json()
        # filled_avg_price is null until the entry fills (off-hours stays open).
        fap = o.get("filled_avg_price")
        entry_price = float(fap) if fap else entry
        return {
            "result": "open",
            "pnl": 0.0,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "quantity": qty,
            "exit_price": None,
            "risk_dollars": round(abs(entry_price - sl) * qty, 2),
            "fill_source": "alpaca_paper",
            "alpaca_order_id": o.get("id"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Alpaca bracket error for %s: %s", ticket.asset, exc)
        return None


# --------------------------------------------------------------------------- #
# Reconcile an open bracket.
# --------------------------------------------------------------------------- #
def reconcile_bracket(record: dict) -> Optional[Tuple[str, float, float]]:
    """
    Poll an Alpaca bracket. Returns (result, exit_price, pnl) once a TP/SL leg
    fills, else None (entry not filled, or still resting). result is
    'take_profit' or 'stop_loss'.
    """
    oid = record.get("alpaca_order_id")
    if not oid:
        return None
    try:
        r = requests.get(_url(f"/v2/orders/{oid}"), headers=_headers(),
                         params={"nested": "true"}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        o = r.json()
        fap = o.get("filled_avg_price")
        if not fap:
            return None  # entry not filled yet
        entry = float(fap)
        qty = int(float(o.get("filled_qty") or record["quantity"]))
        is_long = (o.get("side") == "buy")
        for leg in o.get("legs") or []:
            if leg.get("status") != "filled":
                continue
            lfap = leg.get("filled_avg_price")
            if not lfap:
                continue
            exit_price = float(lfap)
            is_tp = (leg.get("type") == "limit")  # limit leg = take-profit
            pnl = (exit_price - entry) * qty if is_long else (entry - exit_price) * qty
            return ("take_profit" if is_tp else "stop_loss", round(exit_price, 2), round(pnl, 2))
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Alpaca reconcile failed for %s: %s", oid, exc)
        return None
