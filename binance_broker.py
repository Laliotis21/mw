"""
binance_broker.py
=================
Binance Spot **paper** broker. By default talks to the Spot Testnet
(testnet.binance.vision) — a real matching engine with fake balances, so orders
are genuine exchange orders that risk no real money. Flip BINANCE_TESTNET=false
to point at live Binance (real funds; you accept the risk).

Trade model — a real bracket, not a synchronous sim:
    1. MARKET entry (BUY) for the sized quantity.
    2. OCO exit: a take-profit LIMIT and a stop-loss STOP_LOSS_LIMIT, whichever
       fills first cancels the other.
The trade is then OPEN. Outcome is unknown until the OCO resolves, so we record
it as "open" and `reconcile()` later polls the order list to realize P&L.

Spot can't short, so only crypto BUY routes here. SELL/short and non-crypto
return None — the caller (execution.simulate_fill) falls back to the local sim.

No new dependency: raw signed REST over `requests` + stdlib hmac/hashlib.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from decimal import ROUND_DOWN, Decimal
from typing import Optional, Tuple
from urllib.parse import urlencode

import requests

from config import Action, ExecutionTicket, cap_quantity, logger, settings

RECV_WINDOW = 5000
REQUEST_TIMEOUT = 15
_FILTER_CACHE: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Symbol mapping — our tickers (BTC-USD) -> Binance spot pairs (BTCUSDT).
# --------------------------------------------------------------------------- #
def is_crypto(asset: str) -> bool:
    """True for our crypto tickers (BTC-USD / ETH-USDT style)."""
    a = asset.upper()
    return a.endswith("-USD") or a.endswith("-USDT") or a.endswith("USDT")


def to_binance_symbol(asset: str) -> Optional[str]:
    """'BTC-USD' -> 'BTCUSDT'. Returns None if not a mappable crypto ticker."""
    a = asset.upper().replace("-", "")
    if a.endswith("USDT"):
        return a
    if a.endswith("USD"):
        return a[:-3] + "USDT"  # USD-quoted -> USDT pair (testnet has USDT pairs)
    return None


# --------------------------------------------------------------------------- #
# Signed REST plumbing.
# --------------------------------------------------------------------------- #
def _signed(method: str, path: str, params: dict) -> dict:
    """Signed Binance REST call (HMAC-SHA256 over the query string)."""
    settings.require("BINANCE_API_KEY", "BINANCE_SECRET_KEY")
    params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": RECV_WINDOW}
    query = urlencode(params)
    sig = hmac.new(
        settings.BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    url = f"{settings.BINANCE_BASE_URL}{path}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": settings.BINANCE_API_KEY}
    resp = requests.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _public(path: str, params: dict) -> dict:
    resp = requests.get(
        f"{settings.BINANCE_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Exchange filters — Binance rejects orders whose qty/price violate the step /
# tick / minNotional filters, so we snap to them before sending.
# --------------------------------------------------------------------------- #
def _filters(symbol: str) -> dict:
    if symbol in _FILTER_CACHE:
        return _FILTER_CACHE[symbol]
    info = _public("/api/v3/exchangeInfo", {"symbol": symbol})
    syms = info.get("symbols", [])
    if not syms:
        raise ValueError(f"Binance has no symbol {symbol}")
    f = {flt["filterType"]: flt for flt in syms[0]["filters"]}
    out = {
        "step": Decimal(f["LOT_SIZE"]["stepSize"]),
        "tick": Decimal(f["PRICE_FILTER"]["tickSize"]),
        "min_notional": Decimal(
            (f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}).get("minNotional", "0")
        ),
    }
    _FILTER_CACHE[symbol] = out
    return out


def _snap(value: float, step: Decimal, rounding=ROUND_DOWN) -> str:
    """Round `value` down to a multiple of `step`; return as a plain string."""
    q = (Decimal(str(value)) / step).to_integral_value(rounding=rounding) * step
    return format(q.normalize(), "f")


# --------------------------------------------------------------------------- #
# Place a bracket: market entry + OCO exit.
# --------------------------------------------------------------------------- #
def place_bracket(ticket: ExecutionTicket) -> Optional[dict]:
    """
    Place a real spot bracket for a crypto BUY. Returns an 'open' fill record
    (entry filled, OCO resting) or None if this ticket can't route to Binance
    (not crypto / short / API error) so the caller can fall back to local sim.
    """
    if ticket.action != Action.BUY or not is_crypto(ticket.asset):
        return None  # spot can't short; non-crypto not on Binance
    symbol = to_binance_symbol(ticket.asset)
    if symbol is None:
        return None

    try:
        flt = _filters(symbol)
        # No leverage on spot — clamp notional to capital before snapping.
        capped = cap_quantity(ticket.quantity, float(ticket.entry_price), ticket.capital_at_open)
        qty_str = _snap(capped, flt["step"])
        if Decimal(qty_str) <= 0:
            logger.warning("Binance: qty rounds to 0 for %s — fall back.", symbol)
            return None

        # 1) Market entry.
        entry = _signed(
            "POST", "/api/v3/order",
            {"symbol": symbol, "side": "BUY", "type": "MARKET", "quantity": qty_str},
        )
        fills = entry.get("fills") or []
        if fills:
            spent = sum(float(f["price"]) * float(f["qty"]) for f in fills)
            got = sum(float(f["qty"]) for f in fills)
            entry_price = spent / got if got else float(ticket.entry_price)
            filled_qty = got
        else:
            entry_price = float(ticket.entry_price)
            filled_qty = float(qty_str)

        # 2) OCO exit (SELL): TP limit + stop. Re-anchor geometry to the real fill.
        risk_pct = abs(ticket.entry_price - ticket.stop_loss) / ticket.entry_price
        rew_pct = abs(ticket.take_profit - ticket.entry_price) / ticket.entry_price
        tp = entry_price * (1 + rew_pct)
        stop = entry_price * (1 - risk_pct)
        tp_str = _snap(tp, flt["tick"])
        stop_str = _snap(stop, flt["tick"])
        stop_limit_str = _snap(stop * 0.999, flt["tick"])  # limit just below trigger
        sell_qty = _snap(filled_qty, flt["step"])

        oco = _signed(
            "POST", "/api/v3/order/oco",
            {
                "symbol": symbol, "side": "SELL", "quantity": sell_qty,
                "price": tp_str,  # take-profit limit
                "stopPrice": stop_str, "stopLimitPrice": stop_limit_str,
                "stopLimitTimeInForce": "GTC",
            },
        )
        return {
            "result": "open",
            "pnl": 0.0,
            "entry_price": round(entry_price, 2),
            "stop_loss": float(stop_str),
            "take_profit": float(tp_str),
            "quantity": float(sell_qty),
            "exit_price": None,
            "risk_dollars": round((entry_price - float(stop_str)) * float(sell_qty), 2),
            "fill_source": "binance_testnet" if settings.BINANCE_TESTNET else "binance_live",
            "binance_symbol": symbol,
            "binance_order_list_id": oco.get("orderListId"),
        }
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        logger.error("Binance bracket failed for %s: %s %s", ticket.asset, exc, body[:200])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Binance bracket error for %s: %s", ticket.asset, exc)
        return None


# --------------------------------------------------------------------------- #
# Manual close — cancel the OCO and market-sell the spot holding now.
# --------------------------------------------------------------------------- #
def close_position(record: dict) -> Optional[Tuple[str, float, float]]:
    """
    Close an open spot bracket on demand: cancel the resting OCO, then market-
    sell the held base qty. Returns ('manual_close', exit_price, pnl). None on
    error. (Spot is long-only, so closing a BUY is always a SELL.)
    """
    symbol = record.get("binance_symbol")
    list_id = record.get("binance_order_list_id")
    if not symbol:
        return None
    try:
        # 1) Cancel the resting OCO (ignore if already gone).
        if list_id not in (None, -1):
            try:
                _signed("DELETE", "/api/v3/orderList",
                        {"symbol": symbol, "orderListId": list_id})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Binance OCO cancel %s: %s", symbol, exc)
        # 2) Market-sell the held quantity (snap to lot size).
        flt = _filters(symbol)
        qty_str = _snap(float(record["quantity"]), flt["step"])
        if Decimal(qty_str) <= 0:
            return ("canceled", 0.0, 0.0)
        sell = _signed("POST", "/api/v3/order",
                       {"symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": qty_str})
        fills = sell.get("fills") or []
        got = sum(float(f["qty"]) for f in fills) or float(qty_str)
        spent = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        entry = float(record["entry_price"])
        exit_price = spent / got if got and spent else entry
        pnl = round((exit_price - entry) * got, 2)
        return ("manual_close", round(exit_price, 2), pnl)
    except Exception as exc:  # noqa: BLE001
        logger.error("Binance close_position failed for %s: %s", symbol, exc)
        return None


# --------------------------------------------------------------------------- #
# Reconcile an open bracket — poll the OCO, realize P&L when it resolves.
# --------------------------------------------------------------------------- #
def reconcile_bracket(record: dict) -> Optional[Tuple[str, float, float]]:
    """
    Poll a previously-placed OCO. Returns (result, exit_price, pnl) once the
    bracket is ALL_DONE, else None (still working). result is 'take_profit' or
    'stop_loss'. Caller updates equity from pnl.
    """
    symbol = record.get("binance_symbol")
    list_id = record.get("binance_order_list_id")
    if not symbol or list_id in (None, -1):
        return None
    try:
        ol = _signed("GET", "/api/v3/orderList", {"orderListId": list_id})
        if ol.get("listOrderStatus") != "ALL_DONE":
            return None
        order_ids = [o["orderId"] for o in ol.get("orders", [])]
        entry = float(record["entry_price"])
        qty = float(record["quantity"])
        for oid in order_ids:
            o = _signed("GET", "/api/v3/order", {"symbol": symbol, "orderId": oid})
            if o.get("status") != "FILLED":
                continue
            exit_price = float(o.get("price") or 0) or float(o.get("stopPrice") or 0)
            cqq = float(o.get("cummulativeQuoteQty") or 0)
            executed = float(o.get("executedQty") or qty)
            if cqq and executed:
                exit_price = cqq / executed
            # TP leg is a LIMIT_MAKER/LIMIT; stop leg is STOP_LOSS_LIMIT.
            is_tp = "STOP" not in (o.get("type") or "")
            pnl = round((exit_price - entry) * executed, 2)
            return ("take_profit" if is_tp else "stop_loss", round(exit_price, 2), pnl)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Binance reconcile failed for %s: %s", symbol, exc)
        return None
