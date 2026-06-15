"""
backtest.py
===========
Historical backtest of the TECHNICAL core of the rules engine — the only honest
way to know if the strategy has an edge before trusting it live. Candle-only:
it mirrors strategy.py's SMA20/50 trend + RSI + ATR-stop + 2R + breakout rules
(crypto long-only, no-chase, volume confirm), with NO news/catalyst layer (news
is live-only and can't be replayed). So it measures the technical floor.

No lookahead: the signal at bar i uses only data up to i; the trade is then
walked forward over the next bars. One position at a time per asset.

Run:
    v/bin/python backtest.py --asset AAPL --asset MSFT --days 365
    v/bin/python backtest.py --asset BTC-USD --days 720 --max-hold 15
"""

from __future__ import annotations

import argparse

from strategy import ATR_STOP_MULT, REWARD_RISK


def _is_crypto(a: str) -> bool:
    return a.upper().endswith(("-USD", "-USDT", "USDT"))


def backtest_asset(asset: str, days: int = 365, max_hold: int = 20) -> dict:
    """Return per-asset stats: trades, win%, expectancy (R), avg R, profit factor."""
    import numpy as np
    import yfinance as yf

    df = yf.Ticker(asset).history(period=f"{days}d", interval="1d")
    if df is None or df.empty or len(df) < 60:
        return {"asset": asset, "trades": 0, "note": "insufficient history"}

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    n = len(close)
    crypto = _is_crypto(asset)

    def rsi_at(i: int) -> float:
        d = np.diff(close[max(0, i - 14):i + 1])
        if len(d) < 1:
            return 50.0
        gain = d[d > 0].sum() / 14
        loss = -d[d < 0].sum() / 14
        return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)

    r_multiples: list[float] = []
    i = 51
    while i < n - 1:
        last = close[i]
        sma20 = close[i - 20:i].mean()
        sma50 = close[i - 50:i].mean()
        atr = (high[i - 14:i] - low[i - 14:i]).mean()
        hi20 = high[i - 20:i].max()
        lo20 = low[i - 20:i].min()
        rsi = rsi_at(i)
        mom = (close[i] / close[i - 5] - 1) * 100 if i >= 5 else 0.0
        sd = max(ATR_STOP_MULT * atr, last * 0.005)
        avg_vol = vol[i - 20:i].mean()
        vol_ok = avg_vol <= 0 or vol[i] >= 0.7 * avg_vol
        extended = last > sma20 * 1.08

        up = last > sma20 > sma50
        down = last < sma20 < sma50
        brk = last >= hi20 * 0.999
        brkd = last <= lo20 * 1.001
        buy = (((up and 45 <= rsi <= 72) or (brk and rsi < 75 and mom > 0))
               and vol_ok and not extended)
        sell = (((down and 28 <= rsi <= 55) or (brkd and rsi > 25 and mom < 0))
                and vol_ok and not crypto)  # crypto long-only

        action = "BUY" if (buy and not sell) else ("SELL" if (sell and not buy) else None)
        if action is None:
            i += 1
            continue

        entry = last
        if action == "BUY":
            stop, target = entry - sd, entry + REWARD_RISK * sd
        else:
            stop, target = entry + sd, entry - REWARD_RISK * sd

        # Walk forward; stop assumed to fill first on a straddling bar.
        outcome_r = 0.0
        end = min(i + max_hold, n - 1)
        for j in range(i + 1, end + 1):
            hj, lj = high[j], low[j]
            if action == "BUY":
                if lj <= stop:
                    outcome_r = -1.0; i = j; break
                if hj >= target:
                    outcome_r = REWARD_RISK; i = j; break
            else:
                if hj >= stop:
                    outcome_r = -1.0; i = j; break
                if lj <= target:
                    outcome_r = REWARD_RISK; i = j; break
        else:  # neither hit — mark to close in R
            exitp = close[end]
            outcome_r = ((exitp - entry) if action == "BUY" else (entry - exitp)) / sd
            i = end
        r_multiples.append(round(outcome_r, 2))
        i += 1

    if not r_multiples:
        return {"asset": asset, "trades": 0, "note": "no signals"}
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]
    gross_w, gross_l = sum(wins), abs(sum(losses))
    # Equity curve in R, for max drawdown.
    eq = 0.0; peak = 0.0; mdd = 0.0
    for r in r_multiples:
        eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq)
    return {
        "asset": asset,
        "trades": len(r_multiples),
        "win_pct": round(len(wins) / len(r_multiples) * 100, 1),
        "expectancy_r": round(sum(r_multiples) / len(r_multiples), 3),
        "total_r": round(sum(r_multiples), 2),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l else (999.0 if gross_w else 0.0),
        "max_dd_r": round(mdd, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest the technical rules core.")
    p.add_argument("--asset", action="append", default=[], help="Ticker (repeat).")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--max-hold", type=int, default=20, help="Max bars to hold a trade.")
    a = p.parse_args()
    assets = a.asset or ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

    print(f"\nBACKTEST · technical-only (no news) · {a.days}d · max-hold {a.max_hold} bars")
    print("-" * 78)
    print(f"{'asset':10} {'trades':>7} {'win%':>6} {'exp(R)':>8} {'totR':>8} {'PF':>6} {'maxDD(R)':>9}")
    agg_r: list[float] = []
    tot_trades = 0
    for asset in assets:
        s = backtest_asset(asset, a.days, a.max_hold)
        if s["trades"] == 0:
            print(f"{asset:10} {'—':>7}  {s.get('note','')}")
            continue
        print(f"{s['asset']:10} {s['trades']:>7} {s['win_pct']:>6} {s['expectancy_r']:>8} "
              f"{s['total_r']:>8} {s['profit_factor']:>6} {s['max_dd_r']:>9}")
        tot_trades += s["trades"]
        agg_r.append(s["total_r"])
    print("-" * 78)
    if agg_r:
        print(f"TOTAL: {tot_trades} trades · sum {sum(agg_r):+.2f}R across {len(agg_r)} assets · "
              f"avg {sum(agg_r)/len(agg_r):+.2f}R/asset\n"
              f"(>0R = edge; expectancy >0 and PF >1 are what you want.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
