"""Unit tests for the deterministic indicator helpers in strategy.py.

These are pure functions over pandas Series — no network, no yfinance — so they
lock in the math (true ATR, Wilder RSI) that drives every rules-engine trade.
"""

import pandas as pd

from strategy import _true_atr, _wilder_rsi, efficiency_ratio


def test_efficiency_ratio_straight_line_is_one():
    close = pd.Series([float(i) for i in range(21)])  # perfectly clean trend
    assert efficiency_ratio(close, period=20) == 1.0


def test_efficiency_ratio_chop_is_near_zero():
    close = pd.Series([0.0, 1.0] * 11)  # pure zigzag, no net progress
    assert efficiency_ratio(close, period=20) < 0.1


def test_efficiency_ratio_thin_history_is_zero():
    assert efficiency_ratio([1.0, 2.0, 3.0], period=20) == 0.0


def test_wilder_rsi_all_gains_is_100():
    close = pd.Series([float(i) for i in range(1, 40)])  # strictly increasing
    assert _wilder_rsi(close, period=14) == 100.0


def test_wilder_rsi_all_losses_is_0():
    close = pd.Series([float(i) for i in range(40, 1, -1)])  # strictly decreasing
    assert _wilder_rsi(close, period=14) == 0.0


def test_wilder_rsi_neutral_when_too_short():
    close = pd.Series([10.0, 11.0, 10.5])  # fewer than `period` deltas
    assert _wilder_rsi(close, period=14) == 50.0


def test_wilder_rsi_mixed_in_range():
    close = pd.Series([10, 11, 10.5, 11.5, 11, 12, 11.8, 12.5, 12, 13,
                       12.7, 13.5, 13, 14, 13.8, 14.5])
    rsi = _wilder_rsi(close, period=14)
    assert 0.0 < rsi < 100.0


def test_true_atr_equals_high_low_without_gaps():
    # close sits inside the prior bar's range -> true range == high - low.
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([9.0, 9.0, 9.0])
    close = pd.Series([9.5, 9.5, 9.5])
    assert _true_atr(high, low, close, period=14) == 1.0


def test_true_atr_widens_on_a_gap():
    # A gap up makes |H - prevClose| exceed the bar's own high-low range.
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.0])
    close = pd.Series([9.5, 11.5])
    atr = _true_atr(high, low, close, period=14)
    naive = float((high - low).iloc[-1])  # 1.0
    assert atr > naive
