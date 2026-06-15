"""Tests for the trend-alignment gate in strategy.rules_signal. We monkeypatch
the indicator + news layers so no network/yfinance is touched — the gate logic
is exercised directly."""

import strategy
from config import Action


def _ind(**over):
    """A full indicator snapshot that, by default, fires a clean BUY."""
    base = dict(
        last=110.0, sma20=105.0, sma50=100.0, atr=2.0, rsi=60.0,
        hi20=111.0, lo20=95.0, mom=2.0, stop_dist=3.0, bars=80,
        vol_ok=True, er=0.5, sma50_rising=True,
    )
    base.update(over)
    return base


def _patch(monkeypatch, ind):
    monkeypatch.setattr(strategy, "_indicators", lambda a: ind)
    monkeypatch.setattr(strategy, "_news", lambda a: {
        "score": 0.0, "n": 0, "catalyst": False, "top": "", "fresh_hours": None})


def test_aligned_buy_passes(monkeypatch):
    _patch(monkeypatch, _ind(sma50_rising=True))
    sig = strategy.rules_signal("AAPL", "mid_day")
    assert sig.action == Action.BUY


def test_counter_trend_buy_vetoed(monkeypatch):
    # Uptrend signal but SMA50 falling -> counter-trend long is dropped.
    _patch(monkeypatch, _ind(sma50_rising=False))
    sig = strategy.rules_signal("AAPL", "mid_day")
    assert sig.action == Action.HOLD
    assert "counter-trend" in sig.rationale


def test_filter_off_lets_counter_trend_through(monkeypatch):
    monkeypatch.setattr(strategy.settings, "TREND_FILTER", False)
    _patch(monkeypatch, _ind(sma50_rising=False))
    sig = strategy.rules_signal("AAPL", "mid_day")
    assert sig.action == Action.BUY  # gate disabled


def test_thin_history_not_gated(monkeypatch):
    # Unknown slope (thin history) must not veto a signal.
    _patch(monkeypatch, _ind(sma50_rising=None))
    sig = strategy.rules_signal("AAPL", "mid_day")
    assert sig.action == Action.BUY
