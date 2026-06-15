"""Deterministic tests for strategy.simulate_exit — the shared exit engine used
by both the paper fill and the backtest. Pure function, no network."""

from strategy import simulate_exit


# --- 'target' style (fixed stop / 2R) --------------------------------------
def test_target_long_take_profit():
    # entry 100, stop 99 (risk 1), target 102. A bar that tags the target wins 2R.
    res, r, idx = simulate_exit([103], [100], [101], 100, 99, 102, True, "target")
    assert res == "take_profit" and r == 2.0 and idx == 0


def test_target_long_stop_loss():
    res, r, _ = simulate_exit([100.5], [98], [99], 100, 99, 102, True, "target")
    assert res == "stop_loss" and r == -1.0


def test_target_straddle_is_stop_first():
    # One bar tags BOTH stop and target -> conservative: stop wins.
    res, r, _ = simulate_exit([103], [98], [101], 100, 99, 102, True, "target")
    assert res == "stop_loss" and r == -1.0


def test_target_markout():
    res, r, idx = simulate_exit([101, 101], [99.5, 99.5], [100.5, 100.5],
                                100, 99, 102, True, "target")
    assert res == "markout" and r == 0.5 and idx == 1


# --- 'be_partial' style (half at 1R + breakeven) ---------------------------
def test_be_partial_banks_half_then_breakeven():
    # Reaches +1R on bar 1 (half off, stop -> breakeven); bar 2 falls back to
    # breakeven. A trade that would have round-tripped now banks +0.5R.
    res, r, idx = simulate_exit([101.5, 100.2], [100, 99.5], [100.5, 99.8],
                                100, 99, 102, True, "be_partial")
    assert res == "take_profit" and r == 0.5 and idx == 1


def test_be_partial_full_target_is_1_5R():
    # Hits 1R and the 2R target in the same bar: half at +1R, half at +2R = 1.5R.
    res, r, _ = simulate_exit([102.5], [100], [102.2], 100, 99, 102, True, "be_partial")
    assert res == "take_profit" and r == 1.5


def test_be_partial_clean_stop_is_full_loss():
    # Never reaches 1R, stop hit -> full -1R (same as target style).
    res, r, _ = simulate_exit([100.4], [98], [99], 100, 99, 102, True, "be_partial")
    assert res == "stop_loss" and r == -1.0


def test_short_target_take_profit():
    # Short: entry 100, stop 101, target 98. Low tags target -> 2R.
    res, r, _ = simulate_exit([100], [97], [98.5], 100, 101, 98, False, "target")
    assert res == "take_profit" and r == 2.0
