"""Contract tests for config: position sizing and the ExecutionTicket invariants
that are the last line of defence before an order is built."""

import pytest

from config import Action, ExecutionTicket, cap_quantity, settings


def test_cap_quantity_clamps_to_no_leverage():
    # 100 units @ $50 = $5000 notional on a $1000 book -> clamp to 20 units.
    assert cap_quantity(100, 50, 1000) == 20


def test_cap_quantity_passthrough_on_bad_inputs():
    assert cap_quantity(7, 0, 1000) == 7
    assert cap_quantity(7, 50, 0) == 7


def _buy_ticket(**over):
    base = dict(
        asset="AAPL", action=Action.BUY, entry_price=100.0, stop_loss=99.0,
        take_profit=102.0, quantity=10.0, risk_dollars=10.0, risk_pct=0.01,
        reward_risk_ratio=2.0, capital_at_open=1000.0, rationale="t",
    )
    base.update(over)
    return ExecutionTicket(**base)


def test_buy_geometry_ok():
    t = _buy_ticket()
    assert t.action == Action.BUY


def test_buy_geometry_rejects_stop_above_entry():
    with pytest.raises(ValueError):
        _buy_ticket(stop_loss=101.0)  # stop must be < entry for a long


def test_risk_cap_enforced():
    over = settings.MAX_RISK_DOLLARS + 5
    with pytest.raises(ValueError):
        _buy_ticket(risk_dollars=over)


def test_hold_ticket_allows_zero_prices():
    t = ExecutionTicket(
        asset="AAPL", action=Action.HOLD, entry_price=0.0, stop_loss=0.0,
        take_profit=0.0, quantity=0.0, risk_dollars=0.0, risk_pct=0.0,
        reward_risk_ratio=0.0, capital_at_open=1000.0, rationale="stand down",
    )
    assert t.action == Action.HOLD
