"""Tests for the paper broker: atomic persistence, corrupt-log recovery, fill
math, portfolio guards, and the performance summary. No network — fills are
forced via the coinflip path and crafted log dicts."""

import json

import pytest

import execution
from config import Action, ExecutionTicket


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    """Point the module's TRADE_LOG at a throwaway file for each test."""
    p = tmp_path / "trade_log.json"
    monkeypatch.setattr(execution, "TRADE_LOG", p)
    return p


def _buy_ticket(**over):
    base = dict(
        asset="AAPL", action=Action.BUY, entry_price=100.0, stop_loss=99.0,
        take_profit=102.0, quantity=10.0, risk_dollars=10.0, risk_pct=0.01,
        reward_risk_ratio=2.0, capital_at_open=1000.0, rationale="t",
    )
    base.update(over)
    return ExecutionTicket(**base)


# --- persistence -----------------------------------------------------------
def test_save_load_roundtrip(log_path):
    log = execution._load_log()  # fresh default doc
    log["meta"]["equity"] = 1234.56
    execution._save_log(log)
    assert execution._load_log()["meta"]["equity"] == 1234.56


def test_save_is_atomic_no_tmp_left(log_path):
    execution._save_log(execution._load_log())
    leftovers = list(log_path.parent.glob("trade_log.*.tmp"))
    assert leftovers == []


def test_corrupt_log_is_preserved_not_dropped(log_path):
    log_path.write_text("{ this is not valid json")
    fresh = execution._load_log()
    # Started fresh...
    assert fresh["trades"] == []
    # ...but the corrupt bytes were preserved for recovery, not silently lost.
    backups = list(log_path.parent.glob("trade_log.corrupt.*.json"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{ this is not valid json"


# --- fill math -------------------------------------------------------------
def test_coinflip_take_profit_pnl():
    fill = execution._coinflip_fill(_buy_ticket(), outcome="tp")
    assert fill["result"] == "take_profit"
    assert fill["pnl"] == 20.0  # (102 - 100) * 10


def test_coinflip_stop_loss_pnl():
    fill = execution._coinflip_fill(_buy_ticket(), outcome="sl")
    assert fill["result"] == "stop_loss"
    assert fill["pnl"] == -10.0  # -(100 - 99) * 10


def test_sim_costs_applied_to_sim_fills():
    fill = {"fill_source": "yfinance", "result": "take_profit",
            "entry_price": 100.0, "quantity": 10.0}
    # notional 1000 * (2*5 + 2*2) bps / 10000 = 1.4
    assert execution._sim_costs(fill, _buy_ticket()) == 1.4


def test_sim_costs_zero_for_real_broker_fills():
    fill = {"fill_source": "binance", "result": "take_profit",
            "entry_price": 100.0, "quantity": 10.0}
    assert execution._sim_costs(fill, _buy_ticket()) == 0.0


# --- portfolio guards ------------------------------------------------------
def test_portfolio_block_dedup():
    log = {"meta": {"equity": 1000}, "trades": [
        {"asset": "AAPL", "result": "open", "risk_dollars": 10}]}
    assert "already holding AAPL" == execution._portfolio_block(_buy_ticket(), log)


def test_portfolio_block_total_risk():
    log = {"meta": {"equity": 1000}, "trades": [
        {"asset": "MSFT", "result": "open", "risk_dollars": 50},
        {"asset": "TSLA", "result": "open", "risk_dollars": 50}]}
    # open risk 100 + ticket 10 > 10% of 1000 -> blocked
    block = execution._portfolio_block(_buy_ticket(), log)
    assert block is not None and "portfolio risk" in block


def test_portfolio_block_none_when_clear():
    log = {"meta": {"equity": 1000}, "trades": []}
    assert execution._portfolio_block(_buy_ticket(), log) is None


def test_hold_ticket_never_blocked():
    hold = ExecutionTicket(
        asset="AAPL", action=Action.HOLD, entry_price=0.0, stop_loss=0.0,
        take_profit=0.0, quantity=0.0, risk_dollars=0.0, risk_pct=0.0,
        reward_risk_ratio=0.0, capital_at_open=1000.0, rationale="x")
    log = {"meta": {"equity": 1000}, "trades": [
        {"asset": "AAPL", "result": "open", "risk_dollars": 10}]}
    assert execution._portfolio_block(hold, log) is None


# --- execute_ticket end-to-end (forced outcome = no network) ---------------
def test_execute_ticket_commits_pnl_and_equity(log_path):
    # outcome forces the coinflip path, so no yfinance/broker network is hit.
    rec = execution.execute_ticket(_buy_ticket(), outcome="tp")
    assert rec["result"] == "take_profit"
    log = execution._load_log()
    assert log["meta"]["equity"] == rec["equity_after"]
    assert rec["equity_after"] > rec["equity_before"]  # a win raises equity
    assert len(log["trades"]) == 1


def test_execute_ticket_blocked_does_not_move_equity(log_path):
    # Pre-seed an open position in the same asset so the dedup guard blocks it.
    seed = {"meta": {"starting_capital": 1000, "equity": 1000}, "trades": [
        {"asset": "AAPL", "result": "open", "risk_dollars": 10, "equity_after": 1000}]}
    log_path.write_text(json.dumps(seed))
    rec = execution.execute_ticket(_buy_ticket(), outcome="tp")
    assert rec["result"] == "blocked"
    assert execution._load_log()["meta"]["equity"] == 1000  # unchanged


# --- summary ---------------------------------------------------------------
def test_performance_summary(log_path):
    doc = {
        "meta": {"starting_capital": 1000, "equity": 1020},
        "trades": [
            {"asset": "AAPL", "result": "take_profit", "pnl": 20.0,
             "risk_dollars": 10.0, "equity_after": 1020},
        ],
    }
    log_path.write_text(json.dumps(doc))
    s = execution.performance_summary()
    assert s["net_pnl"] == 20.0
    assert s["win_rate_pct"] == 100.0
    assert s["resolved_trades"] == 1
    assert s["expectancy"] == 20.0
