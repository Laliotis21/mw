"""
main.py
=======
Orchestrator. Wires agents + tasks into a sequential CrewAI crew, runs one
decision cycle for a given asset/phase, and paper-executes the resulting ticket.

Pipeline per cycle:
    Researcher (Perplexity)  ->  Analyst (signal)  ->  Risk Manager (ticket)
        -> execution.execute_ticket() -> trade_log.json

Safety:
    - Daily-loss circuit breaker: if equity has dropped past DAILY_MAX_LOSS_PCT
      vs the day's opening equity, the bot stands down for the rest of the run.
    - Every cycle is wrapped so one bad asset can't kill the batch.

Usage:
    python main.py --asset AAPL --phase pre_market
    python main.py --asset AAPL --phase open --asset BTC-USD   # multiple
    python main.py --summary                                   # show stats only
"""

from __future__ import annotations

import argparse
import sys

from crewai import Crew, Process

from config import (
    Action,
    ExecutionTicket,
    MarketPhase,
    OpportunityShortlist,
    TradeSignal,
    current_market_phase,
    logger,
    settings,
)
from execution import (
    current_equity,
    execute_ticket,
    log_usage,
    performance_summary,
    reconcile_open,
)

# Token usage from the most recent crew.kickoff(), stashed so the caller can
# attribute it to the trade it produced (CrewAI aggregates per-crew usage).
_LAST_USAGE: dict = {}


def _usage_dict(token_usage) -> dict:
    """Flatten CrewAI's UsageMetrics into a plain dict tagged with the model."""
    if token_usage is None:
        return {}
    g = lambda a: int(getattr(token_usage, a, 0) or 0)  # noqa: E731
    model = (
        settings.ANTHROPIC_MODEL
        if settings.LLM_PROVIDER == "anthropic"
        else f"ollama/{settings.OLLAMA_MODEL}"
    )
    return {
        "model": model,
        "prompt_tokens": g("prompt_tokens"),
        "completion_tokens": g("completion_tokens"),
        "cached_tokens": g("cached_prompt_tokens"),
        "total_tokens": g("total_tokens"),
        "requests": g("successful_requests"),
    }


def pop_last_usage() -> dict:
    """Return (and clear) the token usage from the last run_cycle/run_discovery."""
    global _LAST_USAGE
    u, _LAST_USAGE = _LAST_USAGE, {}
    return u


def _merge_usage(a: dict, b: dict) -> dict:
    """Sum two token-usage bundles (e.g. analysis crew + risk crew)."""
    if not a:
        return b or {}
    if not b:
        return a
    out = dict(a)
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens",
              "total_tokens", "requests"):
        out[k] = (a.get(k, 0) or 0) + (b.get(k, 0) or 0)
    out["model"] = b.get("model") or a.get("model")
    return out


def run_cycle(
    asset: str,
    market_phase: str,
    step_callback=None,
    task_callback=None,
) -> ExecutionTicket | None:
    """
    Run one full research -> signal -> risk cycle and return the ticket.

    Optional callbacks let a UI watch the agents work in real time:
      - step_callback(step):  fired on every agent step / tool call.
      - task_callback(out):   fired when each task (agent) completes.
    Both are passed straight through to the CrewAI Crew.

    Returns None if the crew failed to produce a valid ticket (logged).
    """
    # Local imports keep agent/LLM construction lazy (per-cycle fresh state).
    from agents import build_analyst, build_researcher, build_risk_officer
    from tasks import build_analysis_task, build_research_task, build_risk_task

    logger.info("=== CYCLE START | asset=%s phase=%s ===", asset, market_phase)

    global _LAST_USAGE
    _LAST_USAGE = {}  # clear up front so a kickoff failure can't leak prior usage

    researcher = build_researcher()
    analyst = build_analyst()

    research_task = build_research_task(researcher, asset, market_phase)
    analysis_task = build_analysis_task(analyst, asset, research_task)

    # Stage 1: research -> signal. We inspect the analyst's call BEFORE paying
    # for the risk agent — a HOLD needs no sizing, so we skip that LLM entirely.
    desk_crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
        task_callback=task_callback,
    )
    try:
        signal_res = desk_crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        logger.error("Research/analysis crew failed for %s: %s", asset, exc)
        return None
    usage = _usage_dict(getattr(signal_res, "token_usage", None))

    signal = getattr(signal_res, "pydantic", None)
    if not isinstance(signal, TradeSignal):
        try:
            signal = TradeSignal.model_validate_json(str(signal_res))
        except Exception as exc:  # noqa: BLE001
            logger.error("No valid TradeSignal for %s: %s", asset, exc)
            _LAST_USAGE = usage
            return None

    # HOLD short-circuit: build a stand-down ticket directly — no risk LLM call.
    if signal.action == Action.HOLD:
        _LAST_USAGE = usage
        logger.info("HOLD %s — skipping risk agent (no sizing needed).", asset)
        return ExecutionTicket(
            asset=asset, action=Action.HOLD,
            entry_price=signal.suggested_entry or 0.0,
            stop_loss=signal.suggested_stop or 0.0,
            take_profit=signal.suggested_target or 0.0,
            quantity=0.0, risk_dollars=0.0, risk_pct=0.0,
            reward_risk_ratio=0.0, capital_at_open=current_equity(),
            rationale=signal.rationale,
        )

    # Stage 2: size the trade (BUY/SELL only). Signal is embedded in the task,
    # so the risk agent runs in its own crew with no cross-crew context.
    risk_officer = build_risk_officer()
    risk_task = build_risk_task(risk_officer, asset, signal)
    risk_crew = Crew(
        agents=[risk_officer],
        tasks=[risk_task],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
        task_callback=task_callback,
    )
    try:
        risk_res = risk_crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        logger.error("Risk crew failed for %s: %s", asset, exc)
        _LAST_USAGE = usage
        return None
    _LAST_USAGE = _merge_usage(usage, _usage_dict(getattr(risk_res, "token_usage", None)))

    ticket = getattr(risk_res, "pydantic", None)
    if not isinstance(ticket, ExecutionTicket):
        try:
            ticket = ExecutionTicket.model_validate_json(str(risk_res))
        except Exception as exc:  # noqa: BLE001
            logger.error("No valid ExecutionTicket from risk crew for %s: %s", asset, exc)
            return None

    logger.info(
        "TICKET | %s %s qty=%s entry=%s sl=%s tp=%s risk=$%.2f",
        ticket.action.value,
        ticket.asset,
        ticket.quantity,
        ticket.entry_price,
        ticket.stop_loss,
        ticket.take_profit,
        ticket.risk_dollars,
    )
    return ticket


def run_discovery(
    market_phase: str,
    step_callback=None,
    task_callback=None,
) -> OpportunityShortlist | None:
    """
    Run the 4-scout discovery crew and return a ranked shortlist of candidates.

    Pipeline:
        World Macro Scout  ─┐
        Stock Scanner      ─┼─→ Opportunity Ranker → OpportunityShortlist
        Crypto Scanner     ─┘

    The ranker consumes all three scout outputs (CrewAI context) and emits a
    strict OpportunityShortlist. Returns None if the crew produced nothing valid.
    """
    from agents import (
        build_crypto_scanner,
        build_opportunity_ranker,
        build_stock_scanner,
        build_world_scout,
    )
    from tasks import (
        build_crypto_scan_task,
        build_rank_task,
        build_stock_scan_task,
        build_world_task,
    )

    logger.info("=== DISCOVERY START | phase=%s ===", market_phase)

    global _LAST_USAGE
    _LAST_USAGE = {}  # clear up front so a kickoff failure can't leak prior usage

    world_scout = build_world_scout()
    stock_scanner = build_stock_scanner()
    crypto_scanner = build_crypto_scanner()
    ranker = build_opportunity_ranker()

    world_task = build_world_task(world_scout, market_phase)
    stock_task = build_stock_scan_task(stock_scanner, market_phase)
    crypto_task = build_crypto_scan_task(crypto_scanner, market_phase)
    rank_task = build_rank_task(
        ranker, market_phase, context_tasks=[world_task, stock_task, crypto_task]
    )

    crew = Crew(
        agents=[world_scout, stock_scanner, crypto_scanner, ranker],
        tasks=[world_task, stock_task, crypto_task, rank_task],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
        task_callback=task_callback,
    )

    try:
        result = crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        logger.error("Discovery crew failed: %s", exc)
        return None

    _LAST_USAGE = _usage_dict(getattr(result, "token_usage", None))

    shortlist = getattr(result, "pydantic", None)
    if not isinstance(shortlist, OpportunityShortlist):
        try:
            shortlist = OpportunityShortlist.model_validate_json(str(result))
        except Exception as exc:  # noqa: BLE001
            logger.error("No valid OpportunityShortlist from discovery: %s", exc)
            return None

    logger.info(
        "SHORTLIST | phase=%s macro=%s(%.2f) ideas=%s",
        shortlist.market_phase.value,
        shortlist.macro_bias.value,
        shortlist.macro_score,
        ", ".join(i.asset for i in shortlist.ideas) or "(none)",
    )
    return shortlist


def _circuit_breaker_tripped(day_open_equity: float) -> bool:
    """True if today's drawdown breached the daily max-loss limit."""
    equity = current_equity()
    loss = day_open_equity - equity
    limit = day_open_equity * settings.DAILY_MAX_LOSS_PCT
    if loss >= limit:
        logger.warning(
            "CIRCUIT BREAKER: day loss $%.2f >= limit $%.2f. Standing down.",
            loss,
            limit,
        )
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent paper trading bot.")
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Asset/ticker. Repeat for multiple, e.g. --asset AAPL --asset MSFT.",
    )
    parser.add_argument(
        "--phase",
        choices=[p.value for p in MarketPhase],
        default=None,
        help="Market phase. Omit to auto-detect from the live US market clock.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help=(
            "Auto-discover assets: run the 4-scout crew to scan live movers + "
            "macro, then trade the ranked shortlist. Ignores --asset."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print performance summary and exit.",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Poll open Binance brackets, realize resolved P&L, then exit.",
    )
    args = parser.parse_args()

    if args.reconcile:
        n = reconcile_open()
        logger.info("Reconciled %d open Binance bracket(s).", n)
        import json

        print(json.dumps(performance_summary(), indent=2))
        return 0

    if args.summary:
        import json

        print(json.dumps(performance_summary(), indent=2))
        return 0

    if not args.asset and not args.discover:
        parser.error("Provide --asset, or use --discover (or --summary).")

    # Validate env once, up front — only require keys the chosen providers need.
    required: list[str] = []
    if settings.LLM_PROVIDER == "anthropic":
        required.append("ANTHROPIC_API_KEY")
    if settings.RESEARCH_SOURCE == "perplexity":
        required.append("PERPLEXITY_API_KEY")
    try:
        if required:
            settings.require(*required)
    except EnvironmentError as exc:
        logger.error(str(exc))
        return 1
    logger.info(
        "Providers | llm=%s research=%s",
        settings.LLM_PROVIDER,
        settings.RESEARCH_SOURCE,
    )

    # Realize any live broker brackets that closed since last run before we measure.
    if settings.FILL_SOURCE in ("binance", "alpaca", "live"):
        reconcile_open()

    # Phase: use --phase if given, else auto-detect from the live market clock.
    phase = args.phase or current_market_phase().value
    if not args.phase:
        logger.info("Auto-detected market phase from US clock: %s", phase)

    day_open_equity = current_equity()
    logger.info(
        "Run start | equity=$%.2f | risk cap/trade=$%.2f | daily stop=%.0f%%",
        day_open_equity,
        settings.MAX_RISK_DOLLARS,
        settings.DAILY_MAX_LOSS_PCT * 100,
    )

    # Build the asset list. --discover replaces the manual list with the scouts'
    # ranked shortlist; otherwise we trade exactly what the user passed.
    if args.discover:
        shortlist = run_discovery(phase)
        log_usage(pop_last_usage())  # attribute scout tokens to the run ledger
        if shortlist is None or not shortlist.ideas:
            logger.warning("Discovery produced no tradable ideas — nothing to do.")
            assets = []
        else:
            assets = [i.asset for i in shortlist.ideas[: settings.MAX_CANDIDATES]]
            logger.info("Discovery shortlist → trading: %s", ", ".join(assets))
    else:
        assets = args.asset

    for asset in assets:
        if _circuit_breaker_tripped(day_open_equity):
            break
        ticket = run_cycle(asset, phase)
        usage = pop_last_usage()
        if ticket is None:
            log_usage(usage)  # crew still burned tokens even with no ticket
            logger.warning("Skipping execution for %s — no valid ticket.", asset)
            continue
        execute_ticket(ticket, market_phase=phase, usage=usage)

    print("\n=== PERFORMANCE SUMMARY ===")
    import json

    print(json.dumps(performance_summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
