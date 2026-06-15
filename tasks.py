"""
tasks.py
========
The three sequential tasks that wire the agents into a pipeline:

    research_task   -> MarketScanReport   (Researcher, uses Perplexity tool)
    analysis_task   -> TradeSignal        (Analyst, consumes research)
    risk_task       -> ExecutionTicket    (Risk Manager, consumes signal)

CrewAI's `context=[...]` makes each task receive the prior task's output, so the
structured data flows 1 -> 2 -> 3. `output_pydantic` forces each step to conform
to the schemas in config.py — that is what guarantees the final ticket is valid
JSON the broker can trust.
"""

from __future__ import annotations

from crewai import Task

from config import (
    ExecutionTicket,
    MarketScanReport,
    OpportunityShortlist,
    TradeSignal,
    settings,
)


# --------------------------------------------------------------------------- #
# Discovery tasks — the 4 scouts run first, in parallel context, to produce the
# ranked shortlist the decision desk then trades one name at a time.
# --------------------------------------------------------------------------- #
def build_world_task(agent, market_phase: str) -> Task:
    return Task(
        description=(
            f"Use the World Macro Scanner for the '{market_phase}' phase. Read the "
            f"global risk backdrop and summarise the 3-5 dominant themes. End with "
            f"a single risk tilt: bias (bullish/bearish/neutral) and a -1..+1 risk "
            f"score (+1 risk-on, -1 risk-off). If the scan is degraded, say so and "
            f"set bias neutral, score 0."
        ),
        expected_output=(
            "A short macro brief: bullet themes, plus an explicit risk bias label "
            "and a -1..+1 risk score on the final line."
        ),
        agent=agent,
    )


def build_stock_scan_task(agent, market_phase: str) -> Task:
    return Task(
        description=(
            f"Use the Stock Movers Scanner for the '{market_phase}' phase to pull "
            f"the live stock/ETF movers. Return the ranked candidates exactly as "
            f"the tool gives them (asset, asset_class, raw_score, change_pct, "
            f"volume, reason). Do not invent tickers or prices."
        ),
        expected_output=(
            "A ranked list of stock/ETF candidate dicts straight from the scanner."
        ),
        agent=agent,
    )


def build_crypto_scan_task(agent, market_phase: str) -> Task:
    return Task(
        description=(
            f"Use the Crypto Movers Scanner for the '{market_phase}' phase to pull "
            f"the live crypto movers. Return the ranked candidates exactly as the "
            f"tool gives them (asset like 'BTC-USD', asset_class, raw_score, "
            f"change_pct, volume, reason). Do not invent coins or prices."
        ),
        expected_output=(
            "A ranked list of crypto candidate dicts straight from the scanner."
        ),
        agent=agent,
    )


def build_rank_task(agent, market_phase: str, context_tasks: list[Task]) -> Task:
    n = settings.MAX_CANDIDATES
    return Task(
        description=(
            f"You receive: (1) a macro risk brief, (2) a stock candidate list, and "
            f"(3) a crypto candidate list for the '{market_phase}' phase. Fuse them "
            f"into ONE ranked shortlist of the top {n} trade ideas.\n\n"
            f"Ranking rules:\n"
            f"  - Keep each idea's asset, asset_class, raw_score, change_pct, "
            f"volume, reason, source from the scanners. Do NOT fabricate values.\n"
            f"  - Prefer ideas whose raw_score sign agrees with the macro risk "
            f"tilt (risk-on favours positive scores, risk-off favours negative).\n"
            f"  - Prefer larger |raw_score| and real volume; break ties by macro "
            f"theme relevance.\n"
            f"  - Mix asset classes when quality is comparable; never return more "
            f"than {n} ideas.\n\n"
            f"You MAY call the Price & Technicals tool on a top candidate to "
            f"confirm it has a real, liquid price before ranking it (drop dead/"
            f"illiquid names). Set macro_bias and macro_score from the macro "
            f"brief. Populate themes. Emit ONLY a strict OpportunityShortlist JSON."
        ),
        expected_output=(
            f"A strict OpportunityShortlist JSON: market_phase, macro_bias, "
            f"macro_score, themes, and ideas (<= {n} TradeIdea entries, best first)."
        ),
        agent=agent,
        context=context_tasks,
        output_pydantic=OpportunityShortlist,
    )


def build_research_task(agent, asset: str, market_phase: str) -> Task:
    return Task(
        description=(
            f"Use the Perplexity Market Scanner tool to gather live intelligence "
            f"for asset '{asset}' during the '{market_phase}' phase. "
            f"Clean the results into structured facts: material headlines, macro "
            f"catalysts, an overall sentiment label and a -1..+1 score, and any "
            f"key price levels the news implies. Keep raw scout notes. If the "
            f"scan returns degraded=true, say so plainly and set sentiment "
            f"neutral with score 0."
        ),
        expected_output=(
            "A MarketScanReport with asset, market_phase, headlines, "
            "macro_catalysts, sentiment, sentiment_score, key_levels, raw_notes, "
            "and sources populated from the live scan."
        ),
        agent=agent,
        output_pydantic=MarketScanReport,
    )


def build_analysis_task(agent, asset: str, research_task: Task) -> Task:
    return Task(
        description=(
            f"Read the research brief for '{asset}'. Call the Price & Technicals "
            f"tool for '{asset}' to get real last price, ATR, RSI, and recent "
            f"support/resistance. Combine the sentiment and catalysts with those "
            f"real levels to reach ONE strict decision: BUY, SELL, or HOLD. "
            f"Provide a confidence 0..1 and a concise rationale. For BUY/SELL you "
            f"MUST provide suggested_entry (near last_price), suggested_stop "
            f"(use the tool's atr-based suggested_stop), and suggested_target "
            f"consistent with the direction (long: stop<entry<target; short: "
            f"target<entry<stop). When edge is unclear or research was degraded, "
            f"return HOLD."
        ),
        expected_output=(
            "A TradeSignal with asset, action, confidence, rationale, and "
            "(for BUY/SELL) suggested_entry, suggested_stop, suggested_target."
        ),
        agent=agent,
        context=[research_task],
        output_pydantic=TradeSignal,
    )


def build_risk_task(agent, asset: str, signal: TradeSignal) -> Task:
    """
    Size + risk-gate a NON-HOLD signal. The analyst's decision is embedded
    directly (no CrewAI context dependency) so this task runs in its own crew —
    HOLD signals are short-circuited upstream and never reach here.
    """
    cap = settings.STARTING_CAPITAL
    max_risk = settings.MAX_RISK_DOLLARS
    return Task(
        description=(
            f"You are the final gate before execution for '{asset}'. Account "
            f"capital is ${cap}. HARD RULE: risk on this trade must not exceed "
            f"${max_risk} ({settings.MAX_RISK_PCT:.0%} of capital).\n\n"
            f"The analyst's decision (RESPECT the action exactly — your job is "
            f"sizing, NOT direction; do NOT downgrade to HOLD):\n"
            f"  action={signal.action.value}\n"
            f"  suggested_entry={signal.suggested_entry}\n"
            f"  suggested_stop={signal.suggested_stop}\n"
            f"  suggested_target={signal.suggested_target}\n"
            f"  rationale={signal.rationale!r}\n\n"
            f"Call the Position Sizer tool with entry_price=suggested_entry, "
            f"stop_loss=suggested_stop, take_profit=suggested_target. Copy its "
            f"returned quantity, risk_dollars, risk_pct, reward_risk_ratio, "
            f"suggested_take_profit (use as take_profit), and capital_at_open "
            f"straight into the ticket — do NOT hand-calculate. The tool enforces "
            f"the ${max_risk} risk cap, no leverage, and a >=1.5 reward:risk "
            f"target. Keep entry_price=suggested_entry and stop_loss=suggested_stop.\n\n"
            f"Emit ONLY a strict JSON ExecutionTicket. No prose outside the JSON."
        ),
        expected_output=(
            "A strict ExecutionTicket JSON: asset, action, entry_price, "
            "stop_loss, take_profit, quantity, risk_dollars, risk_pct, "
            "reward_risk_ratio, capital_at_open, rationale, timestamp_utc. "
            "risk_dollars must be <= the hard cap."
        ),
        agent=agent,
        output_pydantic=ExecutionTicket,
    )
