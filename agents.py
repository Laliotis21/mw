"""
agents.py
=========
The three CrewAI agents, each backed by a Claude LLM.

    1. Lead Market Researcher (Data Scout)
       - Owns the Perplexity tool. Extracts + cleans web intel into facts.
    2. Quantitative Signal Analyst
       - Turns intel + price geometry into a strict BUY / SELL / HOLD.
    3. Chief Risk Officer (Risk Manager)
       - The gatekeeper. Hard 2% ($20) risk cap, position sizing, SL/TP.

The Risk Officer deliberately has NO tools and a low temperature: its job is
arithmetic discipline, not creativity. The Researcher runs warmer because it
summarizes messy prose.
"""

from __future__ import annotations

from crewai import Agent, LLM

from config import settings
from specs_loader import backstory_for
from tools import (
    crypto_scanner_tool,
    market_scanner_tool,
    position_sizer_tool,
    price_technicals_tool,
    stock_scanner_tool,
    world_events_tool,
)


def _llm(temperature: float) -> LLM:
    """
    Build a CrewAI LLM for the configured provider.

    CrewAI routes through LiteLLM, so the model id carries a provider prefix.
      - anthropic -> 'anthropic/<model>' (needs ANTHROPIC_API_KEY)
      - ollama    -> 'ollama/<model>'    (local server, no key, free)
    Temperature is the main knob we vary per role.
    """
    if settings.LLM_PROVIDER == "ollama":
        # Local model. No API key. base_url points at the Ollama daemon.
        return LLM(
            model=f"ollama/{settings.OLLAMA_MODEL}",
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    # Default: Claude via Anthropic.
    return LLM(
        model=f"anthropic/{settings.ANTHROPIC_MODEL}",
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=temperature,
    )


# --------------------------------------------------------------------------- #
# Discovery crew — 4 scouts that find WHAT to trade, feeding the decision desk.
# --------------------------------------------------------------------------- #
def build_world_scout() -> Agent:
    """World/Macro Scout — reads the global risk backdrop, picks no tickers."""
    return Agent(
        role="Global Macro Scout",
        goal=(
            "Read the live world/macro backdrop for the current phase and distil "
            "it into dominant themes plus one risk tilt: bias (bullish/bearish/"
            "neutral) and a -1..+1 risk score. Facts only, never pick tickers."
        ),
        backstory=backstory_for("world_scout"),
        llm=_llm(temperature=0.4),
        tools=[world_events_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )


def build_stock_scanner() -> Agent:
    """Stock Scanner — surfaces the day's biggest stock/ETF movers."""
    return Agent(
        role="Equity Universe Scanner",
        goal=(
            "Scan the live stock/ETF market for the strongest movers (gainers, "
            "losers, most active) and return a clean, ranked candidate list with "
            "the move and volume that flagged each. Do not decide direction."
        ),
        backstory=backstory_for("stock_scanner"),
        llm=_llm(temperature=0.2),
        tools=[stock_scanner_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )


def build_crypto_scanner() -> Agent:
    """Crypto Scanner — surfaces the biggest 24h crypto movers."""
    return Agent(
        role="Crypto Universe Scanner",
        goal=(
            "Scan the live crypto market for the biggest 24h movers by absolute "
            "price change (no stablecoins) and return a clean, ranked candidate "
            "list with the move and volume. Do not decide direction."
        ),
        backstory=backstory_for("crypto_scanner"),
        llm=_llm(temperature=0.2),
        tools=[crypto_scanner_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )


def build_opportunity_ranker() -> Agent:
    """Opportunity Ranker — fuses macro tilt + both scans into a top-N shortlist."""
    return Agent(
        role="Opportunity Ranker",
        goal=(
            f"Fuse the macro risk tilt with the stock and crypto candidate lists "
            f"into a single ranked shortlist of the top {settings.MAX_CANDIDATES} "
            f"trade ideas for this phase. Favour candidates whose direction aligns "
            f"with the macro tilt and that have real volume behind the move. Emit a "
            f"strict OpportunityShortlist. Pick names, never size or place trades."
        ),
        backstory=backstory_for("opportunity_ranker"),
        llm=_llm(temperature=0.2),
        tools=[price_technicals_tool],  # confirm top candidates are real + liquid
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )


def build_researcher() -> Agent:
    """Data Scout — gathers and cleans, never decides direction."""
    return Agent(
        role="Lead Market Researcher (Data Scout)",
        goal=(
            "Extract and clean live, source-backed market intelligence for the "
            "target asset and market phase. Output verifiable facts only — never "
            "trade recommendations."
        ),
        backstory=backstory_for("researcher"),  # full spec from specs/researcher.md
        llm=_llm(temperature=0.4),
        tools=[market_scanner_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=4,
    )


def build_analyst() -> Agent:
    """Quant Analyst — synthesizes intel into a strict directional signal."""
    return Agent(
        role="Quantitative Signal Analyst",
        goal=(
            "Synthesize the research brief with price/technical context into a "
            "single strict decision: BUY, SELL, or HOLD, with a confidence score "
            "and proposed entry, stop, and target levels."
        ),
        backstory=backstory_for("analyst"),  # full spec from specs/analyst.md
        llm=_llm(temperature=0.3),
        tools=[price_technicals_tool],  # anchor entry/stop/target to real levels
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )


def build_risk_officer() -> Agent:
    """Chief Risk Officer — the hard gate. Arithmetic, not opinion."""
    return Agent(
        role="Chief Risk Officer (Risk Manager)",
        goal=(
            f"Enforce capital preservation. Never allow more than "
            f"${settings.MAX_RISK_DOLLARS} "
            f"({settings.MAX_RISK_PCT:.0%}) of risk on a single trade. Compute "
            f"exact position size from entry and stop, set take-profit for a "
            f"minimum 1.5:1 reward:risk, and emit a strict JSON execution ticket."
        ),
        backstory=backstory_for("risk_officer"),  # full spec from specs/risk_officer.md
        llm=_llm(temperature=0.0),  # deterministic — this is arithmetic
        tools=[position_sizer_tool],  # exact, risk-capped sizing (no LLM math errors)
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )
