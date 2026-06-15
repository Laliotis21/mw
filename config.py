"""
config.py
=========
Central configuration + structured-output schemas for the multi-agent trading bot.

Two responsibilities:
  1. Load + validate environment (API keys, capital, risk limits).
  2. Define the Pydantic contracts that flow agent -> agent:
        MarketScanReport  (Researcher output)
        TradeSignal       (Analyst output)
        ExecutionTicket   (Risk Manager output -> broker)

Keeping the schemas here means every module imports one source of truth, so a
field rename can never silently desync the pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

# --------------------------------------------------------------------------- #
# Logging — configure once, import-time, so every module shares the handler.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trading_bot")


# --------------------------------------------------------------------------- #
# Settings — fail fast if a required key is missing.
# --------------------------------------------------------------------------- #
class Settings:
    """Validated runtime settings pulled from the environment."""

    def __init__(self) -> None:
        self.ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")

        # Broker (paper trading). Optional — execution.py runs fully simulated
        # if these are absent, but we surface them for a future live swap.
        self.ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
        self.ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
        self.ALPACA_BASE_URL: str = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        # Binance (crypto paper trading). Spot Testnet by default — real exchange
        # engine, fake balances, zero real money. Flip BINANCE_TESTNET=false to go
        # live (real funds; you accept the risk). Keys from testnet.binance.vision.
        self.BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
        self.BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
        self.BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        self.BINANCE_BASE_URL: str = os.getenv(
            "BINANCE_BASE_URL",
            "https://testnet.binance.vision" if self.BINANCE_TESTNET
            else "https://api.binance.com",
        )

        # Capital + risk.
        self.STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "1000"))
        self.MAX_RISK_PCT: float = float(os.getenv("MAX_RISK_PCT", "0.02"))
        self.DAILY_MAX_LOSS_PCT: float = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.05"))

        # Models.
        self.ANTHROPIC_MODEL: str = os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )
        self.PERPLEXITY_MODEL: str = os.getenv("PERPLEXITY_MODEL", "sonar")

        # Provider switches — let us test agents for free, no paid keys.
        #   LLM_PROVIDER:    'anthropic' (Claude) | 'ollama' (local, free)
        #   RESEARCH_SOURCE: 'perplexity' (paid) | 'mock' (free, canned data)
        # Decision engine: 'rules' (deterministic candle strategy, free, no LLM)
        # or 'llm' (CrewAI/Claude desk — worth it once live news feeds in).
        self.DECISION_ENGINE: str = os.getenv("DECISION_ENGINE", "rules").lower()
        # News/catalyst layer (free, via Alpaca News). USE_NEWS gates it on/off;
        # NEWS_SENTIMENT = 'keyword' (free) or 'llm' (1 cheap Haiku call/ticker).
        self.USE_NEWS: bool = os.getenv("USE_NEWS", "true").lower() == "true"
        self.NEWS_SENTIMENT: str = os.getenv("NEWS_SENTIMENT", "keyword").lower()
        # Autonomous scheduler (scheduler.py): minutes between auto runs.
        self.AUTORUN_INTERVAL_MIN: int = int(os.getenv("AUTORUN_INTERVAL_MIN", "30"))
        # Background live-price poller refresh (seconds) for OPEN POSITIONS.
        self.PRICE_POLL_SEC: int = int(os.getenv("PRICE_POLL_SEC", "2"))
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
        self.RESEARCH_SOURCE: str = os.getenv("RESEARCH_SOURCE", "perplexity").lower()
        # FILL_SOURCE: how paper fills resolve.
        #   yfinance -> real price bars (free)  |  coinflip -> random TP/SL
        #   binance  -> place real orders on Binance Spot Testnet (crypto only;
        #               stocks fall back to yfinance). Bracket = market + OCO.
        self.FILL_SOURCE: str = os.getenv("FILL_SOURCE", "yfinance").lower()
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.OLLAMA_BASE_URL: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

        # Discovery / scanning. The 4 scout agents scan the live market for
        # movers, then hand a shortlist to the decision desk.
        #   SCAN_PER_SOURCE: candidates pulled per source (per screener / coin page)
        #   MAX_CANDIDATES:  top-N shortlisted ideas the desk actually trades
        self.SCAN_PER_SOURCE: int = int(os.getenv("SCAN_PER_SOURCE", "10"))
        self.MAX_CANDIDATES: int = int(os.getenv("MAX_CANDIDATES", "3"))

        # Derived: absolute dollar risk cap per trade. The whole risk engine
        # keys off this single number ($20 on a $1000 / 2% book).
        self.MAX_RISK_DOLLARS: float = round(
            self.STARTING_CAPITAL * self.MAX_RISK_PCT, 2
        )

    def require(self, *keys: str) -> None:
        """Raise if any named setting is empty. Call before a live cycle."""
        missing = [k for k in keys if not getattr(self, k, None)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Copy .env.example -> .env and fill them in."
            )


settings = Settings()


def cap_quantity(qty: float, price: float, capital: float) -> float:
    """
    Clamp position size so notional (qty * price) never exceeds available
    capital. Spot has no leverage — a risk-based qty from a tight stop can demand
    more cash than the book holds, which the exchange rejects. Caps to all-in.
    """
    if price <= 0 or capital <= 0:
        return qty
    return min(qty, capital / price)


# --------------------------------------------------------------------------- #
# Enums — constrain the free-text surface so agents can't invent values.
# --------------------------------------------------------------------------- #
class MarketPhase(str, Enum):
    PRE_MARKET = "pre_market"
    OPEN = "open"
    MID_DAY = "mid_day"
    CLOSE = "close"


def current_market_phase() -> "MarketPhase":
    """
    Detect the US-equity session phase from the wall clock (New York, DST-aware)
    so the bot picks its own phase instead of the user choosing. Boundaries (ET):
        04:00-09:30 pre_market | 09:30-11:00 open | 11:00-15:00 mid_day
        15:00-16:00 close      | otherwise (overnight) -> pre_market (next session)
    Crypto trades 24/7, but the same intraday rhythm is a fine proxy.
    """
    from datetime import time as _t
    from zoneinfo import ZoneInfo

    try:
        et = datetime.now(ZoneInfo("America/New_York")).time()
    except Exception:  # noqa: BLE001 — tz db missing; fall back to a sane default
        return MarketPhase.MID_DAY
    if _t(4, 0) <= et < _t(9, 30):
        return MarketPhase.PRE_MARKET
    if _t(9, 30) <= et < _t(11, 0):
        return MarketPhase.OPEN
    if _t(11, 0) <= et < _t(15, 0):
        return MarketPhase.MID_DAY
    if _t(15, 0) <= et < _t(16, 0):
        return MarketPhase.CLOSE
    return MarketPhase.PRE_MARKET


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AssetClass(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    FX = "fx"


# --------------------------------------------------------------------------- #
# 1) MarketScanReport — Researcher -> Analyst
# --------------------------------------------------------------------------- #
class MarketScanReport(BaseModel):
    """Cleaned web intelligence for one asset at one market phase."""

    asset: str
    market_phase: MarketPhase
    timestamp_utc: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    headlines: List[str] = Field(default_factory=list, description="Material news.")
    macro_catalysts: List[str] = Field(
        default_factory=list, description="Macro events that move the tape."
    )
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = Field(
        0.0, ge=-1.0, le=1.0, description="-1 max bearish .. +1 max bullish."
    )
    key_levels: List[float] = Field(
        default_factory=list, description="Support/resistance the news implies."
    )
    raw_notes: str = Field("", description="Unstructured scout commentary.")
    sources: List[str] = Field(default_factory=list)

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def _fill_ts(cls, v):
        return v or datetime.now(timezone.utc).isoformat()

    @field_validator("key_levels", mode="before")
    @classmethod
    def _coerce_levels(cls, v):
        """LLMs often emit levels as {'value': 124.3, 'type': 'support'} or
        strings. Flatten anything to a plain list of floats; drop junk."""
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        out: List[float] = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("value") or item.get("price") or item.get("level")
            try:
                if item is not None:
                    out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out


# --------------------------------------------------------------------------- #
# 2) TradeSignal — Analyst -> Risk Manager
# --------------------------------------------------------------------------- #
class TradeSignal(BaseModel):
    """Directional decision + the price geometry the analyst proposes."""

    asset: str
    action: Action
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    suggested_entry: Optional[float] = Field(None, gt=0)
    suggested_stop: Optional[float] = Field(None, gt=0)
    suggested_target: Optional[float] = Field(None, gt=0)
    time_horizon: str = Field("intraday", description="intraday | swing")

    @model_validator(mode="after")
    def _check_actionable(self) -> "TradeSignal":
        """A BUY/SELL must carry entry + stop, else it's not actionable."""
        if self.action in (Action.BUY, Action.SELL):
            if self.suggested_entry is None or self.suggested_stop is None:
                raise ValueError(
                    f"{self.action} signal requires suggested_entry and suggested_stop."
                )
        return self


# --------------------------------------------------------------------------- #
# 3) ExecutionTicket — Risk Manager -> broker (final, strict)
# --------------------------------------------------------------------------- #
class ExecutionTicket(BaseModel):
    """
    The only object the broker accepts. Risk Manager is the sole author.
    Invariants enforced here are the last line of defence before money moves:
      - dollar risk never exceeds the hard cap
      - stop sits on the correct side of entry for the direction
    """

    asset: str
    action: Action
    # ge=0 (not gt=0): a HOLD ticket legitimately carries 0/echoed prices. The
    # >0 requirement is enforced for BUY/SELL in _check_geometry instead, so a
    # HOLD no longer fails validation and vanishes before it can be logged.
    entry_price: float = Field(..., ge=0)
    stop_loss: float = Field(..., ge=0)
    take_profit: float = Field(..., ge=0)
    quantity: float = Field(..., ge=0, description="Units/shares. 0 == stand down.")
    risk_dollars: float = Field(..., ge=0, description="Capital at risk to the stop.")
    risk_pct: float = Field(..., ge=0)
    reward_risk_ratio: float = Field(..., ge=0)
    capital_at_open: float
    rationale: str
    timestamp_utc: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def _fill_ts(cls, v):
        # LLMs love emitting null here — backfill so validation never trips.
        return v or datetime.now(timezone.utc).isoformat()

    @field_validator("risk_dollars")
    @classmethod
    def _risk_cap(cls, v: float) -> float:
        if v > settings.MAX_RISK_DOLLARS + 0.01:  # cent tolerance for rounding
            raise ValueError(
                f"risk_dollars ${v} exceeds hard cap ${settings.MAX_RISK_DOLLARS}."
            )
        return v

    @model_validator(mode="after")
    def _check_geometry(self) -> "ExecutionTicket":
        if self.action == Action.HOLD:
            return self
        # BUY/SELL must carry real positive prices (HOLD is exempt above).
        if not (self.entry_price > 0 and self.stop_loss > 0 and self.take_profit > 0):
            raise ValueError(f"{self.action} requires positive entry/stop/target.")
        if self.action == Action.BUY:
            # Long: stop below entry, target above.
            if not (self.stop_loss < self.entry_price < self.take_profit):
                raise ValueError("BUY requires stop < entry < target.")
        elif self.action == Action.SELL:
            # Short: stop above entry, target below.
            if not (self.take_profit < self.entry_price < self.stop_loss):
                raise ValueError("SELL requires target < entry < stop.")
        return self


# --------------------------------------------------------------------------- #
# Discovery layer — the 4 scout agents feed these into the decision desk.
# --------------------------------------------------------------------------- #
class TradeIdea(BaseModel):
    """One scanner-surfaced candidate. Cheap pre-screen, not a decision."""

    asset: str
    asset_class: AssetClass
    raw_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Scanner bias, -1 bearish .. +1 bullish."
    )
    change_pct: Optional[float] = Field(None, description="Intraday/24h % move.")
    volume: Optional[float] = None
    reason: str = Field("", description="Why the scanner flagged it.")
    source: str = Field("", description="Which scanner/source produced it.")


class OpportunityShortlist(BaseModel):
    """
    Ranker output: the macro backdrop plus the top-N candidate trades the desk
    should deep-analyze this cycle. This is the 'signal' the main desk consumes.
    """

    market_phase: MarketPhase
    macro_bias: Sentiment = Sentiment.NEUTRAL
    macro_score: float = Field(
        0.0, ge=-1.0, le=1.0, description="World/macro risk tilt, -1..+1."
    )
    themes: List[str] = Field(
        default_factory=list, description="Dominant market themes right now."
    )
    ideas: List[TradeIdea] = Field(
        default_factory=list, description="Ranked candidates, best first."
    )
    generated_utc: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("generated_utc", mode="before")
    @classmethod
    def _fill_ts(cls, v):
        return v or datetime.now(timezone.utc).isoformat()
