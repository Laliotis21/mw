"""
usage.py
========
LLM token-cost accounting. Converts CrewAI's per-crew token_usage into a dollar
figure using current Anthropic list prices, so the dashboard can show real
$/trade and total API spend.

Local providers (ollama) cost $0 — tokens are still counted (real evidence of
load), the price is just zero. Switch LLM_PROVIDER=anthropic and the same plumbing
reports actual dollars.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("trading_bot")

# USD per 1,000,000 tokens: (input, output). Source: Anthropic list pricing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

CACHED_READ_FACTOR = 0.1  # cached input billed at ~0.1x the input rate


def normalize_model(model: str) -> str:
    """'anthropic/claude-opus-4-8' / 'claude-opus-4-8-20250101' -> 'claude-opus-4-8'."""
    m = (model or "").split("/")[-1].strip().lower()
    if m in PRICING:
        return m
    # Drop a trailing -YYYYMMDD date snapshot if present.
    parts = m.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[0]
    return m


def is_local(model: str) -> bool:
    """True for free local providers (ollama, llama.cpp, etc.) — never billed."""
    m = (model or "").lower()
    return "ollama" in m or "llama.cpp" in m or m.startswith("local/")


def price_for(model: str) -> tuple[float, float] | None:
    """(input, output) $/1M for a model id, or None if not a priced (paid) model."""
    return PRICING.get(normalize_model(model))


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Dollar cost for one usage bundle. Returns 0.0 for unpriced models (e.g. a
    local ollama model) so free runs read as $0 while still logging tokens.
    `cached_tokens` is the cache-read subset of `prompt_tokens` (billed at 0.1x).
    """
    price = price_for(model)
    if price is None:
        # A local model is genuinely free; an unknown PAID model would silently
        # under-report spend — warn loudly so the ledger isn't quietly wrong.
        if not is_local(model):
            logger.warning(
                "No price for model %r — billing it as $0. Add it to usage.PRICING "
                "or the API-spend total will be understated.", model,
            )
        return 0.0
    in_rate, out_rate = price
    # CrewAI reports `prompt_tokens` (Anthropic input_tokens = UNCACHED input)
    # and `cached_tokens` (cache_read_input_tokens) as DISJOINT counts — cached
    # is NOT a subset of prompt_tokens. Bill uncached at full rate, cache reads
    # at 0.1x, and add them. (Cache-write 1.25x premium isn't surfaced
    # separately by CrewAI, so it rides in prompt_tokens at ~full rate.)
    cached = max(0, cached_tokens)
    cost = (
        prompt_tokens * in_rate
        + cached * in_rate * CACHED_READ_FACTOR
        + completion_tokens * out_rate
    ) / 1_000_000
    return round(cost, 6)
