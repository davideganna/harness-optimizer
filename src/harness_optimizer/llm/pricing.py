"""USD prices per million tokens, used for cost accounting.

Prices change; treat this table as a convenience and override it via `register_price`.
"""

from __future__ import annotations

import logging

from harness_optimizer.llm.base import Usage

log = logging.getLogger(__name__)

# model -> (input $/MTok, output $/MTok)
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "mock": (0.0, 0.0),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

# Providers whose inference costs nothing per token (local servers, test doubles).
FREE_PROVIDERS = {"ollama", "mock"}

_warned: set[str] = set()


def register_price(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    PRICES[model] = (input_per_mtok, output_per_mtok)


def cost_usd(model: str, usage: Usage, provider: str | None = None) -> float:
    if provider in FREE_PROVIDERS:
        return 0.0
    if model not in PRICES:
        if model not in _warned:
            log.warning("no price registered for %r; cost will be reported as 0", model)
            _warned.add(model)
        return 0.0
    inp, out = PRICES[model]
    return (
        usage.input_tokens * inp
        + usage.cache_read_tokens * inp * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * inp * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * out
    ) / 1_000_000
