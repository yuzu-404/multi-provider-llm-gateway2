
from decimal import Decimal, ROUND_HALF_UP


class UnsupportedModelError(ValueError):
    """Raised when a model has no configured pricing."""


# USD per 1,000,000 tokens.
# Prices must be verified against official provider pricing
# before using this application for real requests.
MODEL_PRICING: dict[str, dict[str, Decimal]] = {
    "gpt-4o-mini": {
        "input": Decimal("0.15"),
        "output": Decimal("0.60"),
    },
    "gpt-4o": {
        "input": Decimal("2.50"),
        "output": Decimal("10.00"),
    },
    "claude-haiku-5-5": {
        "input": Decimal("0.10"),
        "output": Decimal("0.50"),
    },
    "claude-sonnet-5-5": {
        "input": Decimal("2.00"),
        "output": Decimal("10.00"),
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Calculate request cost in USD to six decimal places."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    if model not in MODEL_PRICING:
        raise UnsupportedModelError(
            f"No pricing configured for model: {model}"
        )

    rates = MODEL_PRICING[model]
    million = Decimal("1000000")

    cost = (
        Decimal(input_tokens) * rates["input"] / million
        + Decimal(output_tokens) * rates["output"] / million
    )

    return cost.quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )
