"""
Enrich parsed trades with derived fields before windowing.

Runs as a beam.Map transform after parse/filter, before any windowing.
"""

# Dollar volume tiers — used for cross-stock comparisons and alerts
MARKET_CAP_TIERS = {
    "MEGA": {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"},
}

BLOCK_TRADE_THRESHOLD = 10_000  # shares — institutional order size


def enrich_trade(trade: dict) -> dict:
    """
    Add derived fields to a parsed trade dict.

    Fields added:
    - is_block_trade: True if volume >= 10,000 shares
    - market_cap_tier: "MEGA" for Mag-7, "OTHER" for everything else
    """
    return {
        **trade,
        "is_block_trade": trade["volume"] >= BLOCK_TRADE_THRESHOLD,
        "market_cap_tier": "MEGA" if trade["symbol"] in MARKET_CAP_TIERS["MEGA"] else "OTHER",
    }
