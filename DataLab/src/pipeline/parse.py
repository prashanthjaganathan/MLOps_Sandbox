"""
Parse and validate raw Finnhub trade JSON into clean pipeline dicts.

These functions are used as beam.Map / beam.Filter transforms.
They are pure functions with no side effects — easy to unit test.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_trade(raw_json: str) -> dict:
    """
    Parse a raw Finnhub trade JSON string into a clean dict.

    Returns None on parse failure so the caller can filter it out.
    Catches json.JSONDecodeError and KeyError — bad messages are logged
    and skipped, never allowed to crash the pipeline.
    """
    try:
        raw = json.loads(raw_json)
        price = float(raw["p"])
        volume = int(raw["v"])
        return {
            "symbol": str(raw["s"]),
            "price": price,
            "volume": volume,
            "timestamp": datetime.fromtimestamp(raw["t"] / 1000),
            "conditions": raw.get("c", []),
            "dollar_value": round(price * volume, 2),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse trade: {raw_json!r} — {e}")
        return None


def is_valid_trade(trade: dict | None) -> bool:
    """
    Hard filter: drop None results from parse_trade and any trade with
    non-positive price or volume. These represent data quality issues
    in the upstream feed.
    """
    if trade is None:
        return False
    return trade["price"] > 0 and trade["volume"] > 0
