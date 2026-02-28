"""
StockTrade dataclass — the boundary object between raw Finnhub JSON and the pipeline.

Used only at the ingest boundary. Inside the Beam pipeline, everything is plain dicts.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StockTrade:
    symbol: str          # "AAPL"
    price: float         # 227.31
    volume: int          # 100 shares
    timestamp: datetime  # when the trade happened (event time)
    conditions: list     # Finnhub trade condition codes e.g. [1, 12]
    dollar_value: float  # price * volume, computed at parse time

    @classmethod
    def from_finnhub(cls, raw: dict) -> "StockTrade":
        """Parse a raw Finnhub trade dict into a StockTrade."""
        price = float(raw["p"])
        volume = int(raw["v"])
        return cls(
            symbol=str(raw["s"]),
            price=price,
            volume=volume,
            timestamp=datetime.fromtimestamp(raw["t"] / 1000),
            conditions=raw.get("c", []),
            dollar_value=round(price * volume, 2),
        )

    def to_dict(self) -> dict:
        """Convert to a plain dict for use inside the Beam pipeline."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "conditions": self.conditions,
            "dollar_value": self.dollar_value,
        }
