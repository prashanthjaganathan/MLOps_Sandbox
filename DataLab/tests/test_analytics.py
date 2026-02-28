"""
Tests for src/pipeline/analytics.py

Uses Apache Beam's TestPipeline to run DoFns in a controlled environment.

Run with: python -m pytest tests/test_analytics.py -v
"""

import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from datetime import datetime

from src.pipeline.analytics import ComputeWindowAnalytics


def make_trade(symbol: str, price: float, volume: int, dollar_value: float | None = None) -> dict:
    return {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "timestamp": datetime(2024, 9, 1, 14, 30, 0),
        "conditions": [1],
        "dollar_value": dollar_value if dollar_value is not None else round(price * volume, 2),
        "is_block_trade": volume >= 10_000,
        "market_cap_tier": "MEGA",
    }


class TestComputeWindowAnalytics:
    def test_vwap_computed_correctly(self):
        """VWAP = sum(price * volume) / sum(volume)"""
        trades = [
            make_trade("AAPL", 230.0, 100),
            make_trade("AAPL", 232.0, 200),
        ]
        # VWAP = (230*100 + 232*200) / (100+200) = (23000 + 46400) / 300 = 231.33...
        expected_vwap = round((230.0 * 100 + 232.0 * 200) / 300, 2)

        with TestPipeline() as p:
            result = (
                p
                | beam.Create([("AAPL", trades)])
                | beam.WindowInto(beam.window.FixedWindows(30))
                | beam.ParDo(ComputeWindowAnalytics())
            )
            assert_that(
                result | beam.Map(lambda r: r["vwap"]),
                equal_to([expected_vwap]),
            )

    def test_block_trade_detection(self):
        """Trades with volume >= 10,000 should be counted as block trades."""
        trades = [
            make_trade("NVDA", 140.0, 5_000),   # not a block trade
            make_trade("NVDA", 141.0, 15_000),  # block trade
            make_trade("NVDA", 139.0, 20_000),  # block trade
        ]

        with TestPipeline() as p:
            result = (
                p
                | beam.Create([("NVDA", trades)])
                | beam.WindowInto(beam.window.FixedWindows(30))
                | beam.ParDo(ComputeWindowAnalytics())
            )
            assert_that(
                result | beam.Map(lambda r: r["block_trade_count"]),
                equal_to([2]),
            )

    def test_volatility_is_high_minus_low(self):
        trades = [
            make_trade("TSLA", 350.0, 100),
            make_trade("TSLA", 355.0, 100),
            make_trade("TSLA", 348.0, 100),
        ]
        expected_volatility = round(355.0 - 348.0, 2)

        with TestPipeline() as p:
            result = (
                p
                | beam.Create([("TSLA", trades)])
                | beam.WindowInto(beam.window.FixedWindows(30))
                | beam.ParDo(ComputeWindowAnalytics())
            )
            assert_that(
                result | beam.Map(lambda r: r["volatility"]),
                equal_to([expected_volatility]),
            )

    def test_empty_trades_yields_nothing(self):
        with TestPipeline() as p:
            result = (
                p
                | beam.Create([("AAPL", [])])
                | beam.WindowInto(beam.window.FixedWindows(30))
                | beam.ParDo(ComputeWindowAnalytics())
            )
            assert_that(result, equal_to([]))

    def test_price_open_close_are_first_last(self):
        trades = [
            make_trade("MSFT", 420.0, 100),
            make_trade("MSFT", 421.0, 100),
            make_trade("MSFT", 419.5, 100),
        ]

        with TestPipeline() as p:
            result = (
                p
                | beam.Create([("MSFT", trades)])
                | beam.WindowInto(beam.window.FixedWindows(30))
                | beam.ParDo(ComputeWindowAnalytics())
            )
            assert_that(
                result | beam.Map(lambda r: (r["price_open"], r["price_close"])),
                equal_to([(420.0, 419.5)]),
            )
