"""
Tests for src/pipeline/anomaly.py

Run with: python -m pytest tests/test_anomaly.py -v
"""

import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to, is_empty

from src.pipeline.anomaly import DetectPriceAnomalies


def make_analytics(
    symbol: str,
    vwap: float,
    volatility: float,
    block_trade_count: int = 0,
    block_trade_volume: int = 0,
) -> dict:
    return {
        "symbol": symbol,
        "window_start": "2024-09-01T14:30:00",
        "window_end": "2024-09-01T14:30:30",
        "vwap": vwap,
        "volatility": volatility,
        "block_trade_count": block_trade_count,
        "block_trade_volume": block_trade_volume,
        "trade_count": 100,
        "total_shares": 10000,
        "total_dollar_volume": vwap * 10000,
        "price_high": vwap + volatility / 2,
        "price_low": vwap - volatility / 2,
        "price_open": vwap,
        "price_close": vwap,
    }


class TestDetectPriceAnomalies:
    def _run(self, current: dict, baseline: dict) -> list:
        """Helper: run DetectPriceAnomalies and collect results."""
        results = []
        dofn = DetectPriceAnomalies()
        five_min_context = {current["symbol"]: baseline}
        for alert in dofn.process(current, five_min_context=five_min_context):
            results.append(alert)
        return results

    def test_price_spike_detected(self):
        current = make_analytics("AAPL", vwap=232.0, volatility=0.5)
        baseline = make_analytics("AAPL", vwap=230.0, volatility=5.0)
        # Change: (232 - 230) / 230 = 0.0087 > 0.005 threshold
        alerts = self._run(current, baseline)
        types = [a["type"] for a in alerts]
        assert "PRICE_SPIKE" in types

    def test_price_drop_detected(self):
        current = make_analytics("AAPL", vwap=228.0, volatility=0.5)
        baseline = make_analytics("AAPL", vwap=230.0, volatility=5.0)
        # Change: (228 - 230) / 230 = -0.0087 < -0.005 threshold
        alerts = self._run(current, baseline)
        types = [a["type"] for a in alerts]
        assert "PRICE_DROP" in types

    def test_no_alert_within_threshold(self):
        current = make_analytics("AAPL", vwap=230.5, volatility=0.5)
        baseline = make_analytics("AAPL", vwap=230.0, volatility=5.0)
        # Change: 0.5/230 = 0.0022 < 0.005 — no alert
        alerts = self._run(current, baseline)
        price_alerts = [a for a in alerts if a["type"] in ("PRICE_SPIKE", "PRICE_DROP")]
        assert len(price_alerts) == 0

    def test_volatility_surge_detected(self):
        # baseline_vol_30s = 9.0 / 10 = 0.9
        # current volatility = 4.0 → ratio = 4.0 / 0.9 = 4.4 > 3.0
        current = make_analytics("NVDA", vwap=140.0, volatility=4.0)
        baseline = make_analytics("NVDA", vwap=140.0, volatility=9.0)
        alerts = self._run(current, baseline)
        types = [a["type"] for a in alerts]
        assert "VOLATILITY_SURGE" in types

    def test_block_trade_cluster_detected(self):
        current = make_analytics("TSLA", vwap=350.0, volatility=1.0, block_trade_count=3, block_trade_volume=45000)
        baseline = make_analytics("TSLA", vwap=350.0, volatility=10.0)
        alerts = self._run(current, baseline)
        types = [a["type"] for a in alerts]
        assert "BLOCK_TRADE_CLUSTER" in types

    def test_no_alert_without_baseline(self):
        current = make_analytics("AAPL", vwap=232.0, volatility=0.5)
        dofn = DetectPriceAnomalies()
        alerts = list(dofn.process(current, five_min_context={}))
        assert alerts == []

    def test_multiple_alerts_can_fire_simultaneously(self):
        # Big price spike + high volatility + block cluster all at once
        current = make_analytics("NVDA", vwap=145.0, volatility=5.0, block_trade_count=5, block_trade_volume=80000)
        baseline = make_analytics("NVDA", vwap=140.0, volatility=9.0)
        alerts = self._run(current, baseline)
        types = {a["type"] for a in alerts}
        assert "PRICE_SPIKE" in types
        assert "VOLATILITY_SURGE" in types
        assert "BLOCK_TRADE_CLUSTER" in types
