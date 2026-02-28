"""
Tests for src/pipeline/parse.py

Run with: python -m pytest tests/test_parse.py -v
"""

import json
import pytest
from src.pipeline.parse import parse_trade, is_valid_trade


VALID_RAW = json.dumps({"s": "AAPL", "p": 227.31, "v": 100, "c": [1, 12], "t": 1725198451165})


class TestParseTrade:
    def test_parses_valid_message(self):
        result = parse_trade(VALID_RAW)
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 227.31
        assert result["volume"] == 100
        assert result["dollar_value"] == round(227.31 * 100, 2)
        assert result["conditions"] == [1, 12]

    def test_returns_none_on_invalid_json(self):
        assert parse_trade("not json at all") is None

    def test_returns_none_on_missing_field(self):
        # Missing "v" (volume)
        bad = json.dumps({"s": "AAPL", "p": 227.31, "t": 1725198451165})
        assert parse_trade(bad) is None

    def test_returns_none_on_wrong_type(self):
        bad = json.dumps({"s": "AAPL", "p": "not_a_float", "v": 100, "t": 1725198451165})
        assert parse_trade(bad) is None

    def test_dollar_value_computed_correctly(self):
        raw = json.dumps({"s": "NVDA", "p": 140.50, "v": 200, "t": 1725198451165})
        result = parse_trade(raw)
        assert result["dollar_value"] == round(140.50 * 200, 2)

    def test_conditions_defaults_to_empty_list(self):
        raw = json.dumps({"s": "TSLA", "p": 350.0, "v": 50, "t": 1725198451165})
        result = parse_trade(raw)
        assert result["conditions"] == []


class TestIsValidTrade:
    def test_valid_trade_passes(self):
        trade = parse_trade(VALID_RAW)
        assert is_valid_trade(trade) is True

    def test_none_fails(self):
        assert is_valid_trade(None) is False

    def test_zero_price_fails(self):
        trade = {"symbol": "AAPL", "price": 0.0, "volume": 100, "dollar_value": 0.0}
        assert is_valid_trade(trade) is False

    def test_negative_price_fails(self):
        trade = {"symbol": "AAPL", "price": -1.0, "volume": 100, "dollar_value": -100.0}
        assert is_valid_trade(trade) is False

    def test_zero_volume_fails(self):
        trade = {"symbol": "AAPL", "price": 230.0, "volume": 0, "dollar_value": 0.0}
        assert is_valid_trade(trade) is False
