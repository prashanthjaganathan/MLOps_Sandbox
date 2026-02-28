"""
Phase 2: Windowed analytics — VWAP, volume, volatility, block trades.

ComputeWindowAnalytics is a DoFn that receives (symbol, [trades]) from
GroupByKey and emits one analytics dict per window per symbol.

This is the core of why Beam exists. Try doing this with pandas on a
never-ending stream — you can't. Beam's windowing model makes it trivial.
"""

import apache_beam as beam


class ComputeWindowAnalytics(beam.DoFn):
    """
    For each window, for each stock, compute:

    - VWAP: volume-weighted average price — what institutional traders watch
    - Trade count and total shares
    - Total dollar volume (price × volume summed)
    - Price high, low, open (first), close (last)
    - Volatility: high − low within the window
    - Block trade count and volume (single trades ≥ 10,000 shares)

    Receives: (symbol, iterable_of_trades)
    Yields:   one dict per (symbol, window)
    """

    BLOCK_TRADE_THRESHOLD = 10_000

    def process(self, element, window=beam.DoFn.WindowParam):
        symbol, trades_iter = element
        trades = list(trades_iter)

        if not trades:
            return

        prices = [t["price"] for t in trades]
        volumes = [t["volume"] for t in trades]
        dollar_values = [t["dollar_value"] for t in trades]

        total_shares = sum(volumes)
        if total_shares == 0:
            return

        # VWAP: the price you'd pay if you bought proportionally to
        # how much everyone else was trading. This is the #1 metric
        # institutional traders use to evaluate execution quality.
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_shares

        block_trades = [t for t in trades if t["volume"] >= self.BLOCK_TRADE_THRESHOLD]

        yield {
            "symbol": symbol,
            "window_start": window.start.to_utc_datetime().isoformat(),
            "window_end": window.end.to_utc_datetime().isoformat(),
            "vwap": round(vwap, 2),
            "trade_count": len(trades),
            "total_shares": total_shares,
            "total_dollar_volume": round(sum(dollar_values), 2),
            "price_high": max(prices),
            "price_low": min(prices),
            "price_open": prices[0],
            "price_close": prices[-1],
            "volatility": round(max(prices) - min(prices), 2),
            "block_trade_count": len(block_trades),
            "block_trade_volume": sum(t["volume"] for t in block_trades),
        }
