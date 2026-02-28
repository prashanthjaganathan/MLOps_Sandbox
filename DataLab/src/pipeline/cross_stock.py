"""
Phase 3: Cross-stock analysis — sector momentum and unusual volume detection.

This is where single-stock pandas completely falls apart. To compare stocks
against each other within the same time window, you need to see ALL stocks
in the SAME window simultaneously. Beam's windowing + GroupByKey makes this
trivial. With a naive script, it's a race condition.

DetectSectorMomentum:
    Receives all 30s analytics for all stocks in a window (keyed by "all"),
    checks if the Mag-7 tech stocks are moving together or diverging.

DetectUnusualVolume:
    Compares each stock's current 30s volume against its 5-minute baseline
    using a Beam side input. Alerts when a stock is trading at 3x+ normal volume.
"""

import apache_beam as beam

TECH_STOCKS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"}

UNUSUAL_VOLUME_THRESHOLD = 3.0  # 3x normal volume triggers an alert


class DetectSectorMomentum(beam.DoFn):
    """
    Takes all 30-second window analytics (all stocks in the same window),
    filters to Mag-7 tech stocks, and determines sector direction.

    Direction logic:
    - BULLISH  : 5+ of 7 stocks have close > open
    - BEARISH  : 5+ of 7 stocks have close < open
    - MIXED    : neither threshold reached

    Also identifies the biggest mover (leader) by absolute % change.

    Receives: ("all", iterable_of_analytics_dicts)
    Yields:   one sector signal dict per window
    """

    BULLISH_THRESHOLD = 5
    BEARISH_THRESHOLD = 5

    def process(self, element, window=beam.DoFn.WindowParam):
        _, all_analytics_iter = element
        all_analytics = list(all_analytics_iter)

        tech = [a for a in all_analytics if a["symbol"] in TECH_STOCKS]

        if len(tech) < 3:
            return  # not enough tech stocks in this window to be meaningful

        up_count = sum(1 for a in tech if a["price_close"] > a["price_open"])
        down_count = len(tech) - up_count

        if up_count >= self.BULLISH_THRESHOLD:
            direction = "BULLISH"
        elif down_count >= self.BEARISH_THRESHOLD:
            direction = "BEARISH"
        else:
            direction = "MIXED"

        # Find the stock with the largest absolute % move this window
        def move_pct(a):
            if a["price_open"] == 0:
                return 0.0
            return (a["price_close"] - a["price_open"]) / a["price_open"]

        leader = max(tech, key=lambda a: abs(move_pct(a)))
        leader_pct = move_pct(leader) * 100

        yield {
            "window_start": window.start.to_utc_datetime().isoformat(),
            "window_end": window.end.to_utc_datetime().isoformat(),
            "sector": "TECH_MAG7",
            "direction": direction,
            "stocks_up": up_count,
            "stocks_down": down_count,
            "stocks_analyzed": len(tech),
            "leader": leader["symbol"],
            "leader_move_pct": round(leader_pct, 3),
            "total_dollar_volume": round(sum(a["total_dollar_volume"] for a in tech), 2),
        }


class DetectUnusualVolume(beam.DoFn):
    """
    Compare each stock's current 30s dollar volume against its 5-minute baseline.

    The 5-minute baseline is passed as a Beam side input (AsDict), so this DoFn
    can look up any symbol's baseline without a join or shuffle.

    Normalization: 5-min total dollar volume ÷ 10 = expected 30s volume.

    Alerts when: current_30s_volume / expected_30s_volume >= 3.0

    Receives: (symbol, [current_30s_analytics])
    Side input: {symbol: 5m_analytics_dict}
    Yields: alert dict when unusual volume detected
    """

    def process(self, element, five_min_baselines):
        symbol, current_iter = element
        current_list = list(current_iter)

        if not current_list:
            return

        current = current_list[0]
        baseline = five_min_baselines.get(symbol)

        if not baseline:
            return  # no 5m baseline yet (first window)

        # Normalize 5-min volume to 30-second equivalent
        expected_30s_volume = baseline["total_dollar_volume"] / 10

        if expected_30s_volume <= 0:
            return

        volume_ratio = current["total_dollar_volume"] / expected_30s_volume

        if volume_ratio >= UNUSUAL_VOLUME_THRESHOLD:
            yield {
                "alert_type": "UNUSUAL_VOLUME",
                "type": "UNUSUAL_VOLUME",
                "symbol": symbol,
                "window_start": current["window_start"],
                "current_volume": current["total_dollar_volume"],
                "normal_volume": round(expected_30s_volume, 2),
                "volume_ratio": round(volume_ratio, 1),
                "message": (
                    f"📊 {symbol} UNUSUAL VOLUME: {volume_ratio:.1f}x normal  "
                    f"(${current['total_dollar_volume']:,.0f} vs "
                    f"normal ${expected_30s_volume:,.0f})"
                ),
            }
