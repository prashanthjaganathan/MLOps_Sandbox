"""
Phase 4: Anomaly detection — price spikes, volatility surges, block trade clusters.

DetectPriceAnomalies compares each 30-second window's analytics against the
5-minute baseline (passed as a side input) and emits structured alerts.

Three alert types:
  PRICE_SPIKE / PRICE_DROP  : VWAP moved > 0.5% vs 5-min baseline
  VOLATILITY_SURGE          : window volatility > 3x normalized 5-min volatility
  BLOCK_TRADE_CLUSTER       : 3+ block trades in a single 30s window
"""

import apache_beam as beam


class DetectPriceAnomalies(beam.DoFn):
    """
    Receives each 30s analytics record and the 5-min baseline as a side input.
    Emits zero or more alert dicts per record.

    Thresholds are class constants — easy to tune without touching pipeline logic.
    """

    PRICE_THRESHOLD = 0.005        # 0.5% VWAP deviation triggers spike/drop alert
    VOLATILITY_MULTIPLIER = 3.0    # 3x normalized 5-min volatility triggers surge alert
    BLOCK_CLUSTER_THRESHOLD = 3    # 3+ block trades in one 30s window triggers cluster alert

    def process(self, element, five_min_context):
        current = element
        symbol = current["symbol"]
        baseline = five_min_context.get(symbol)

        if not baseline:
            return  # no 5m baseline yet; skip until we have enough history

        alerts = []

        # --- Price spike / drop ---
        if baseline["vwap"] > 0:
            price_change = (current["vwap"] - baseline["vwap"]) / baseline["vwap"]
            if abs(price_change) > self.PRICE_THRESHOLD:
                alert_type = "PRICE_SPIKE" if price_change > 0 else "PRICE_DROP"
                emoji = "🚀" if price_change > 0 else "📉"
                alerts.append({
                    "type": alert_type,
                    "symbol": symbol,
                    "change_pct": round(price_change * 100, 3),
                    "current_vwap": current["vwap"],
                    "baseline_vwap": baseline["vwap"],
                    "message": (
                        f"{emoji} {symbol} {alert_type}: {price_change*100:+.3f}%  "
                        f"(VWAP ${current['vwap']:,.2f} vs 5m ${baseline['vwap']:,.2f})"
                    ),
                })

        # --- Volatility surge ---
        # Normalize 5-min volatility to a 30s equivalent for fair comparison
        baseline_vol_30s = baseline["volatility"] / 10
        if baseline_vol_30s > 0:
            vol_ratio = current["volatility"] / baseline_vol_30s
            if vol_ratio > self.VOLATILITY_MULTIPLIER:
                alerts.append({
                    "type": "VOLATILITY_SURGE",
                    "symbol": symbol,
                    "vol_ratio": round(vol_ratio, 1),
                    "current_volatility": current["volatility"],
                    "baseline_volatility": round(baseline_vol_30s, 4),
                    "message": (
                        f"⚡ {symbol} VOLATILITY SURGE: {vol_ratio:.1f}x normal  "
                        f"(range ${current['volatility']:.2f} vs normal ${baseline_vol_30s:.2f})"
                    ),
                })

        # --- Block trade cluster ---
        if current["block_trade_count"] >= self.BLOCK_CLUSTER_THRESHOLD:
            alerts.append({
                "type": "BLOCK_TRADE_CLUSTER",
                "symbol": symbol,
                "block_count": current["block_trade_count"],
                "block_volume": current["block_trade_volume"],
                "message": (
                    f"🐋 {symbol} WHALE ACTIVITY: "
                    f"{current['block_trade_count']} block trades, "
                    f"{current['block_trade_volume']:,} shares in 30s"
                ),
            })

        for alert in alerts:
            alert["window_start"] = current["window_start"]
            alert["window_end"] = current["window_end"]
            yield alert
