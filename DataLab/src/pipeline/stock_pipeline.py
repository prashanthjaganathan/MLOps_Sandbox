"""
Main Apache Beam pipeline — stock analytics engine.

This file evolves through all phases. Each phase adds transforms on top
of the previous one. The runner (DirectRunner vs DataflowRunner) is the
only thing that changes between local and cloud execution.

Phase 1: Parse → Filter → Print  (basic ingestion, no windowing)
Phase 2: + Windowed analytics (30s, 5m, sliding)
Phase 3: + Cross-stock analysis (sector momentum, unusual volume)
Phase 4: + Anomaly detection (price spikes, volatility surges, block clusters)
Phase 5: + BigQuery / Pub/Sub output (GCP deployment)

Usage:
    # Phase 1 — local file, DirectRunner
    python -m src.pipeline.stock_pipeline --input trades.jsonl

    # Phase 2+ — with windowed analytics
    python -m src.pipeline.stock_pipeline --input trades.jsonl --enable-analytics

    # Phase 5 — Dataflow
    python -m src.pipeline.stock_pipeline \
        --runner DataflowRunner \
        --input-subscription projects/my-project/subscriptions/stock-trades \
        --output-table my-project:stocks.window_analytics \
        --project my-project --region us-east1 \
        --temp-location gs://my-bucket/temp \
        --streaming
"""

import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions

from src.pipeline.parse import parse_trade, is_valid_trade
from src.pipeline.enrich import enrich_trade
from src.pipeline.analytics import ComputeWindowAnalytics
from src.pipeline.cross_stock import DetectSectorMomentum, DetectUnusualVolume
from src.pipeline.anomaly import DetectPriceAnomalies
from src.output.console_sink import print_analytics, print_alert, print_sector_signal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Pipeline options
# ---------------------------------------------------------------------------

class StockPipelineOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input",
            default="trades.jsonl",
            help="Input: path to JSONL file (DirectRunner) or Pub/Sub subscription URI (DataflowRunner)",
        )
        parser.add_argument(
            "--output-table",
            default=None,
            help="BigQuery output table (project:dataset.table). Required for DataflowRunner.",
        )
        parser.add_argument(
            "--enable-analytics",
            action="store_true",
            default=True,
            help="Enable windowed analytics (Phase 2+)",
        )
        parser.add_argument(
            "--enable-cross-stock",
            action="store_true",
            default=True,
            help="Enable cross-stock analysis (Phase 3+)",
        )
        parser.add_argument(
            "--enable-anomaly",
            action="store_true",
            default=True,
            help="Enable anomaly detection (Phase 4+)",
        )


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def assign_event_time(trade: dict) -> beam.window.TimestampedValue:
    """
    Tell Beam: use the exchange timestamp as the event time.

    Without this, ReadFromText assigns no timestamp and every element
    lands in the GlobalWindow (bounds: -inf to +inf). When analytics.py
    calls window.start.to_utc_datetime(), it overflows because infinity
    can't be converted to a datetime.

    With this, Beam routes each trade to the correct 30s / 5m window
    based on WHEN THE TRADE ACTUALLY HAPPENED on the exchange, not when
    our pipeline processed it. That's event-time processing.
    """
    # trade["t"] is epoch milliseconds from Finnhub; Beam wants seconds
    return beam.window.TimestampedValue(trade, trade["timestamp"].timestamp())


def build_pipeline(p: beam.Pipeline, options: StockPipelineOptions) -> None:
    runner = p.options.view_as(StandardOptions).runner or "DirectRunner"
    custom = p.options.view_as(StockPipelineOptions)
    input_source = custom.input

    # ------------------------------------------------------------------
    # SOURCE: local file (DirectRunner) or Pub/Sub (DataflowRunner)
    # ------------------------------------------------------------------
    if "DataflowRunner" in runner and input_source.startswith("projects/"):
        raw = p | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=input_source).with_output_types(bytes)
        raw_strings = raw | "DecodeBytes" >> beam.Map(lambda b: b.decode("utf-8"))
    else:
        raw_strings = p | "ReadFromFile" >> beam.io.ReadFromText(input_source)

    # ------------------------------------------------------------------
    # PHASE 1: Parse, validate, enrich, assign event time
    # ------------------------------------------------------------------
    trades = (
        raw_strings
        | "ParseTrades" >> beam.Map(parse_trade)
        | "FilterValid" >> beam.Filter(is_valid_trade).with_output_types(dict)
        | "EnrichTrades" >> beam.Map(enrich_trade)
        | "AssignEventTime" >> beam.Map(assign_event_time)
    )

    # Phase 1 basic output — just confirm trades are flowing
    if not custom.enable_analytics:
        trades | "PrintRaw" >> beam.Map(
            lambda t: print(f"[TRADE] {t['symbol']} ${t['price']:.2f} x {t['volume']:,}")
        )
        return

    # ------------------------------------------------------------------
    # PHASE 2: Windowed analytics
    # ------------------------------------------------------------------

    # 30-second fixed windows — real-time snapshot
    analytics_30s = (
        trades
        | "Window30s" >> beam.WindowInto(beam.window.FixedWindows(30))
        | "KeyBySymbol30s" >> beam.Map(lambda t: (t["symbol"], t))
        | "GroupBySymbol30s" >> beam.GroupByKey()
        | "Analytics30s" >> beam.ParDo(ComputeWindowAnalytics())
        | "Tag30s" >> beam.Map(lambda x: {**x, "window_size": "30s"})
    )

    # 5-minute fixed windows — trend detection
    analytics_5m = (
        trades
        | "Window5m" >> beam.WindowInto(beam.window.FixedWindows(300))
        | "KeyBySymbol5m" >> beam.Map(lambda t: (t["symbol"], t))
        | "GroupBySymbol5m" >> beam.GroupByKey()
        | "Analytics5m" >> beam.ParDo(ComputeWindowAnalytics())
        | "Tag5m" >> beam.Map(lambda x: {**x, "window_size": "5m"})
    )

    # 5-minute sliding windows, recalculated every 30s — rolling average
    analytics_sliding = (
        trades
        | "WindowSliding" >> beam.WindowInto(
            beam.window.SlidingWindows(size=300, period=30)
        )
        | "KeyBySymbolSliding" >> beam.Map(lambda t: (t["symbol"], t))
        | "GroupBySymbolSliding" >> beam.GroupByKey()
        | "AnalyticsSliding" >> beam.ParDo(ComputeWindowAnalytics())
        | "TagSliding" >> beam.Map(lambda x: {**x, "window_size": "sliding_5m"})
    )

    # Print 30s and sliding to terminal; 5m used downstream for baselines
    analytics_30s | "Print30s" >> beam.Map(print_analytics)
    analytics_sliding | "PrintSliding" >> beam.Map(print_analytics)

    if not custom.enable_cross_stock:
        analytics_5m | "Print5m" >> beam.Map(print_analytics)
        return

    # ------------------------------------------------------------------
    # PHASE 3: Cross-stock analysis
    # ------------------------------------------------------------------

    # Sector momentum: group ALL 30s analytics into one element per window
    (
        analytics_30s
        | "KeyAllForSector" >> beam.Map(lambda x: ("all", x))
        | "GroupAllForSector" >> beam.GroupByKey()
        | "SectorMomentum" >> beam.ParDo(DetectSectorMomentum())
        | "PrintSector" >> beam.Map(print_sector_signal)
    )

    # Unusual volume: compare 30s against 5m baseline via side input
    five_min_baseline = analytics_5m | "KeyFor5mBaseline" >> beam.Map(
        lambda x: (x["symbol"], x)
    )
    five_min_dict = beam.pvalue.AsDict(five_min_baseline)

    analytics_30s_keyed = (
        analytics_30s
        | "KeyFor30sVolume" >> beam.Map(lambda x: (x["symbol"], x))
        | "GroupFor30sVolume" >> beam.GroupByKey()
    )

    unusual_volume_alerts = (
        analytics_30s_keyed
        | "UnusualVolume" >> beam.ParDo(DetectUnusualVolume(), five_min_baselines=five_min_dict)
    )
    unusual_volume_alerts | "PrintUnusualVolume" >> beam.Map(print_alert)

    if not custom.enable_anomaly:
        return

    # ------------------------------------------------------------------
    # PHASE 4: Anomaly detection
    # ------------------------------------------------------------------

    five_min_context = beam.pvalue.AsDict(
        analytics_5m | "KeyFor5mContext" >> beam.Map(lambda x: (x["symbol"], x))
    )

    anomaly_alerts = (
        analytics_30s
        | "DetectAnomalies" >> beam.ParDo(DetectPriceAnomalies(), five_min_context=five_min_context)
    )
    anomaly_alerts | "PrintAnomalies" >> beam.Map(print_alert)

    # ------------------------------------------------------------------
    # PHASE 5: BigQuery output (only when output table is configured)
    # ------------------------------------------------------------------
    if custom.output_table:
        from src.output.bigquery_sink import write_analytics_to_bq, write_alerts_to_bq

        merged_analytics = (
            (analytics_30s, analytics_5m, analytics_sliding)
            | "MergeAnalytics" >> beam.Flatten()
        )
        write_analytics_to_bq(merged_analytics, custom.output_table)

        merged_alerts = (
            (unusual_volume_alerts, anomaly_alerts)
            | "MergeAlerts" >> beam.Flatten()
        )
        write_alerts_to_bq(merged_alerts, custom.output_table.replace("window_analytics", "alerts"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(argv=None) -> None:
    options = PipelineOptions(argv)
    custom = options.view_as(StockPipelineOptions)

    logger.info(f"Starting stock analytics pipeline | input={custom.input}")

    with beam.Pipeline(options=options) as p:
        build_pipeline(p, custom)


if __name__ == "__main__":
    run()