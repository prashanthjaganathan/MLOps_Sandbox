"""
Phase 5: BigQuery sink — writes window analytics and alerts to BigQuery tables.

Used only when --output-table is set (DataflowRunner / GCP deployment).
The DirectRunner path skips this entirely and uses console_sink.py instead.
"""

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition


# ---------------------------------------------------------------------------
# BigQuery table schemas
# ---------------------------------------------------------------------------

ANALYTICS_SCHEMA = {
    "fields": [
        {"name": "symbol", "type": "STRING", "mode": "REQUIRED"},
        {"name": "window_start", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "window_end", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "window_size", "type": "STRING", "mode": "REQUIRED"},
        {"name": "vwap", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "trade_count", "type": "INT64", "mode": "REQUIRED"},
        {"name": "total_shares", "type": "INT64", "mode": "REQUIRED"},
        {"name": "total_dollar_volume", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "price_high", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "price_low", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "price_open", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "price_close", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "volatility", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "block_trade_count", "type": "INT64", "mode": "REQUIRED"},
        {"name": "block_trade_volume", "type": "INT64", "mode": "REQUIRED"},
    ]
}

ALERTS_SCHEMA = {
    "fields": [
        {"name": "type", "type": "STRING", "mode": "REQUIRED"},
        {"name": "symbol", "type": "STRING", "mode": "REQUIRED"},
        {"name": "window_start", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "message", "type": "STRING", "mode": "REQUIRED"},
        {"name": "change_pct", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "current_vwap", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "baseline_vwap", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "vol_ratio", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "volume_ratio", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "block_count", "type": "INT64", "mode": "NULLABLE"},
        {"name": "block_volume", "type": "INT64", "mode": "NULLABLE"},
    ]
}


# ---------------------------------------------------------------------------
# Sink helpers
# ---------------------------------------------------------------------------

def _clean_analytics_row(record: dict) -> dict:
    """Keep only the fields that exist in ANALYTICS_SCHEMA."""
    fields = {f["name"] for f in ANALYTICS_SCHEMA["fields"]}
    return {k: v for k, v in record.items() if k in fields}


def _clean_alert_row(record: dict) -> dict:
    """Keep only the fields that exist in ALERTS_SCHEMA."""
    fields = {f["name"] for f in ALERTS_SCHEMA["fields"]}
    return {k: v for k, v in record.items() if k in fields}


def write_analytics_to_bq(pcollection, table: str):
    """Write window analytics PCollection to BigQuery."""
    return (
        pcollection
        | "CleanAnalyticsRows" >> beam.Map(_clean_analytics_row)
        | "WriteAnalyticsToBQ" >> WriteToBigQuery(
            table=table,
            schema=ANALYTICS_SCHEMA,
            write_disposition=BigQueryDisposition.WRITE_APPEND,
            create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
        )
    )


def write_alerts_to_bq(pcollection, table: str):
    """Write alerts PCollection to BigQuery."""
    return (
        pcollection
        | "CleanAlertRows" >> beam.Map(_clean_alert_row)
        | "WriteAlertsToBQ" >> WriteToBigQuery(
            table=table,
            schema=ALERTS_SCHEMA,
            write_disposition=BigQueryDisposition.WRITE_APPEND,
            create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
        )
    )
