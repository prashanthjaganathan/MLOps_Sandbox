#!/usr/bin/env bash
# Run the pipeline on Google Cloud Dataflow.
# SAME pipeline code as run_local.sh — only the runner and I/O sources change.
#
# Prerequisites:
#   1. gcloud auth application-default login
#   2. Set environment variables below (or export them before running)
#   3. Pub/Sub topic "stock-trades" must exist
#   4. GCS bucket must exist
#
# Usage:
#   GCP_PROJECT=my-project GCS_BUCKET=my-bucket bash scripts/run_dataflow.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — set these or export them before running
# ---------------------------------------------------------------------------
GCP_PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET env var}"
REGION="${REGION:-us-east1}"
PUBSUB_SUBSCRIPTION="${PUBSUB_SUBSCRIPTION:-projects/${GCP_PROJECT}/subscriptions/stock-trades}"
OUTPUT_TABLE="${OUTPUT_TABLE:-${GCP_PROJECT}:stocks.window_analytics}"
MAX_WORKERS="${MAX_WORKERS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "  Stock Analytics Pipeline — Dataflow (GCP)"
echo "=================================================="
echo "  Project     : $GCP_PROJECT"
echo "  Region      : $REGION"
echo "  Input       : $PUBSUB_SUBSCRIPTION"
echo "  Output      : $OUTPUT_TABLE"
echo "  Max workers : $MAX_WORKERS"
echo "=================================================="

python -m src.pipeline.stock_pipeline \
    --runner=DataflowRunner \
    --project="$GCP_PROJECT" \
    --region="$REGION" \
    --temp_location="gs://${GCS_BUCKET}/temp" \
    --staging_location="gs://${GCS_BUCKET}/staging" \
    --input="$PUBSUB_SUBSCRIPTION" \
    --output-table="$OUTPUT_TABLE" \
    --enable-analytics \
    --enable-cross-stock \
    --enable-anomaly \
    --streaming \
    --autoscaling_algorithm=THROUGHPUT_BASED \
    --max_num_workers="$MAX_WORKERS" \
    --job_name="stock-analytics-$(date +%Y%m%d-%H%M%S)"
