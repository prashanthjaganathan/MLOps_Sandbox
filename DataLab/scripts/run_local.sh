#!/usr/bin/env bash
# Run the full pipeline locally using Apache Beam's DirectRunner.
# No GCP account needed. Reads from trades.jsonl written by the simulator.
#
# Usage:
#   # Terminal 1: start the simulator
#   python -m src.ingest.simulator --stocks 7 --output trades.jsonl
#
#   # Terminal 2: run this script
#   bash scripts/run_local.sh
#
#   # To test at 500 stocks:
#   python -m src.ingest.simulator --stocks 500 --output trades.jsonl
#   bash scripts/run_local.sh --input trades.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

INPUT_FILE="${1:-trades.jsonl}"

echo "=================================================="
echo "  Stock Analytics Pipeline — DirectRunner (local)"
echo "=================================================="
echo "  Input : $INPUT_FILE"
echo "  Runner: DirectRunner"
echo ""
echo "  Starting pipeline... (Ctrl+C to stop)"
echo "=================================================="

python -m src.pipeline.stock_pipeline \
    --runner=DirectRunner \
    --input="$INPUT_FILE" \
    --enable-analytics \
    --enable-cross-stock \
    --enable-anomaly
