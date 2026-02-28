#!/usr/bin/env bash
# Generate test trade data using the simulator.
#
# Usage:
#   bash scripts/generate_test_data.sh              # 7 stocks, continuous stream
#   bash scripts/generate_test_data.sh 500          # 500 stocks, continuous stream
#   bash scripts/generate_test_data.sh 7 1000       # 7 stocks, 1000 trades then stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

STOCKS="${1:-7}"
COUNT="${2:-}"
OUTPUT="trades.jsonl"

echo "Generating trades: stocks=$STOCKS output=$OUTPUT"

if [ -n "$COUNT" ]; then
    python -m src.ingest.simulator \
        --stocks "$STOCKS" \
        --output "$OUTPUT" \
        --count "$COUNT" \
        --lag-prob 0.01
else
    python -m src.ingest.simulator \
        --stocks "$STOCKS" \
        --output "$OUTPUT" \
        --lag-prob 0.01
fi
