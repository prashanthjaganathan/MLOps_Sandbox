#!/usr/bin/env bash
# Benchmark: naive script vs Beam pipeline at 7 stocks and 500 stocks.
# Runs each version for 2 minutes and reports memory + throughput.
#
# Usage:
#   bash scripts/benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DURATION=120  # seconds per benchmark run

echo "============================================================"
echo "  Benchmark: Naive Script vs Apache Beam"
echo "  Duration per run: ${DURATION}s"
echo "============================================================"

run_benchmark() {
    local label="$1"
    local stocks="$2"
    local script="$3"
    local output_file="trades_bench_${stocks}.jsonl"

    echo ""
    echo "--- $label ($stocks stocks) ---"

    # Start simulator in background
    python -m src.ingest.simulator \
        --stocks "$stocks" \
        --output "$output_file" \
        --rate 50 &
    SIM_PID=$!
    sleep 2  # let simulator warm up

    # Start the script under test, capture peak memory via /usr/bin/time
    if [ "$script" = "naive" ]; then
        /usr/bin/time -l python naive/simple_analytics.py \
            --input "$output_file" \
            --stocks "$stocks" \
            2>&1 &
    else
        /usr/bin/time -l python -m src.pipeline.stock_pipeline \
            --runner=DirectRunner \
            --input "$output_file" \
            2>&1 &
    fi
    SCRIPT_PID=$!

    sleep "$DURATION"

    kill "$SCRIPT_PID" 2>/dev/null || true
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SCRIPT_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true

    rm -f "$output_file"
    echo "  Done."
}

run_benchmark "Naive Script" 7 "naive"
run_benchmark "Naive Script" 500 "naive"
run_benchmark "Apache Beam" 500 "beam"

echo ""
echo "============================================================"
echo "  Benchmark complete. Compare memory and throughput above."
echo "  See naive/README_naive.md for the full comparison table."
echo "============================================================"
