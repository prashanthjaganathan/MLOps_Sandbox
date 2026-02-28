"""
Phase 0: The Naive Script — intentionally limited.

This is the "before" version. It works fine for 7 stocks.
Run it with --stocks=500 and watch it fail in four distinct ways.

Usage:
    # Generate trades first (in another terminal):
    python -m src.ingest.simulator --stocks 7 --output trades.jsonl

    # Run this naive script:
    python naive/simple_analytics.py --stocks 7
    python naive/simple_analytics.py --stocks 500   # watch it break
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime


WATCHLIST_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

WINDOW_SECONDS = 30


# ---------------------------------------------------------------------------
# FAILURE 1: This dict grows forever. No eviction. No window closing.
# At 500 stocks × 50 trades/sec, this hits gigabytes of RAM.
# ---------------------------------------------------------------------------
state: dict[str, list[dict]] = defaultdict(list)

# Track when the current window started
window_start = time.time()

# For lag measurement (Failure 4)
trades_processed = 0
processing_start = time.time()


def compute_vwap(trades: list[dict]) -> float:
    total_volume = sum(t["v"] for t in trades)
    if total_volume == 0:
        return 0.0
    return sum(t["p"] * t["v"] for t in trades) / total_volume


def print_window_analytics(symbol: str, trades: list[dict], window_num: int) -> None:
    if not trades:
        return

    prices = [t["p"] for t in trades]
    volumes = [t["v"] for t in trades]

    vwap = compute_vwap(trades)
    total_volume = sum(volumes)
    dollar_volume = sum(t["p"] * t["v"] for t in trades)
    volatility = max(prices) - min(prices)

    print(
        f"[Window {window_num:>3}] {symbol:<6}  "
        f"VWAP: ${vwap:>10,.2f}  "
        f"Trades: {len(trades):>5}  "
        f"Volume: ${dollar_volume:>14,.2f}  "
        f"Volatility: ${volatility:>6,.2f}"
    )


def flush_window(window_num: int) -> None:
    """
    Print analytics for all symbols and clear state.

    FAILURE 3: This iterates symbols sequentially. AAPL's window closes
    at a slightly different wall-clock time than META's. Cross-stock
    comparisons in this window are comparing misaligned time slices.
    """
    print(f"\n{'='*80}")
    print(f"  Window {window_num} | {datetime.now().strftime('%H:%M:%S')} | {len(state)} symbols tracked")
    print(f"{'='*80}")

    for symbol in sorted(state.keys()):
        trades = state[symbol]
        if trades:
            print_window_analytics(symbol, trades, window_num)

    # FAILURE 1: We clear the list but the dict key stays. At 500 symbols,
    # even the empty lists add up. More critically, if we DON'T clear
    # (e.g. for a rolling window), the lists grow without bound.
    state.clear()


def inject_late_trade(symbol: str) -> None:
    """
    Simulate a late-arriving trade (45 seconds old).

    FAILURE 2: This trade has a timestamp from the PREVIOUS window,
    but we append it to the CURRENT window's list because we have no
    event-time tracking. The VWAP will be silently wrong.
    """
    late_trade = {
        "s": symbol,
        "p": 999.99,   # obviously wrong price to make the corruption visible
        "v": 10000,
        "t": int((time.time() - 45) * 1000),  # 45 seconds in the past
        "c": [1],
        "_late": True,
    }
    state[symbol].append(late_trade)
    print(f"  [LATE TRADE INJECTED] {symbol} — timestamp is 45s old, appended to current window anyway")


def run(input_file: str, stocks: int, inject_late: bool) -> None:
    global window_start, trades_processed, processing_start

    tracemalloc.start()

    watchlist = WATCHLIST_7 if stocks <= 7 else None  # None means accept all symbols
    window_num = 0
    last_mem_report = time.time()

    print(f"Starting naive analytics | stocks={stocks} | window={WINDOW_SECONDS}s")
    print(f"Input: {input_file}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if not os.path.exists(input_file):
                print(f"Waiting for {input_file}...")
                time.sleep(1)
                continue

            with open(input_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        trade = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    symbol = trade.get("s", "")
                    if watchlist and symbol not in watchlist:
                        continue

                    # FAILURE 1: Unconditional append. No eviction. No bound.
                    state[symbol].append(trade)
                    trades_processed += 1

                    now = time.time()

                    # FAILURE 4: Measure how far behind we're falling
                    elapsed_wall = now - processing_start
                    if elapsed_wall > 0:
                        actual_rate = trades_processed / elapsed_wall
                        # At 500 stocks, this rate will drop below the input rate

                    # Window flush every 30 seconds
                    if now - window_start >= WINDOW_SECONDS:
                        window_num += 1
                        flush_window(window_num)
                        window_start = now

                        # Inject a late trade every other window to demonstrate Failure 2
                        if inject_late and window_num % 2 == 0 and state:
                            inject_late("AAPL" if "AAPL" in state else list(state.keys())[0])

                    # Memory report every 60 seconds
                    if now - last_mem_report >= 60:
                        current, peak = tracemalloc.get_traced_memory()
                        elapsed = now - processing_start
                        rate = trades_processed / elapsed if elapsed > 0 else 0
                        print(
                            f"\n[MEM] Current: {current/1024/1024:.1f} MB | "
                            f"Peak: {peak/1024/1024:.1f} MB | "
                            f"Trades: {trades_processed:,} | "
                            f"Rate: {rate:.0f} trades/sec | "
                            f"Symbols tracked: {len(state)}"
                        )
                        last_mem_report = now

            # File exhausted — wait for more data
            time.sleep(0.1)

    except KeyboardInterrupt:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - processing_start
        print(f"\n\n{'='*60}")
        print("NAIVE SCRIPT SUMMARY")
        print(f"{'='*60}")
        print(f"Total trades processed : {trades_processed:,}")
        print(f"Elapsed time           : {elapsed:.1f}s")
        print(f"Average rate           : {trades_processed/elapsed:.0f} trades/sec")
        print(f"Peak memory            : {peak/1024/1024:.1f} MB")
        print(f"Symbols in state       : {len(state)}")
        print(f"{'='*60}")
        print("\nFailures demonstrated:")
        print("  Failure 1 (Memory)     : state dict never evicts — see peak MB above")
        print("  Failure 2 (Late data)  : injected late trades corrupted VWAP silently")
        print("  Failure 3 (Alignment)  : each symbol's window closed at different times")
        print("  Failure 4 (Throughput) : single thread; falls behind at 500 stocks")
        print("\nThis is why we need Apache Beam. See src/pipeline/stock_pipeline.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive stock analytics script (Phase 0)")
    parser.add_argument("--input", default="trades.jsonl", help="Input JSONL file")
    parser.add_argument("--stocks", type=int, default=7, help="Number of stocks (7 or 500)")
    parser.add_argument("--inject-late", action="store_true", help="Inject late trades to show Failure 2")
    args = parser.parse_args()

    run(args.input, args.stocks, args.inject_late)
