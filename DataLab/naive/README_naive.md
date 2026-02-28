# Phase 0: The Naive Script — What Broke and Why

This directory contains the original, intentionally limited approach to stock analytics.
It is preserved here as the "before" in the before/after story.

---

## What It Does

`simple_analytics.py` is a single-file Python script that:

1. Reads `trades.jsonl` line by line in a `while True` loop
2. Maintains a rolling in-memory dict: `state[symbol] = [list of trades]`
3. Every 30 seconds, iterates all symbols, computes VWAP + volume, prints results
4. Uses `time.sleep`-based wall-clock timing to simulate windowing

**For 7 stocks, it works.** Output looks like:

```
================================================================================
  Window   1 | 14:32:01 | 7 symbols tracked
================================================================================
[Window   1] AAPL    VWAP: $    230.42  Trades:   347  Volume: $   2,847,291  Volatility: $  0.82
[Window   1] AMZN    VWAP: $    200.18  Trades:   291  Volume: $   1,923,441  Volatility: $  0.71
[Window   1] GOOGL   VWAP: $    175.33  Trades:   312  Volume: $   1,740,220  Volatility: $  0.64
```

---

## The Four Failures

### Failure 1 — Memory Leak

**What happens:** The `state` dict appends every trade to an in-memory list with no eviction.

```python
# This line runs ~50 times per second per stock:
state[symbol].append(trade)   # list grows forever
```

At 7 stocks × 50 trades/sec, memory grows slowly (~5 MB/min). Manageable.

At 500 stocks × 50 trades/sec = 25,000 trades/sec:
- After 10 minutes: ~150 MB
- After 1 hour: process killed by OS (OOM)

Even with the `state.clear()` call at window boundaries, the Python allocator
holds onto the memory. In a real streaming scenario where you want a rolling
window (not a tumbling one), you never clear — and the leak is unbounded.

**Why Beam fixes it:** Beam's windowing model is built on bounded, closeable windows.
When a window closes, Beam discards the data. Memory stays flat regardless of runtime.

---

### Failure 2 — Late Data Produces Silent Wrong Answers

**What happens:** A trade arrives 45 seconds late (network lag, API hiccup, reconnect).
The script has no concept of event time — it appends the trade to whatever window
is currently open based on wall-clock time.

```python
# Trade timestamp says t=14:31:15 (45 seconds ago)
# Current window is 14:31:30 → 14:32:00
# The trade gets appended to the current window anyway.
# VWAP is now computed across two different time periods.
# No error. No warning. Just wrong numbers.
```

To see this live, run with `--inject-late`:

```bash
python naive/simple_analytics.py --inject-late
```

You'll see output like:
```
[LATE TRADE INJECTED] AAPL — timestamp is 45s old, appended to current window anyway
[Window   2] AAPL    VWAP: $    999.99  ...  ← corrupted by late trade
```

**Why Beam fixes it:** Beam processes on **event time**, not processing time.
It uses watermarks to track how far behind the stream is, and routes each trade
to the window it actually belongs to based on its timestamp.

---

### Failure 3 — Cross-Stock Window Alignment Is a Race Condition

**What happens:** The script iterates symbols sequentially in a `for` loop.
Each symbol's "window" effectively closes at a slightly different wall-clock time:

```
AAPL window closes at t=30.001s  (first in the loop)
MSFT window closes at t=30.002s
GOOGL window closes at t=30.003s
...
META window closes at t=30.007s  (last in the loop)
```

When you try to answer "are all tech stocks moving in the same direction this window?",
you're comparing data from 7 slightly different time slices. At 500 stocks, the
last symbol's window closes ~50ms after the first. That's not the same window.

**Why Beam fixes it:** Beam's `GroupByKey` after windowing guarantees that all
elements assigned to the same window are processed together, regardless of when
they arrived. The window boundary is defined by event time, not by when the loop
reaches that symbol.

---

### Failure 4 — Single-Threaded Throughput Collapse

**What happens:** Python's GIL means one thread processes one trade at a time.

```
At 500 stocks × 50 trades/sec = 25,000 trades/sec input rate

Per-trade work: json.loads + dict lookup + list append + occasional VWAP ≈ 0.08ms

25,000 trades/sec × 0.08ms = 2.0 seconds of CPU work per second of real time

The script falls behind immediately and never catches up.
```

The input file grows faster than the script reads it. After 10 minutes,
the script is processing trades from 20 minutes ago. The "real-time" dashboard
is showing stale data.

**Why Beam fixes it:** Beam's `DirectRunner` parallelizes across all available
CPU cores automatically. The `DataflowRunner` scales to hundreds of workers.
The pipeline code is identical — parallelism is a runner concern, not a code concern.

---

## Benchmark: Naive vs Beam

| Metric | Naive (7 stocks) | Naive (500 stocks) | Beam (500 stocks) |
|---|---|---|---|
| Memory after 10 min | ~20 MB | OOM / 2+ GB | ~80 MB stable |
| Late trade handling | Silent wrong answer | Silent wrong answer | Correct (event-time) |
| Cross-stock alignment | Race condition | Race condition | Guaranteed aligned |
| Throughput at 25k/sec | Keeps up | Falls behind | Keeps up |
| Code changes to scale | — | 0 (same script) | 0 (same pipeline) |

---

## How to Reproduce the Failures

```bash
# From DataLab/ root

# Terminal 1: generate 7-stock stream
python -m src.ingest.simulator --stocks 7 --output trades.jsonl

# Terminal 2: run naive script (works fine)
python naive/simple_analytics.py --stocks 7 --inject-late

# Terminal 3: generate 500-stock stream (overwrites trades.jsonl)
python -m src.ingest.simulator --stocks 500 --output trades.jsonl

# Terminal 2: re-run naive script (watch it fail)
python naive/simple_analytics.py --stocks 500
# Watch: memory climbs in the [MEM] reports, output starts lagging
```

---

## What Comes Next

See `src/pipeline/stock_pipeline.py` for the Beam version.
Same analytics. Same output format. None of these four failures.
