# Real-Time US Stock Market Analytics Engine

Processes live NYSE/NASDAQ trades for 500+ stocks via Apache Beam,  
running locally or at scale on Google Cloud Dataflow.

**Tech:** Apache Beam · Python · Finnhub WebSocket · Google Cloud Dataflow · Pub/Sub · BigQuery

---

## Extension of [Apache Beam](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Apache_Beam_Labs) Data Lab

The course lab ([Try_Apache_Beam_Python.ipynb](https://github.com/raminmohammadi/MLOps/blob/main/Labs/Data_Labs/Apache_Beam_Labs/Try_Apache_Beam_Python.ipynb)) ran a word count on a static text file — one linear pipeline, no time, no state, output to a file. This project extends that foundation in  many dimensions:


| #   | What               | Lab                                                  | This Project                                                                                  |
| --- | ------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | **Input**          | Reads one text file, then stops                      | Live stock trade stream — data arrives forever                                                |
| 2   | **Time**           | No timestamps, no time concept                       | Every trade has an exchange timestamp; Beam uses it to place trades in the right window       |
| 3   | **Windowing**      | None — one global bucket                             | 30-second and 5-minute time buckets that open and close automatically                         |
| 4   | **Transforms**     | Built-in lambdas (`Map`, `FlatMap`, `CombinePerKey`) | Custom classes that compute VWAP, detect anomalies, and track sector momentum                 |
| 5   | **Pipeline shape** | Single straight line: input → transform → output     | One input splits into 3 parallel branches; branches share data with each other                |
| 6   | **Side inputs**    | None                                                 | 5-minute baseline is computed and fed into the anomaly detector to compare current vs history |
| 7   | **Output**         | One text file                                        | Live colored terminal dashboard + BigQuery                                                    |
| 8   | **Runner**         | DirectRunner only (laptop)                           | DirectRunner locally or Google Cloud Dataflow — same code, one config change                  |


The core Beam insight the lab introduces — *describe what to compute, not how* — is what makes this pipeline scale from 7 stocks on a laptop to 500+ stocks on Dataflow without touching the pipeline logic.

---

## The Problem

I started with the simplest possible approach: a Python script that reads
trades from a file, maintains a dict of recent prices in memory, and prints
VWAP every 30 seconds.

It worked perfectly for 7 stocks.

```
[Window   1] AAPL    VWAP: $    230.42  Trades:   347  Volume: $   2,847,291  Volatility: $  0.82
[Window   1] NVDA    VWAP: $    141.18  Trades:   512  Volume: $   4,102,444  Volatility: $  1.34
```

## What Broke at Scale

When I scaled to 500 stocks, four things failed silently:

**1. Memory leak** — The in-memory state dict appended every trade to an unbounded list.
At 500 stocks × 50 trades/sec, memory hit 2+ GB and the process was killed by the OS.

**2. Silent data corruption** — A trade arriving 45 seconds late (network lag, API hiccup)
was appended to the *current* window instead of the window it belonged to.
The VWAP was wrong. No error. No warning.

**3. Cross-stock race condition** — To compare all stocks in the same 30-second window,
you need all of them to close their window at exactly the same time.
With `time.sleep(30)`, each symbol's window closed when the loop reached it —
milliseconds apart. The "cross-stock comparison" was comparing different time slices.

**4. Single-threaded throughput collapse** — At 25,000 trades/sec (500 stocks),
a single Python thread fell behind immediately and never caught up.

See `[naive/](naive/)` for the original script and the failure logs.

## Why Apache Beam

Beam solves all four problems by design, not by configuration:


| Problem               | Naive Script                | Apache Beam                                                                  |
| --------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| Memory                | Unbounded list growth → OOM | Windows close and discard data automatically                                 |
| Late data             | Silently wrong VWAP         | Event-time watermarks route trades to the correct window                     |
| Cross-stock alignment | Race condition              | `GroupByKey` guarantees all stocks in the same window are processed together |
| Throughput            | Single thread, falls behind | DirectRunner parallelizes across cores; Dataflow scales to 100s of workers   |


## The Result

Same pipeline code. 7 stocks → 500 stocks → Google Cloud Dataflow. Zero changes to pipeline logic.

---

## Features

- **Multi-window analytics** — 30s fixed, 5m fixed, 5m sliding (rolling average)
- **VWAP computation** — volume-weighted average price, the #1 institutional metric
- **Block trade / whale detection** — single orders ≥ 10,000 shares
- **Cross-stock sector momentum** — are the Mag-7 moving together or diverging?
- **Unusual volume detection** — 3x normal volume triggers an alert
- **Price spike/drop anomaly detection** — 0.5% VWAP deviation vs 5-min baseline
- **Volatility surge detection** — 3x normal intra-window price range
- **Runs locally OR on Dataflow** — zero code changes between environments

---

## Project Structure

```
DataLab/
├── naive/
│   ├── simple_analytics.py     # Phase 0: the naive script (intentionally broken at scale)
│   └── README_naive.md         # Documents exactly what breaks and why
├── src/
│   ├── models/trade.py         # StockTrade dataclass (ingest boundary only)
│   ├── ingest/
│   │   ├── finnhub_websocket.py  # Live Finnhub WebSocket → file or Pub/Sub
│   │   └── simulator.py          # Fake trade generator (7 or 500 stocks)
│   ├── pipeline/
│   │   ├── parse.py            # Parse + validate raw Finnhub JSON
│   │   ├── enrich.py           # Add derived fields (block trade flag, tier)
│   │   ├── analytics.py        # Window analytics DoFn (VWAP, volume, volatility)
│   │   ├── cross_stock.py      # Sector momentum + unusual volume (side inputs)
│   │   ├── anomaly.py          # Price spike, volatility surge, block clusters
│   │   └── stock_pipeline.py   # Main pipeline — wires all phases together
│   └── output/
│       ├── console_sink.py     # Colored terminal dashboard
│       └── bigquery_sink.py    # BigQuery writer (Phase 5)
├── scripts/
│   ├── run_local.sh            # DirectRunner (laptop)
│   ├── run_dataflow.sh         # DataflowRunner (GCP)
│   ├── generate_test_data.sh   # Start the simulator
│   └── benchmark.sh            # Naive vs Beam comparison
├── tests/
│   ├── test_parse.py
│   ├── test_analytics.py
│   └── test_anomaly.py
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key (optional — simulator works without it)

```bash
cp .env.example .env
# Edit .env and add your Finnhub API key from https://finnhub.io
```

### 3. Start the simulator (Terminal 1)

```bash
# 7 stocks — development
bash scripts/generate_test_data.sh 7

# 500 stocks — stress test
bash scripts/generate_test_data.sh 500
```

### 4. Run the pipeline (Terminal 2)

```bash
bash scripts/run_local.sh
```

### 5. Or use live Finnhub data (Terminal 1)

```bash
python -m src.ingest.finnhub_websocket --output trades.jsonl
```

---

## Terminal Output

```
[30s      ] AAPL   VWAP: $    230.42  Trades:   347  Vol: $   2,847,291  Vola: $  0.82
[30s      ] NVDA   VWAP: $    141.18  Trades:   512  Vol: $   4,102,444  Vola: $  1.34  🐋 1 blocks (15,000 sh)
[30s      ] TSLA   VWAP: $    352.90  Trades:   289  Vol: $   1,983,120  Vola: $  2.10
[sliding_5m] AAPL  VWAP: $    230.38  Trades:  3402  Vol: $  28,103,882  Vola: $  1.45
[SECTOR] TECH_MAG7    BULLISH  ↑5 ↓2  Leader: NVDA +1.240%  Vol: $  142,000,000
[ALERT] 🚀 NVDA PRICE_SPIKE: +0.872%  (VWAP $141.18 vs 5m $139.96)
[ALERT] ⚡ TSLA VOLATILITY_SURGE: 4.2x normal  (range $2.10 vs normal $0.50)
[ALERT] 🐋 AAPL WHALE ACTIVITY: 3 block trades, 48,000 shares in 30s
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Phase 5: Deploy to Google Cloud Dataflow (Under Testing)

```bash
# Set your GCP config
export GCP_PROJECT=your-project-id
export GCS_BUCKET=your-bucket-name

# Create Pub/Sub topic and subscription
gcloud pubsub topics create stock-trades
gcloud pubsub subscriptions create stock-trades-sub --topic=stock-trades

# Start the Finnhub → Pub/Sub publisher
python -m src.ingest.finnhub_websocket \
    --pubsub-topic projects/$GCP_PROJECT/topics/stock-trades

# Deploy the pipeline (same code, different runner)
bash scripts/run_dataflow.sh
```

The Dataflow job graph will show all three window branches, the cross-stock
join, and the anomaly detection — all running in parallel across auto-scaled workers.

---

## Benchmark: Naive vs Beam

```bash
bash scripts/benchmark.sh
```


| Metric                | Naive (7 stocks)    | Naive (500 stocks)  | Beam (500 stocks)    |
| --------------------- | ------------------- | ------------------- | -------------------- |
| Memory after 10 min   | ~20 MB              | OOM / 2+ GB         | ~80 MB stable        |
| Late trade handling   | Silent wrong answer | Silent wrong answer | Correct (event-time) |
| Cross-stock alignment | Race condition      | Race condition      | Guaranteed aligned   |
| Throughput at 25k/sec | Keeps up            | Falls behind        | Keeps up             |
| Code changes to scale | —                   | 0 (same script)     | 0 (same pipeline)    |


