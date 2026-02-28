"""
Trade simulator — generates realistic fake stock trades for offline development.

Use this when markets are closed, to avoid API rate limits, or to stress-test
the pipeline at 500 stocks without a Finnhub subscription.

Usage:
    # 7 stocks, 50 trades/sec, write to trades.jsonl
    python -m src.ingest.simulator

    # 500 stocks, 100 trades/sec
    python -m src.ingest.simulator --stocks 500 --rate 100

    # Generate a static snapshot file (for tests)
    python -m src.ingest.simulator --stocks 7 --count 1000 --output tests/fixtures/trades.jsonl
"""

import argparse
import json
import random
import string
import sys
import time
from datetime import datetime


MAG7 = {
    "AAPL": {"base_price": 230.0, "volatility": 0.002, "avg_volume": 150},
    "MSFT": {"base_price": 420.0, "volatility": 0.0015, "avg_volume": 120},
    "GOOGL": {"base_price": 175.0, "volatility": 0.002, "avg_volume": 100},
    "AMZN": {"base_price": 200.0, "volatility": 0.0018, "avg_volume": 130},
    "NVDA": {"base_price": 140.0, "volatility": 0.003, "avg_volume": 200},
    "TSLA": {"base_price": 350.0, "volatility": 0.004, "avg_volume": 180},
    "META": {"base_price": 600.0, "volatility": 0.002, "avg_volume": 110},
}


def generate_symbol_universe(n: int) -> list[str]:
    """Generate n unique fake ticker symbols (3–4 uppercase letters)."""
    tickers: set[str] = set(MAG7.keys())
    while len(tickers) < n:
        length = random.choice([3, 4])
        ticker = "".join(random.choices(string.ascii_uppercase, k=length))
        tickers.add(ticker)
    return list(tickers)[:n]


def build_stock_universe(n: int) -> dict[str, dict]:
    """Build a universe of n stocks with realistic parameters."""
    if n <= 7:
        return dict(list(MAG7.items())[:n])

    universe = dict(MAG7)
    extra_symbols = generate_symbol_universe(n)[7:]
    for sym in extra_symbols:
        universe[sym] = {
            "base_price": round(random.uniform(10.0, 800.0), 2),
            "volatility": round(random.uniform(0.001, 0.005), 4),
            "avg_volume": random.randint(50, 300),
        }
    return universe


class TradeSimulator:
    def __init__(self, n_stocks: int = 7):
        self.universe = build_stock_universe(n_stocks)
        # Each stock gets its own random-walk price state
        self.current_prices = {sym: info["base_price"] for sym, info in self.universe.items()}
        self.symbols = list(self.universe.keys())

    def next_trade(self, lag_seconds: float = 0.0) -> dict:
        """Generate a single realistic trade message in Finnhub format."""
        symbol = random.choice(self.symbols)
        config = self.universe[symbol]

        # Random walk: price drifts by a small Gaussian each trade
        change = random.gauss(0, config["volatility"])
        self.current_prices[symbol] = max(0.01, self.current_prices[symbol] * (1 + change))
        price = round(self.current_prices[symbol], 2)

        # Volume: Gaussian around average, occasionally a block trade
        avg_vol = config["avg_volume"]
        volume = max(1, int(random.gauss(avg_vol, avg_vol * 0.5)))
        if random.random() < 0.02:  # 2% chance of block trade
            volume *= random.randint(50, 200)

        # Event timestamp — optionally in the past to simulate late data
        event_time = datetime.now().timestamp() - lag_seconds
        return {
            "s": symbol,
            "p": price,
            "v": volume,
            "t": int(event_time * 1000),
            "c": [1],
        }


def run(
    n_stocks: int,
    rate: int,
    output_file: str,
    count: int | None,
    lag_prob: float,
    lag_seconds: float,
) -> None:
    sim = TradeSimulator(n_stocks)
    total = 0
    interval = 1.0 / rate  # seconds between trades

    print(
        f"Simulator started | stocks={n_stocks} | rate={rate}/sec | output={output_file}",
        file=sys.stderr,
    )

    with open(output_file, "w") as f:
        while count is None or total < count:
            # Occasionally inject a late trade to simulate network lag
            lag = lag_seconds if random.random() < lag_prob else 0.0
            trade = sim.next_trade(lag_seconds=lag)

            f.write(json.dumps(trade) + "\n")
            f.flush()
            total += 1

            if count is None:
                time.sleep(interval)

            if total % 10_000 == 0:
                print(f"  {total:,} trades written", file=sys.stderr)

    print(f"Done. {total:,} trades written to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock trade simulator")
    parser.add_argument("--stocks", type=int, default=7, help="Number of stocks to simulate")
    parser.add_argument("--rate", type=int, default=50, help="Trades per second")
    parser.add_argument("--output", default="trades.jsonl", help="Output JSONL file")
    parser.add_argument("--count", type=int, default=None, help="Stop after N trades (default: run forever)")
    parser.add_argument("--lag-prob", type=float, default=0.01, help="Probability of a late trade (0–1)")
    parser.add_argument("--lag-seconds", type=float, default=45.0, help="How many seconds late a late trade is")
    args = parser.parse_args()

    run(
        n_stocks=args.stocks,
        rate=args.rate,
        output_file=args.output,
        count=args.count,
        lag_prob=args.lag_prob,
        lag_seconds=args.lag_seconds,
    )
