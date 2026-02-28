"""
Finnhub WebSocket client — connects to Finnhub's free real-time trade stream
and writes raw trade JSON to a local file or publishes to Google Cloud Pub/Sub.

Finnhub free tier: 60 API calls/min, WebSocket streams real-time trades.
Sign up at finnhub.io to get your free API key.

Usage:
    # Write to local file (for DirectRunner / development)
    FINNHUB_API_KEY=your_key python -m src.ingest.finnhub_websocket

    # Publish to Pub/Sub (for DataflowRunner / production)
    FINNHUB_API_KEY=your_key python -m src.ingest.finnhub_websocket \
        --pubsub-topic projects/your-project/topics/stock-trades
"""

import argparse
import json
import logging
import os
import sys
import time

import websocket
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FINNHUB_WS_URL = "wss://ws.finnhub.io?token={api_key}"

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]


class FinnhubClient:
    def __init__(self, api_key: str, output_file: str | None, pubsub_topic: str | None):
        self.api_key = api_key
        self.output_file = output_file
        self.pubsub_topic = pubsub_topic
        self._pubsub_client = None
        self._file_handle = None
        self.trade_count = 0

        if pubsub_topic:
            from google.cloud import pubsub_v1
            self._pubsub_client = pubsub_v1.PublisherClient()
            logger.info(f"Publishing to Pub/Sub topic: {pubsub_topic}")
        else:
            self._file_handle = open(output_file, "w")
            logger.info(f"Writing to file: {output_file}")

    def _emit(self, trade: dict) -> None:
        payload = json.dumps(trade)
        if self._pubsub_client:
            self._pubsub_client.publish(
                self.pubsub_topic,
                payload.encode("utf-8"),
                symbol=trade["s"],
            )
        else:
            self._file_handle.write(payload + "\n")
            self._file_handle.flush()
        self.trade_count += 1

    def on_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("type") == "trade":
            for trade in data.get("data", []):
                self._emit(trade)
                if self.trade_count % 1000 == 0:
                    logger.info(f"{self.trade_count:,} trades received")

        elif data.get("type") == "ping":
            pass  # Finnhub sends periodic pings; no action needed

    def on_error(self, ws, error) -> None:
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg) -> None:
        logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")
        if self._file_handle:
            self._file_handle.close()

    def on_open(self, ws) -> None:
        logger.info(f"Connected to Finnhub. Subscribing to {WATCHLIST}")
        for symbol in WATCHLIST:
            ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))

    def start(self, reconnect: bool = True) -> None:
        url = FINNHUB_WS_URL.format(api_key=self.api_key)
        while True:
            ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
            if not reconnect:
                break
            logger.info("Reconnecting in 5 seconds...")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finnhub WebSocket client")
    parser.add_argument("--output", default="trades.jsonl", help="Output JSONL file (local mode)")
    parser.add_argument("--pubsub-topic", default=None, help="Pub/Sub topic (GCP mode)")
    args = parser.parse_args()

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        logger.error("FINNHUB_API_KEY environment variable not set.")
        logger.error("Get a free key at https://finnhub.io and add it to your .env file.")
        sys.exit(1)

    client = FinnhubClient(
        api_key=api_key,
        output_file=args.output if not args.pubsub_topic else None,
        pubsub_topic=args.pubsub_topic,
    )
    client.start()


if __name__ == "__main__":
    main()
