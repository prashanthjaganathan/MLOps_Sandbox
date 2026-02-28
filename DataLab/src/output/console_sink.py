"""
Console sink — pretty terminal output for all pipeline events.

All functions here are used as beam.Map targets. They receive a single
dict and print formatted output. Return value is ignored by Beam.
"""

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"

# Window size label colors
WINDOW_COLORS = {
    "30s": CYAN,
    "5m": GREEN,
    "sliding_5m": MAGENTA,
}


def print_analytics(record: dict) -> None:
    """
    Print a single window analytics record.

    Example output:
    [30s]  AAPL   VWAP: $   230.42  Trades:   347  Vol: $   2,847,291  Vola: $  0.82  Blk: 1
    """
    window_size = record.get("window_size", "?")
    color = WINDOW_COLORS.get(window_size, RESET)

    block_indicator = ""
    if record["block_trade_count"] > 0:
        block_indicator = f"  {YELLOW}🐋 {record['block_trade_count']} blocks ({record['block_trade_volume']:,} sh){RESET}"

    print(
        f"{color}[{window_size:<9}]{RESET} "
        f"{BOLD}{record['symbol']:<6}{RESET} "
        f"VWAP: ${record['vwap']:>10,.2f}  "
        f"Trades: {record['trade_count']:>5}  "
        f"Vol: ${record['total_dollar_volume']:>14,.0f}  "
        f"Vola: ${record['volatility']:>6,.2f}"
        f"{block_indicator}"
    )


def print_alert(alert: dict) -> None:
    """
    Print a structured alert with color coding by severity.

    Alert types: PRICE_SPIKE, PRICE_DROP, VOLATILITY_SURGE,
                 BLOCK_TRADE_CLUSTER, UNUSUAL_VOLUME
    """
    alert_type = alert.get("type", alert.get("alert_type", "ALERT"))

    if alert_type in ("PRICE_SPIKE", "PRICE_DROP"):
        color = GREEN if alert_type == "PRICE_SPIKE" else RED
    elif alert_type == "VOLATILITY_SURGE":
        color = YELLOW
    elif alert_type == "BLOCK_TRADE_CLUSTER":
        color = MAGENTA
    elif alert_type == "UNUSUAL_VOLUME":
        color = CYAN
    else:
        color = RESET

    message = alert.get("message", str(alert))
    print(f"{color}{BOLD}[ALERT]{RESET} {message}")


def print_sector_signal(signal: dict) -> None:
    """
    Print a cross-stock sector momentum signal.

    Example:
    [SECTOR] TECH_MAG7  BULLISH  ↑5 ↓2  Leader: NVDA +1.24%  Vol: $142,000,000
    """
    direction = signal["direction"]
    if direction == "BULLISH":
        dir_color = GREEN
        arrow = "↑"
    elif direction == "BEARISH":
        dir_color = RED
        arrow = "↓"
    else:
        dir_color = YELLOW
        arrow = "~"

    leader_pct = signal["leader_move_pct"]
    leader_sign = "+" if leader_pct >= 0 else ""

    print(
        f"{BOLD}[SECTOR]{RESET} "
        f"{signal['sector']:<12} "
        f"{dir_color}{BOLD}{direction:<8}{RESET} "
        f"↑{signal['stocks_up']} ↓{signal['stocks_down']}  "
        f"Leader: {BOLD}{signal['leader']}{RESET} {leader_sign}{leader_pct:.3f}%  "
        f"Vol: ${signal['total_dollar_volume']:>14,.0f}"
    )
