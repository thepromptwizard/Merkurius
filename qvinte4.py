import yfinance as yf
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os
import sys
import time


# ============ CONFIGURATION ============

CRYPTOS = {
    "1": ("BTC", "Bitcoin"),
    "2": ("ETH", "Ethereum"),
    "3": ("BNB", "Binance Coin"),
    "4": ("XRP", "Ripple"),
    "5": ("ADA", "Cardano"),
    "6": ("SOL", "Solana"),
    "7": ("DOGE", "Dogecoin"),
    "8": ("DOT", "Polkadot"),
    "9": ("MATIC", "Polygon"),
    "10": ("LTC", "Litecoin"),
    "11": ("SHIB", "Shiba Inu"),
    "12": ("AVAX", "Avalanche"),
    "13": ("LINK", "Chainlink"),
    "14": ("UNI", "Uniswap"),
    "15": ("ATOM", "Cosmos"),
}

FIATS = {
    "1": ("USD", "US Dollar"),
    "2": ("EUR", "Euro"),
    "3": ("GBP", "British Pound"),
    "4": ("JPY", "Japanese Yen"),
    "5": ("AUD", "Australian Dollar"),
    "6": ("CAD", "Canadian Dollar"),
    "7": ("CHF", "Swiss Franc"),
}

TIMEFRAMES = {
    "1": ("1m", "1 minute", "1d"),
    "2": ("5m", "5 minutes", "5d"),
    "3": ("15m", "15 minutes", "5d"),
    "4": ("1h", "1 hour", "1mo"),
    "5": ("1d", "1 day", "6mo"),
}


# ============ ANALYSIS ENGINE ============

class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    CAUTION = "CAUTION"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NO_TRADE = "NO TRADE"


@dataclass
class MarketData:
    """Container for all market data inputs."""
    ticker: str
    crypto: str
    fiat: str
    current_price: float
    sma_20: float
    sma_50: float
    sma_200: float
    rsi: float
    macd: float
    macd_signal: float
    volume: float
    avg_volume: float
    high_24h: float
    low_24h: float
    change_24h: float
    support: Optional[float] = None
    resistance: Optional[float] = None


@dataclass
class Analysis:
    """Container for analysis results."""
    trend: str
    trend_strength: str
    rsi_condition: str
    macd_condition: str
    volume_condition: str
    proximity: str
    signal: Signal
    recommendation: str
    confidence: int
    reasons: list
    scores: dict


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_signal_color(signal: Signal) -> str:
    """Get ANSI color code for signal."""
    colors = {
        Signal.STRONG_BUY: "\033[92m",    # Bright green
        Signal.BUY: "\033[32m",            # Green
        Signal.HOLD: "\033[93m",           # Yellow
        Signal.CAUTION: "\033[33m",        # Orange/dark yellow
        Signal.SELL: "\033[31m",           # Red
        Signal.STRONG_SELL: "\033[91m",    # Bright red
        Signal.NO_TRADE: "\033[90m",       # Gray
    }
    return colors.get(signal, "\033[0m")


def reset_color() -> str:
    return "\033[0m"


# ============ ANALYSIS FUNCTIONS ============

def analyze_trend(data: MarketData) -> tuple[str, str, int]:
    """Analyze trend using multiple SMAs."""
    score = 0

    if data.current_price > data.sma_20:
        score += 1
    if data.current_price > data.sma_50:
        score += 1
    if data.current_price > data.sma_200:
        score += 2

    if data.sma_20 > data.sma_50 > data.sma_200:
        score += 2  # Golden alignment
    elif data.sma_20 < data.sma_50 < data.sma_200:
        score -= 2  # Death alignment

    if score >= 4:
        return "BULLISH", "STRONG", score
    elif score >= 2:
        return "BULLISH", "MODERATE", score
    elif score <= -2:
        return "BEARISH", "STRONG", score
    elif score < 0:
        return "BEARISH", "MODERATE", score
    else:
        return "NEUTRAL", "WEAK", score


def analyze_rsi(rsi: float) -> tuple[str, int]:
    """Analyze RSI conditions."""
    if rsi > 80:
        return "EXTREMELY OVERBOUGHT", -2
    elif rsi > 70:
        return "OVERBOUGHT", -1
    elif rsi < 20:
        return "EXTREMELY OVERSOLD", 2
    elif rsi < 30:
        return "OVERSOLD", 1
    elif 45 <= rsi <= 55:
        return "NEUTRAL", 0
    elif rsi > 55:
        return "BULLISH MOMENTUM", 1
    else:
        return "BEARISH MOMENTUM", -1


def analyze_macd(macd: float, signal: float) -> tuple[str, int]:
    """Analyze MACD crossover and momentum."""
    if macd > signal and macd > 0:
        return "STRONG BULLISH", 2
    elif macd > signal:
        return "BULLISH CROSSOVER", 1
    elif macd < signal and macd < 0:
        return "STRONG BEARISH", -2
    elif macd < signal:
        return "BEARISH CROSSOVER", -1
    else:
        return "NEUTRAL", 0


def analyze_volume(volume: float, avg_volume: float) -> tuple[str, int]:
    """Analyze volume for confirmation."""
    ratio = volume / avg_volume if avg_volume > 0 else 1

    if ratio > 2.0:
        return "VERY HIGH (Breakout?)", 2
    elif ratio > 1.3:
        return "ABOVE AVERAGE", 1
    elif ratio < 0.5:
        return "VERY LOW (Weak move)", -1
    elif ratio < 0.7:
        return "BELOW AVERAGE", 0
    else:
        return "NORMAL", 0


def analyze_support_resistance(data: MarketData) -> tuple[str, int]:
    """Analyze proximity to support/resistance."""
    range_size = data.high_24h - data.low_24h
    if range_size == 0:
        return "NO DATA", 0

    position = (data.current_price - data.low_24h) / range_size

    if data.support and data.resistance:
        support_dist = (data.current_price - data.support) / data.current_price * 100
        resist_dist = (data.resistance - data.current_price) / data.current_price * 100

        if support_dist < 2:
            return "NEAR SUPPORT (Bounce zone)", 2
        elif resist_dist < 2:
            return "NEAR RESISTANCE (Rejection zone)", -1

    if position < 0.2:
        return "NEAR 24H LOW", 1
    elif position > 0.8:
        return "NEAR 24H HIGH", -1
    else:
        return "MID-RANGE", 0


def calculate_signal(scores: dict, reasons: list) -> tuple[Signal, str, int]:
    """Calculate final signal based on all scores."""
    total = sum(scores.values())

    positive = sum(1 for s in scores.values() if s > 0)
    negative = sum(1 for s in scores.values() if s < 0)
    agreement = max(positive, negative) / len(scores) * 100
    confidence = int(min(agreement + abs(total) * 5, 100))

    if total >= 6:
        signal = Signal.STRONG_BUY
        rec = "EXCELLENT ENTRY - Multiple confirmations aligned"
    elif total >= 3:
        signal = Signal.BUY
        rec = "GOOD ENTRY - Favorable conditions"
    elif total >= 1:
        signal = Signal.HOLD
        rec = "WAIT - Look for better confirmation"
    elif total >= -2:
        signal = Signal.CAUTION
        rec = "RISKY - Mixed signals"
    elif total >= -5:
        signal = Signal.SELL
        rec = "EXIT - Conditions deteriorating"
    else:
        signal = Signal.STRONG_SELL
        rec = "DANGER - Strong bearish pressure"

    return signal, rec, confidence


def analyze_market(data: MarketData) -> Analysis:
    """Complete market analysis."""
    reasons = []
    scores = {}

    trend, trend_strength, scores["trend"] = analyze_trend(data)
    rsi_cond, scores["rsi"] = analyze_rsi(data.rsi)
    macd_cond, scores["macd"] = analyze_macd(data.macd, data.macd_signal)
    vol_cond, scores["volume"] = analyze_volume(data.volume, data.avg_volume)
    prox_cond, scores["proximity"] = analyze_support_resistance(data)

    if scores["trend"] > 0:
        reasons.append(f"[+] Trend is {trend} ({trend_strength})")
    else:
        reasons.append(f"[-] Trend is {trend} ({trend_strength})")

    if scores["rsi"] > 0:
        reasons.append(f"[+] RSI indicates {rsi_cond}")
    elif scores["rsi"] < 0:
        reasons.append(f"[-] RSI indicates {rsi_cond}")

    if scores["macd"] > 0:
        reasons.append(f"[+] MACD is {macd_cond}")
    elif scores["macd"] < 0:
        reasons.append(f"[-] MACD is {macd_cond}")

    if scores["volume"] != 0:
        reasons.append(f"[*] Volume is {vol_cond}")

    if scores["proximity"] != 0:
        reasons.append(f"[*] Price is {prox_cond}")

    signal, recommendation, confidence = calculate_signal(scores, reasons)

    return Analysis(
        trend=trend,
        trend_strength=trend_strength,
        rsi_condition=rsi_cond,
        macd_condition=macd_cond,
        volume_condition=vol_cond,
        proximity=prox_cond,
        signal=signal,
        recommendation=recommendation,
        confidence=confidence,
        reasons=reasons,
        scores=scores
    )


# ============ DATA FETCHING ============

def fetch_market_data(crypto: str, fiat: str, interval: str, period: str) -> Optional[MarketData]:
    """Fetch real market data and calculate all required indicators."""
    ticker = f"{crypto}-{fiat}"

    print(f"\n  Fetching {ticker} data...")

    try:
        coin = yf.Ticker(ticker)
        data = coin.history(period=period, interval=interval)

        if data.empty:
            print("  ERROR: No data received")
            return None

        # Calculate all indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = data.ta.rsi(length=14)

        # MACD calculation
        macd_df = data.ta.macd(fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            data['MACD'] = macd_df.iloc[:, 0]
            data['MACD_Signal'] = macd_df.iloc[:, 2]
        else:
            data['MACD'] = 0
            data['MACD_Signal'] = 0

        data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()

        latest = data.iloc[-1]
        current_price = latest['Close']

        def safe_get(series, key, default):
            val = series.get(key, default)
            return default if pd.isna(val) else val

        sma_20 = safe_get(latest, 'SMA_20', current_price)
        sma_50 = safe_get(latest, 'SMA_50', current_price)
        sma_200 = safe_get(latest, 'SMA_200', current_price)
        rsi = safe_get(latest, 'RSI', 50.0)
        macd = safe_get(latest, 'MACD', 0.0)
        macd_signal = safe_get(latest, 'MACD_Signal', 0.0)
        volume = safe_get(latest, 'Volume', 0)
        avg_volume = safe_get(latest, 'Avg_Volume', volume)

        # 24h data
        recent = data.tail(min(288, len(data)))
        high_24h = recent['High'].max()
        low_24h = recent['Low'].min()

        # 24h change
        if len(data) > 1:
            open_24h = data.iloc[-min(288, len(data))]['Open']
            change_24h = ((current_price - open_24h) / open_24h) * 100
        else:
            change_24h = 0

        # Support/resistance
        support = recent['Low'].nsmallest(3).mean()
        resistance = recent['High'].nlargest(3).mean()

        print(f"  Data loaded: {len(data)} candles")

        return MarketData(
            ticker=ticker,
            crypto=crypto,
            fiat=fiat,
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            volume=volume,
            avg_volume=avg_volume,
            high_24h=high_24h,
            low_24h=low_24h,
            change_24h=change_24h,
            support=support,
            resistance=resistance
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ============ DISPLAY ============

def get_currency_symbol(fiat: str) -> str:
    """Get currency symbol for fiat."""
    symbols = {
        "USD": "$", "EUR": "E", "GBP": "L", "JPY": "Y",
        "AUD": "A$", "CAD": "C$", "CHF": "Fr"
    }
    return symbols.get(fiat, "$")


def display_analysis(data: MarketData, analysis: Analysis):
    """Display formatted analysis report."""
    sym = get_currency_symbol(data.fiat)
    color = get_signal_color(analysis.signal)
    reset = reset_color()

    # Price formatting based on value
    if data.current_price < 0.01:
        price_fmt = f"{sym}{data.current_price:.6f}"
        range_fmt = f"{sym}{data.low_24h:.6f} - {sym}{data.high_24h:.6f}"
    elif data.current_price < 1:
        price_fmt = f"{sym}{data.current_price:.4f}"
        range_fmt = f"{sym}{data.low_24h:.4f} - {sym}{data.high_24h:.4f}"
    else:
        price_fmt = f"{sym}{data.current_price:,.2f}"
        range_fmt = f"{sym}{data.low_24h:,.2f} - {sym}{data.high_24h:,.2f}"

    change_color = "\033[92m" if data.change_24h >= 0 else "\033[91m"
    change_sign = "+" if data.change_24h >= 0 else ""

    width = 60

    print()
    print("+" + "=" * width + "+")
    print("|" + f" QVINTE MARKET SNIPER - {data.crypto}/{data.fiat} ".center(width) + "|")
    print("+" + "=" * width + "+")

    # Price section
    print("|" + " PRICE ".center(width, "-") + "|")
    print("|" + f"  Current:   {price_fmt}".ljust(width) + "|")
    print("|" + f"  24h Range: {range_fmt}".ljust(width) + "|")
    print("|" + f"  24h Change:{change_color} {change_sign}{data.change_24h:.2f}%{reset}".ljust(width + 9) + "|")

    # Moving Averages
    print("+" + "-" * width + "+")
    print("|" + " MOVING AVERAGES ".center(width, "-") + "|")
    if data.current_price < 1:
        print("|" + f"  SMA 20:  {sym}{data.sma_20:.4f}".ljust(width) + "|")
        print("|" + f"  SMA 50:  {sym}{data.sma_50:.4f}".ljust(width) + "|")
        print("|" + f"  SMA 200: {sym}{data.sma_200:.4f}".ljust(width) + "|")
    else:
        print("|" + f"  SMA 20:  {sym}{data.sma_20:,.2f}".ljust(width) + "|")
        print("|" + f"  SMA 50:  {sym}{data.sma_50:,.2f}".ljust(width) + "|")
        print("|" + f"  SMA 200: {sym}{data.sma_200:,.2f}".ljust(width) + "|")

    # Indicators
    print("+" + "-" * width + "+")
    print("|" + " INDICATORS ".center(width, "-") + "|")

    trend_color = "\033[92m" if "BULLISH" in analysis.trend else "\033[91m" if "BEARISH" in analysis.trend else "\033[93m"
    print("|" + f"  Trend:    {trend_color}{analysis.trend} ({analysis.trend_strength}){reset}".ljust(width + 9) + "|")

    rsi_color = "\033[91m" if "OVERBOUGHT" in analysis.rsi_condition else "\033[92m" if "OVERSOLD" in analysis.rsi_condition else "\033[93m"
    print("|" + f"  RSI:      {rsi_color}{data.rsi:.1f} - {analysis.rsi_condition}{reset}".ljust(width + 9) + "|")

    macd_color = "\033[92m" if "BULLISH" in analysis.macd_condition else "\033[91m" if "BEARISH" in analysis.macd_condition else "\033[93m"
    print("|" + f"  MACD:     {macd_color}{analysis.macd_condition}{reset}".ljust(width + 9) + "|")

    vol_ratio = data.volume / data.avg_volume if data.avg_volume > 0 else 1
    print("|" + f"  Volume:   {vol_ratio:.0%} of avg - {analysis.volume_condition}".ljust(width) + "|")
    print("|" + f"  Position: {analysis.proximity}".ljust(width) + "|")

    # Signal
    print("+" + "-" * width + "+")
    print("|" + " SIGNAL ".center(width, "-") + "|")
    print("|" + f"  {color}>>> {analysis.signal.value} <<<{reset}".center(width + 9) + "|")
    print("|" + f"  Confidence: {analysis.confidence}%".center(width) + "|")

    # Score breakdown
    print("+" + "-" * width + "+")
    print("|" + " SCORE BREAKDOWN ".center(width, "-") + "|")
    total = sum(analysis.scores.values())
    score_parts = [f"{k}:{v:+d}" for k, v in analysis.scores.items()]
    print("|" + f"  {' | '.join(score_parts)} = {total:+d}".ljust(width) + "|")

    # Reasoning
    print("+" + "-" * width + "+")
    print("|" + " REASONING ".center(width, "-") + "|")
    for reason in analysis.reasons:
        print("|" + f"  {reason}".ljust(width) + "|")

    # Recommendation
    print("+" + "=" * width + "+")
    print("|" + f"  {color}{analysis.recommendation}{reset}".ljust(width + 9) + "|")
    print("+" + "=" * width + "+")


# ============ MENU SYSTEM ============

def print_header():
    """Print application header."""
    clear_screen()
    print()
    print("  ===================================================")
    print("   QVINTE MARKET SNIPER v4.0")
    print("   Crypto Technical Analysis Tool")
    print("  ===================================================")
    print()


def select_from_menu(options: dict, title: str, allow_custom: bool = False) -> Optional[tuple]:
    """Generic menu selector."""
    print(f"\n  {title}")
    print("  " + "-" * 40)

    for key, value in options.items():
        print(f"    [{key}] {value[0]:6} - {value[1]}")

    if allow_custom:
        print(f"    [C] Custom - Enter your own")
    print(f"    [Q] Cancel")
    print()

    choice = input("  Select option: ").strip().upper()

    if choice == 'Q':
        return None
    elif choice == 'C' and allow_custom:
        custom = input("  Enter custom value: ").strip().upper()
        return (custom, "Custom")
    elif choice in options:
        return options[choice]
    else:
        print("  Invalid selection!")
        return select_from_menu(options, title, allow_custom)


def main_menu():
    """Main application menu."""
    selected_crypto = ("BTC", "Bitcoin")
    selected_fiat = ("USD", "US Dollar")
    selected_timeframe = ("5m", "5 minutes", "5d")

    while True:
        print_header()

        print("  CURRENT SETTINGS")
        print("  " + "-" * 40)
        print(f"    Crypto:    {selected_crypto[0]} ({selected_crypto[1]})")
        print(f"    Fiat:      {selected_fiat[0]} ({selected_fiat[1]})")
        print(f"    Timeframe: {selected_timeframe[1]}")
        print()

        print("  MENU")
        print("  " + "-" * 40)
        print("    [1] Change Cryptocurrency")
        print("    [2] Change Fiat Currency")
        print("    [3] Change Timeframe")
        print("    [4] Run Analysis")
        print("    [5] Quick Analysis (Multi-Crypto)")
        print("    [6] Continuous Monitor")
        print()
        print("    [Q] Quit")
        print()

        choice = input("  Select option: ").strip().upper()

        if choice == '1':
            result = select_from_menu(CRYPTOS, "SELECT CRYPTOCURRENCY", allow_custom=True)
            if result:
                selected_crypto = result

        elif choice == '2':
            result = select_from_menu(FIATS, "SELECT FIAT CURRENCY")
            if result:
                selected_fiat = result

        elif choice == '3':
            result = select_from_menu(TIMEFRAMES, "SELECT TIMEFRAME")
            if result:
                selected_timeframe = result

        elif choice == '4':
            run_single_analysis(selected_crypto[0], selected_fiat[0],
                              selected_timeframe[0], selected_timeframe[2])
            input("\n  Press Enter to continue...")

        elif choice == '5':
            run_quick_analysis(selected_fiat[0], selected_timeframe[0],
                             selected_timeframe[2])
            input("\n  Press Enter to continue...")

        elif choice == '6':
            run_continuous_monitor(selected_crypto[0], selected_fiat[0],
                                  selected_timeframe[0], selected_timeframe[2])

        elif choice == 'Q':
            print("\n  Goodbye!\n")
            sys.exit(0)


def run_single_analysis(crypto: str, fiat: str, interval: str, period: str):
    """Run analysis for a single crypto pair."""
    print_header()
    print(f"  Analyzing {crypto}/{fiat}...")

    market_data = fetch_market_data(crypto, fiat, interval, period)

    if market_data is None:
        print("\n  Analysis failed: Could not fetch market data")
        return

    analysis = analyze_market(market_data)
    display_analysis(market_data, analysis)


def run_quick_analysis(fiat: str, interval: str, period: str):
    """Run quick analysis for top cryptos."""
    print_header()
    print("  QUICK ANALYSIS - Top Cryptocurrencies")
    print("  " + "=" * 50)

    top_cryptos = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE"]
    results = []

    for crypto in top_cryptos:
        print(f"\n  Fetching {crypto}...", end=" ", flush=True)
        data = fetch_market_data(crypto, fiat, interval, period)
        if data:
            analysis = analyze_market(data)
            results.append((crypto, data, analysis))
            color = get_signal_color(analysis.signal)
            reset = reset_color()
            print(f"{color}{analysis.signal.value}{reset} ({analysis.confidence}%)")
        else:
            print("FAILED")

    # Summary table
    print("\n")
    print("  " + "=" * 70)
    print(f"  {'CRYPTO':<8} {'PRICE':>12} {'24H CHG':>10} {'SIGNAL':>15} {'CONF':>6}")
    print("  " + "-" * 70)

    for crypto, data, analysis in results:
        sym = get_currency_symbol(fiat)
        if data.current_price < 1:
            price_str = f"{sym}{data.current_price:.4f}"
        else:
            price_str = f"{sym}{data.current_price:,.2f}"

        change_sign = "+" if data.change_24h >= 0 else ""
        change_str = f"{change_sign}{data.change_24h:.2f}%"

        color = get_signal_color(analysis.signal)
        reset = reset_color()

        print(f"  {crypto:<8} {price_str:>12} {change_str:>10} {color}{analysis.signal.value:>15}{reset} {analysis.confidence:>5}%")

    print("  " + "=" * 70)


def run_continuous_monitor(crypto: str, fiat: str, interval: str, period: str):
    """Run continuous monitoring with auto-refresh."""
    refresh_seconds = 60

    print("\n  Starting continuous monitor...")
    print(f"  Refresh interval: {refresh_seconds} seconds")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            print_header()
            print(f"  CONTINUOUS MONITOR - {crypto}/{fiat}")
            print(f"  Auto-refresh every {refresh_seconds}s | Press Ctrl+C to stop")
            print()

            market_data = fetch_market_data(crypto, fiat, interval, period)

            if market_data:
                analysis = analyze_market(market_data)
                display_analysis(market_data, analysis)

                # Alert for strong signals
                if analysis.signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]:
                    print(f"\n  !!! ALERT: {analysis.signal.value} DETECTED !!!")
            else:
                print("  Failed to fetch data. Retrying...")

            print(f"\n  Next refresh in {refresh_seconds} seconds...")
            time.sleep(refresh_seconds)

    except KeyboardInterrupt:
        print("\n\n  Monitor stopped.")


# ============ MAIN ============

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  Goodbye!\n")
        sys.exit(0)
