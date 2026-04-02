"""
Data fetcher for Indian stock market data.

Fetches and caches:
- Nifty 200 constituent list (current)
- Daily OHLCV data for all constituents via yfinance
- Nifty 50 index data (benchmark)
- India VIX data
- Sector/industry classification
"""

import os
import pickle
import datetime
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Nifty 200 Constituents ──────────────────────────────────────────────────
# We maintain a hardcoded list of Nifty 200 tickers. Fetching dynamically from
# NSE is unreliable (rate limits, geo-blocking). This list is the current
# Nifty 200 composition as of March 2026.
#
# NOTE: Using current constituents introduces survivorship bias for backtests.
# A production system should use point-in-time membership data.

NIFTY_200_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "BHARTIARTL", "ITC", "SBIN", "LT", "KOTAKBANK", "BAJFINANCE",
    "AXISBANK", "HCLTECH", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN",
    "WIPRO", "ULTRACEMCO", "NESTLEIND", "TATAMOTORS", "M&M", "NTPC",
    "POWERGRID", "TECHM", "BAJAJFINSV", "TATASTEEL", "HDFCLIFE", "ONGC",
    "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA", "DIVISLAB",
    "GRASIM", "BPCL", "CIPLA", "BRITANNIA", "DRREDDY", "APOLLOHOSP",
    "EICHERMOT", "SBILIFE", "INDUSINDBK", "HEROMOTOCO", "DABUR",
    "BAJAJ-AUTO", "PIDILITIND", "GODREJCP", "SIEMENS", "HAVELLS",
    "AMBUJACEM", "SHREECEM", "BERGEPAINT", "DLF", "VEDL", "HINDPETRO",
    "INDIGO", "TATACONSUM", "MCDOWELL-N", "IOC", "BANDHANBNK",
    "BANKBARODA", "PNB", "ICICIPRULI", "IDFCFIRSTB", "GAIL",
    "NAUKRI", "COLPAL", "MARICO", "TRENT", "PIIND", "MUTHOOTFIN",
    "TORNTPHARM", "LUPIN", "AUROPHARMA", "LICI", "SBICARD",
    "ZOMATO", "PAYTM", "DMART", "ADANIGREEN", "ADANITRANS",
    "INDUSTOWER", "TATAPOWER", "PEL", "VOLTAS", "JUBLFOOD",
    "MFSL", "CHOLAFIN", "BALKRISIND", "PERSISTENT", "COFORGE",
    "LTIM", "ABCAPITAL", "ACC", "PAGEIND", "TATAELXSI",
    "POLYCAB", "CUMMINSIND", "ESCORTS", "MRF", "GODREJPROP",
    "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE", "SUNTV", "MAXHEALTH",
    "ALKEM", "IPCALAB", "LAURUSLABS", "METROPOLIS", "DIXON",
    "ASTRAL", "SUPREMEIND", "AIAENG", "ABFRL", "CONCOR",
    "BIOCON", "CANFINHOME", "FEDERALBNK", "IDBI", "IRCTC",
    "HAL", "BEL", "BHEL", "SAIL", "NMDC", "RECLTD", "PFC",
    "NHPC", "IRFC", "SJVN", "GICRE", "NIACL",
    "INDIANB", "CENTRALBK", "UNIONBANK", "CANBK", "MAHABANK",
    "MANAPPURAM", "ATUL", "DEEPAKNTR", "NAVINFLUOR", "SRF",
    "AARTIIND", "COROMANDEL", "UPL", "TATACOMM", "LTTS",
    "MPHASIS", "ZYDUSLIFE", "GMRAIRPORT", "JINDALSAW",
    "HINDCOPPER", "NATIONALUM", "PETRONET", "IGL", "MGL",
    "ATGL", "CROMPTON", "WHIRLPOOL", "BATAINDIA", "RELAXO",
    "LICHSGFIN", "RBLBANK", "AUBANK", "MOTHERSON",
    "EXIDEIND", "AMARAJABAT", "BOSCHLTD", "THERMAX",
    "KPITTECH", "SONACOMS", "SYNGENE", "FORTIS", "LALPATHLAB",
    "YESBANK", "JKCEMENT", "RAMCOCEM", "DALBHARAT",
    "STARHEALTH", "HDFCAMC", "ICICIGI", "ISEC",
    "CLEAN", "IREDA", "JSWENERGY", "TORNTPOWER", "CESC",
    "CGPOWER", "KAYNES", "TIINDIA", "SUNDARMFIN",
]

# Yahoo Finance uses .NS suffix for NSE stocks
def to_yf_ticker(ticker: str) -> str:
    return f"{ticker}.NS"


def get_nifty200_tickers() -> list[str]:
    """Return the Nifty 200 ticker list."""
    return NIFTY_200_TICKERS.copy()


def fetch_price_data(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str | None = None,
    cache_file: str = "price_data.pkl",
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV data for a list of NSE tickers via yfinance.
    Returns a dict mapping ticker -> DataFrame with columns:
    [Open, High, Low, Close, Volume, Adj Close]

    Data is cached to disk to avoid repeated downloads.
    """
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        print(f"Loading cached price data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if end is None:
        end = datetime.date.today().isoformat()

    print(f"Fetching price data for {len(tickers)} tickers from {start} to {end}...")
    price_data = {}
    failed = []

    for ticker in tqdm(tickers, desc="Downloading"):
        yf_ticker = to_yf_ticker(ticker)
        try:
            df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df is not None and len(df) > 100:  # Need at least ~100 trading days
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                price_data[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  Failed {ticker}: {e}")
        time.sleep(0.2)  # Rate limiting

    print(f"\nSuccessfully fetched: {len(price_data)} tickers")
    if failed:
        print(f"Failed/insufficient data: {len(failed)} tickers: {failed[:20]}...")

    with open(cache_path, "wb") as f:
        pickle.dump(price_data, f)
    print(f"Cached to {cache_path}")

    return price_data


def fetch_benchmark_data(
    start: str = "2014-01-01",
    end: str | None = None,
    cache_file: str = "benchmark_data.pkl",
) -> dict[str, pd.DataFrame]:
    """
    Fetch benchmark index data:
    - ^NSEI  : Nifty 50
    - ^CRSLDX: Nifty 500
    - ^INDIAVIX: India VIX
    """
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        print(f"Loading cached benchmark data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if end is None:
        end = datetime.date.today().isoformat()

    benchmarks = {
        "NIFTY50": "^NSEI",
        "NIFTY500": "^CRSLDX",
        "INDIAVIX": "^INDIAVIX",
    }

    data = {}
    for name, yf_ticker in benchmarks.items():
        print(f"Fetching {name} ({yf_ticker})...")
        try:
            df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and len(df) > 0:
                data[name] = df
                print(f"  Got {len(df)} rows")
        except Exception as e:
            print(f"  Failed {name}: {e}")

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cached to {cache_path}")
    return data


def fetch_sector_data(
    tickers: list[str],
    cache_file: str = "sector_data.pkl",
) -> dict[str, dict]:
    """
    Fetch sector/industry classification for each ticker via yfinance .info.
    Returns dict mapping ticker -> {sector, industry, marketCap}.
    """
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        print(f"Loading cached sector data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Fetching sector data for {len(tickers)} tickers...")
    sector_data = {}

    for ticker in tqdm(tickers, desc="Sector info"):
        yf_ticker = to_yf_ticker(ticker)
        try:
            info = yf.Ticker(yf_ticker).info
            sector_data[ticker] = {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "marketCap": info.get("marketCap", 0),
            }
        except Exception:
            sector_data[ticker] = {
                "sector": "Unknown",
                "industry": "Unknown",
                "marketCap": 0,
            }
        time.sleep(0.15)

    with open(cache_path, "wb") as f:
        pickle.dump(sector_data, f)
    print(f"Cached to {cache_path}")
    return sector_data


def build_close_price_matrix(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a DataFrame of adjusted close prices: rows=dates, columns=tickers."""
    close_dict = {}
    for ticker, df in price_data.items():
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close_dict[ticker] = df[col]

    close_df = pd.DataFrame(close_dict)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()
    return close_df


def build_volume_matrix(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a DataFrame of daily volume: rows=dates, columns=tickers."""
    vol_dict = {}
    for ticker, df in price_data.items():
        if "Volume" in df.columns:
            vol_dict[ticker] = df["Volume"]
    vol_df = pd.DataFrame(vol_dict)
    vol_df.index = pd.to_datetime(vol_df.index)
    vol_df = vol_df.sort_index()
    return vol_df


def load_all_data(
    start: str = "2014-01-01",
    end: str | None = None,
) -> dict:
    """
    Load all data needed for backtesting.
    Returns a dict with keys:
    - tickers: list of ticker symbols
    - price_data: dict of ticker -> DataFrame
    - close_prices: DataFrame (dates x tickers)
    - volume: DataFrame (dates x tickers)
    - benchmarks: dict of benchmark -> DataFrame
    - sectors: dict of ticker -> {sector, industry, marketCap}
    """
    tickers = get_nifty200_tickers()
    price_data = fetch_price_data(tickers, start=start, end=end)

    # Only keep tickers we successfully fetched
    valid_tickers = list(price_data.keys())

    benchmarks = fetch_benchmark_data(start=start, end=end)
    sectors = fetch_sector_data(valid_tickers)

    close_prices = build_close_price_matrix(price_data)
    volume = build_volume_matrix(price_data)

    return {
        "tickers": valid_tickers,
        "price_data": price_data,
        "close_prices": close_prices,
        "volume": volume,
        "benchmarks": benchmarks,
        "sectors": sectors,
    }


if __name__ == "__main__":
    data = load_all_data()
    print(f"\nData summary:")
    print(f"  Tickers: {len(data['tickers'])}")
    print(f"  Date range: {data['close_prices'].index[0]} to {data['close_prices'].index[-1]}")
    print(f"  Benchmarks: {list(data['benchmarks'].keys())}")
    print(f"  Sectors: {len(set(s['sector'] for s in data['sectors'].values()))} unique sectors")
