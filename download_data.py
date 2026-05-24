"""
download_data.py — Pull daily prices for the universe, benchmark, and
risk-free rate from Yahoo Finance via yfinance.

Run once (or whenever the universe changes):
    python download_data.py

Writes one Parquet file per ticker into data/prices/, plus a combined
adjusted-close panel at data/adj_close.parquet for fast loading.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


PRICES_DIR = config.DATA_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)


def download_one(ticker: str, retries: int = 3, pause: float = 1.0) -> pd.DataFrame:
    """Download a single ticker with simple retry. Returns OHLCV+AdjClose."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=config.START_DATE,
                end=config.END_DATE,
                auto_adjust=False,         # keep raw Close + Adj Close separate
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError("empty frame")
            # Flatten multiindex columns yfinance sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            return df
        except Exception as e:
            print(f"  attempt {attempt}/{retries} failed for {ticker}: {e}")
            time.sleep(pause * attempt)
    raise RuntimeError(f"could not download {ticker} after {retries} attempts")


def main() -> int:
    all_tickers = list(config.FUND_TICKERS) + [
        config.BENCHMARK_TICKER,
        config.BENCHMARK_PRICE_TICKER,
        config.RISK_FREE_TICKER,
    ]
    print(f"Downloading {len(all_tickers)} symbols "
          f"from {config.START_DATE} to {config.END_DATE}")

    adj_close = {}
    failed = []

    for i, tk in enumerate(all_tickers, 1):
        print(f"[{i:2d}/{len(all_tickers)}] {tk} ...", end=" ", flush=True)
        try:
            df = download_one(tk)
        except Exception as e:
            print(f"FAILED — {e}")
            failed.append(tk)
            continue

        out_path = PRICES_DIR / f"{tk.replace('^', '').replace('/', '_')}.parquet"
        df.to_parquet(out_path)

        if "Adj Close" in df.columns:
            adj_close[tk] = df["Adj Close"]
        elif "Close" in df.columns:
            adj_close[tk] = df["Close"]

        first = df.index.min().date()
        last  = df.index.max().date()
        print(f"ok  rows={len(df):5d}  range={first}..{last}")

    if failed:
        print(f"\nWARNING: failed to download {len(failed)} symbols: {failed}")

    if adj_close:
        panel = pd.DataFrame(adj_close).sort_index()
        panel_path = config.DATA_DIR / "adj_close.parquet"
        panel.to_parquet(panel_path)
        print(f"\nWrote combined panel: {panel_path}  "
              f"shape={panel.shape}  "
              f"date range={panel.index.min().date()}..{panel.index.max().date()}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
