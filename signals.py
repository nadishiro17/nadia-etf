"""
signals.py — Pure signal functions on price/return panels.

All functions take a wide DataFrame (rows = dates, columns = tickers) and
return another wide DataFrame of the same shape. No side effects, no I/O.

These are the building blocks for the three books in portfolio.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns as decimals (not ×100 — internal calcs use decimals)."""
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


# ---------------------------------------------------------------------------
# Mean-reversion signal: rolling z-score of price vs its moving average
# ---------------------------------------------------------------------------
def zscore(prices: pd.DataFrame, lookback: int = config.MR_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    z = (P_t - mean_{t-lookback..t}) / std_{t-lookback..t}

    Negative z = stretched below recent average = mean-reversion buy signal.
    """
    roll = prices.rolling(lookback, min_periods=lookback)
    return (prices - roll.mean()) / roll.std()


# ---------------------------------------------------------------------------
# Momentum signal: 12-1 cumulative return
# ---------------------------------------------------------------------------
def momentum(
    prices: pd.DataFrame,
    lookback: int = config.MOM_LOOKBACK_DAYS,
    skip: int = config.MOM_SKIP_DAYS,
) -> pd.DataFrame:
    """
    Cumulative return from (t - lookback) to (t - skip).

    Standard "12-1" momentum: 12-month return excluding the most recent month.
    Skipping the most recent month avoids contaminating the signal with
    short-term reversal (Jegadeesh & Titman 1993).
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# Realized volatility (annualized, from log returns)
# ---------------------------------------------------------------------------
def realized_vol(
    returns: pd.DataFrame,
    lookback: int = config.VOL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Annualized rolling-window realized volatility, assuming 252 trading days."""
    return returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )


def realized_vol_series(
    returns: pd.Series, lookback: int = config.VOL_LOOKBACK_DAYS
) -> pd.Series:
    return returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )


# ---------------------------------------------------------------------------
# Regime indicator on SPY (or any benchmark)
# ---------------------------------------------------------------------------
def regime(
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    train_end: pd.Timestamp | None = None,
    trend_lookback: int = config.REGIME_TREND_LOOKBACK,
    vol_lookback: int = config.REGIME_VOL_LOOKBACK,
    vol_percentile: float = config.REGIME_VOL_PERCENTILE,
) -> pd.DataFrame:
    """
    Two-axis regime: trend (bull/bear) × turbulence (calm/turbulent).

    Trend = SPY above its `trend_lookback`-day moving average?
    Turbulent = SPY `vol_lookback`-day realized vol above the `vol_percentile`-th
                percentile of vol values observed up to `train_end`.

    The vol threshold is computed on training data only to avoid look-ahead bias.
    If `train_end` is None, we use the first 50% of the series for thresholding.

    Returns
    -------
    DataFrame with columns ['trend', 'turbulent', 'regime']:
        trend     : 'bull' or 'bear'
        turbulent : True / False
        regime    : tuple suitable for indexing into config.REGIME_TILTS
    """
    sma = benchmark_prices.rolling(trend_lookback, min_periods=trend_lookback).mean()
    trend = np.where(benchmark_prices > sma, "bull", "bear")

    vol = realized_vol_series(benchmark_returns, lookback=vol_lookback)

    # Threshold: use training data only
    if train_end is None:
        train_end = benchmark_prices.index[len(benchmark_prices) // 2]
    vol_train = vol.loc[:train_end].dropna()
    if len(vol_train) == 0:
        vol_threshold = vol.dropna().quantile(vol_percentile)
    else:
        vol_threshold = vol_train.quantile(vol_percentile)

    turbulent = (vol > vol_threshold).fillna(False)

    out = pd.DataFrame(
        {
            "trend": trend,
            "vol": vol,
            "vol_threshold": vol_threshold,
            "turbulent": turbulent,
        },
        index=benchmark_prices.index,
    )
    out["regime"] = list(zip(out["trend"], out["turbulent"]))
    return out


# ---------------------------------------------------------------------------
# Top-N ranking helper
# ---------------------------------------------------------------------------
def top_n_mask(scores: pd.DataFrame, n: int, ascending: bool = False) -> pd.DataFrame:
    """
    Boolean mask: True for the top-N tickers (by score) on each row.
    `ascending=False` -> top-N by largest score; `ascending=True` -> bottom-N.
    """
    ranks = scores.rank(axis=1, ascending=ascending, method="first")
    return ranks <= n
