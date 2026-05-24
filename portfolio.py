"""
portfolio.py — Risk-parity allocator + regime tilt + portfolio combiner.

Combines the three books (mean_rev, momentum, defensive) into a single
stock-level weight panel. The combination is done in three steps:

1. Compute each book's daily return series from its target weights and
   the underlying stock returns.
2. Compute book-level risk-parity weights: each book weighted inversely
   to its trailing realized volatility. Each book contributes equal risk.
3. Apply the regime tilt: a multiplicative adjustment based on the
   current trend × turbulence regime, then renormalize.

Final stock weights = sum over books of (book_weight × book_stock_weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def book_returns(book_weights: pd.DataFrame, stock_returns: pd.DataFrame) -> pd.Series:
    """
    Daily return of a book given its target weights and the underlying
    stock returns. We use yesterday's target weights × today's return
    (no look-ahead).
    """
    aligned_w = book_weights.shift(1).reindex_like(stock_returns).fillna(0.0)
    return (aligned_w * stock_returns).sum(axis=1)


# ---------------------------------------------------------------------------
# Risk-parity book allocation
# ---------------------------------------------------------------------------
def risk_parity_weights(
    books_rets: pd.DataFrame,
    lookback: int = config.RP_VOL_LOOKBACK,
    min_w: float = config.RP_MIN_WEIGHT,
    max_w: float = config.RP_MAX_WEIGHT,
) -> pd.DataFrame:
    """
    Inverse-vol book weights.

    For each book b on each day t:
        w_b(t) ∝ 1 / vol_b(t-1)

    Weights are floored/capped and renormalized.

    Parameters
    ----------
    books_rets : DataFrame  rows=dates, columns=book names (mean_rev, momentum, defensive)
    """
    vol = books_rets.rolling(lookback, min_periods=lookback).std() * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )
    inv_vol = 1.0 / vol.shift(1).replace(0, np.nan)
    raw = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    raw = raw.clip(lower=min_w, upper=max_w)
    # Renormalize after clipping
    return raw.div(raw.sum(axis=1), axis=0).fillna(1.0 / books_rets.shape[1])


# ---------------------------------------------------------------------------
# Regime tilt
# ---------------------------------------------------------------------------
def apply_regime_tilt(
    rp_weights: pd.DataFrame,
    regime_df: pd.DataFrame,
    tilts: dict = config.REGIME_TILTS,
) -> pd.DataFrame:
    """
    Multiplicatively adjust book weights by the regime tilt, then renormalize.

    `regime_df` must have a 'regime' column with tuples (trend, turbulent)
    matching the keys of `tilts`. Book order in tilts values must be
    (mean_rev, momentum, defensive).
    """
    cols = ["mean_rev", "momentum", "defensive"]
    assert list(rp_weights.columns) == cols, f"Expected columns {cols}, got {list(rp_weights.columns)}"

    # Build per-date multiplier matrix (lagged by 1 day to avoid look-ahead)
    multipliers = pd.DataFrame(1.0, index=rp_weights.index, columns=cols)
    regime_lagged = regime_df["regime"].shift(1)
    for ts, key in regime_lagged.items():
        if key in tilts:
            multipliers.loc[ts] = list(tilts[key])

    tilted = rp_weights * multipliers
    return tilted.div(tilted.sum(axis=1), axis=0).fillna(1.0 / len(cols))


# ---------------------------------------------------------------------------
# Combine books into stock-level weights
# ---------------------------------------------------------------------------
def combine_books(
    book_weights: dict[str, pd.DataFrame],
    book_allocation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stack book weights into a stock-level weight panel.

    book_weights   : dict of book_name -> DataFrame (rows=dates, cols=tickers)
    book_allocation: DataFrame (rows=dates, cols=book names)

    Returns: DataFrame (rows=dates, cols=tickers) of total stock weights.
    """
    stock_weights = None
    for book_name, w in book_weights.items():
        scaled = w.mul(book_allocation[book_name], axis=0)
        stock_weights = scaled if stock_weights is None else stock_weights.add(scaled, fill_value=0)
    return stock_weights


# ---------------------------------------------------------------------------
# Risk overlay (portfolio-level vol circuit breaker)
# ---------------------------------------------------------------------------
def apply_vol_circuit_breaker(
    stock_weights: pd.DataFrame,
    stock_returns: pd.DataFrame,
    lookback: int = config.VOL_LOOKBACK_DAYS,
    threshold: float = config.VOL_CIRCUIT_BREAKER,
    scale: float = config.DERISK_SCALE,
) -> pd.DataFrame:
    """
    Scale all positions to `scale` (e.g. 0.5) on dates where the portfolio's
    trailing realized volatility exceeded `threshold`. Remaining capital
    sits in cash (return 0 in our model).
    """
    port_ret = (stock_weights.shift(1) * stock_returns).sum(axis=1)
    port_vol = port_ret.rolling(lookback, min_periods=lookback).std() * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )
    derisk = (port_vol.shift(1) > threshold).fillna(False)
    multiplier = pd.Series(1.0, index=stock_weights.index)
    multiplier[derisk] = scale
    return stock_weights.mul(multiplier, axis=0)


# ---------------------------------------------------------------------------
# Rebalance frequency: resample weights to weekly (or whatever cadence)
# ---------------------------------------------------------------------------
def resample_to_rebalance(weights: pd.DataFrame, freq: str = config.REBALANCE_FREQ) -> pd.DataFrame:
    """
    Forward-fill weights between rebalance dates. We only "act" on rebalance
    dates; on other days we hold the prior weights.
    """
    rebal_dates = weights.resample(freq).last().index
    rebal_weights = weights.loc[weights.index.intersection(rebal_dates)]
    return rebal_weights.reindex(weights.index).ffill().fillna(0.0)
