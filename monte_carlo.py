"""
monte_carlo.py — Block bootstrap Monte Carlo for the strategy.

For N simulations:
  1. Sample blocks of B consecutive trading days from the actual return
     panel (with replacement). All 50 stocks' returns are sampled jointly
     so cross-sectional correlations are preserved.
  2. Concatenate blocks to form a synthetic 26-year return panel of the
     same length as history. Convert back to prices (cumprod).
  3. Run the strategy on the synthetic prices.
  4. Record the simulation's CAGR / Sharpe / max drawdown / final equity.

Result: a distribution over 1000 alternative 26-year histories that share
the cross-sectional / short-range time-series characteristics of the real
data but are otherwise independent re-orderings.

Why block-bootstrap (not Gaussian):
  Our pooled daily-return distribution showed kurtosis ≈ 24. Gaussian MC
  would massively understate tail risk. Block-bootstrap preserves the
  empirical distribution including fat tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import backtest
import evaluate


def block_bootstrap_returns(
    stock_returns: pd.DataFrame,
    bench_returns: pd.Series,
    n_days: int,
    block_size: int = config.MC_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sample blocks of consecutive days from stock_returns and bench_returns
    (jointly, so cross-sectional structure is preserved).

    Returns synthetic stock_returns and bench_returns of length `n_days`.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid = stock_returns.dropna(how="any").index
    if len(valid) < block_size * 2:
        raise ValueError("Not enough valid days for block-bootstrap")

    aligned_bench = bench_returns.reindex(valid).fillna(0.0)
    aligned_stock = stock_returns.reindex(valid)

    n_blocks = int(np.ceil(n_days / block_size))
    max_start = len(valid) - block_size
    starts = rng.integers(0, max_start, size=n_blocks)

    stock_chunks, bench_chunks = [], []
    for s in starts:
        stock_chunks.append(aligned_stock.iloc[s : s + block_size].values)
        bench_chunks.append(aligned_bench.iloc[s : s + block_size].values)

    synth_stock = np.vstack(stock_chunks)[:n_days]
    synth_bench = np.concatenate(bench_chunks)[:n_days]

    # Build a fresh DatetimeIndex of business days, matching length
    new_index = pd.bdate_range(start="2000-01-03", periods=n_days)

    return (
        pd.DataFrame(synth_stock, index=new_index, columns=aligned_stock.columns),
        pd.Series(synth_bench, index=new_index, name="bench"),
    )


def returns_to_prices(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    """Convert daily simple returns to a price series starting at `start_price`."""
    return start_price * (1.0 + returns.fillna(0.0)).cumprod()


def returns_to_price_series(returns: pd.Series, start_price: float = 100.0) -> pd.Series:
    return start_price * (1.0 + returns.fillna(0.0)).cumprod()


def run_single_mc(
    stock_returns: pd.DataFrame,
    bench_returns: pd.Series,
    n_days: int,
    rng: np.random.Generator,
) -> dict:
    """Run one MC iteration. Returns a dict of summary metrics."""
    synth_rets, synth_bench_rets = block_bootstrap_returns(
        stock_returns, bench_returns, n_days, rng=rng
    )
    synth_prices = returns_to_prices(synth_rets)
    synth_bench_prices = returns_to_price_series(synth_bench_rets)

    train_end = synth_prices.index[len(synth_prices) // 5]  # first 20% for regime threshold

    result = backtest.run_strategy(
        synth_prices, synth_bench_prices, train_end=train_end
    )
    sim = result["simulation"]
    eq = sim["equity"].dropna()
    if len(eq) < 252:
        return {"final_equity": np.nan, "CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan,
                "bench_final": np.nan, "bench_CAGR": np.nan}

    daily = sim["net_return"].dropna()
    return {
        "final_equity": eq.iloc[-1],
        "CAGR":   evaluate.cagr(eq),
        "Sharpe": evaluate.sharpe(daily),
        "MaxDD":  evaluate.max_drawdown(eq),
        "bench_final": synth_bench_prices.iloc[-1] / synth_bench_prices.iloc[0],
        "bench_CAGR":  evaluate.cagr(synth_bench_prices / synth_bench_prices.iloc[0]),
    }


def run_monte_carlo(
    stock_returns: pd.DataFrame,
    bench_returns: pd.Series,
    n_sims: int = config.MC_N_SIMULATIONS,
    n_days: int | None = None,
    seed: int = config.MC_SEED,
) -> pd.DataFrame:
    """Run `n_sims` Monte Carlo paths and return a DataFrame of per-path metrics."""
    if n_days is None:
        n_days = len(stock_returns.dropna(how="any"))

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        if (i + 1) % max(1, n_sims // 20) == 0:
            print(f"  MC {i + 1}/{n_sims}")
        rows.append(run_single_mc(stock_returns, bench_returns, n_days, rng))
    return pd.DataFrame(rows)
