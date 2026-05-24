"""
backtest.py — End-to-end backtest: assemble books, allocate, simulate trades,
return equity curve and per-day diagnostics.

Pure pandas/numpy implementation. vectorbt is used in monte_carlo.py for
the bulk MC runs, but for the single historical backtest we keep things
transparent and debuggable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import signals
import books
import portfolio


def simulate(
    stock_weights: pd.DataFrame,
    stock_returns: pd.DataFrame,
    commission_per_side: float = config.COMMISSION_PER_TRADE,
) -> pd.DataFrame:
    """
    Apply weights to returns with transaction costs.

    Cost model: on each rebalance, the absolute change in each stock's
    weight is the "turnover." Cost = turnover × commission_per_side
    (covers commission + half-spread + slippage on both legs).

    Returns DataFrame with daily columns:
        gross_return  — w_{t-1} · r_t
        turnover      — sum(|w_t - w_{t-1}|)
        tx_cost       — turnover × commission_per_side
        net_return    — gross_return - tx_cost (on the day cost is incurred)
        equity        — cumulative net wealth, starting at 1
    """
    w = stock_weights.fillna(0.0)
    r = stock_returns.fillna(0.0)
    aligned_w = w.shift(1).fillna(0.0)

    gross = (aligned_w * r).sum(axis=1)

    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    tx_cost = turnover * commission_per_side

    net = gross - tx_cost
    equity = (1.0 + net).cumprod()

    return pd.DataFrame({
        "gross_return": gross,
        "turnover": turnover,
        "tx_cost": tx_cost,
        "net_return": net,
        "equity": equity,
    })


def apply_management_fee(equity: pd.Series, annual_rate: float) -> pd.Series:
    """
    Convert an equity curve from gross-of-fees to net-of-management-fee.
    Daily fee = annual_rate / 252, deducted from each day's return.
    """
    if annual_rate <= 0:
        return equity
    daily_fee = annual_rate / config.TRADING_DAYS_PER_YEAR
    daily_net = equity.pct_change().fillna(0.0) - daily_fee
    return (1.0 + daily_net).cumprod()


def apply_performance_fee(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    perf_rate: float,
    freq: str = "YE",
) -> pd.Series:
    """
    Performance fee charged at year-end on returns above benchmark (high-water).

    On each year-end:
      - Compute fund return for the year and benchmark return for the year.
      - If fund > benchmark: charge perf_rate × (fund - benchmark) on principal.
    """
    if perf_rate <= 0:
        return equity

    # Align
    eq = equity.copy()
    bench = benchmark_equity.reindex(eq.index).ffill()

    out = eq.copy()
    year_ends = eq.resample(freq).last().index
    cumulative_fee_drag = 1.0  # carry forward as a multiplier

    prev_idx = eq.index[0]
    prev_eq = eq.iloc[0]
    prev_bench = bench.iloc[0]
    for ye in year_ends:
        if ye not in eq.index:
            ye = eq.loc[:ye].index[-1]
        fund_ret = eq.loc[ye] / prev_eq - 1.0
        bench_ret = bench.loc[ye] / prev_bench - 1.0
        excess = fund_ret - bench_ret
        if excess > 0:
            fee = perf_rate * excess
            # Apply fee at ye — scale all values from ye onward
            mask = out.index >= ye
            out.loc[mask] = out.loc[mask] * (1.0 - fee)
        prev_eq = eq.loc[ye]
        prev_bench = bench.loc[ye]
        prev_idx = ye
    return out


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_strategy(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    train_end: pd.Timestamp | None = None,
    apply_circuit_breaker: bool = True,
) -> dict:
    """
    Full pipeline:
        prices → signals → books → risk parity → regime tilt → combine →
        circuit breaker → weekly resample → simulate

    Returns a dict with all intermediate artifacts plus the final equity curve.
    """
    rets = signals.log_returns(prices)
    simple_rets = signals.simple_returns(prices)
    bench_rets = signals.simple_returns(benchmark_prices)

    # Books
    mr_w  = books.mean_reversion_book(prices)
    mom_w = books.momentum_book(prices)
    def_w = books.defensive_book(prices)

    # Book returns
    book_rets = pd.DataFrame({
        "mean_rev":  portfolio.book_returns(mr_w,  simple_rets),
        "momentum":  portfolio.book_returns(mom_w, simple_rets),
        "defensive": portfolio.book_returns(def_w, simple_rets),
    })

    # Risk-parity allocation between books
    rp = portfolio.risk_parity_weights(book_rets)

    # Regime tilt
    reg = signals.regime(benchmark_prices, bench_rets, train_end=train_end)
    rp_tilted = portfolio.apply_regime_tilt(rp, reg)

    # Combine into stock-level weights
    stock_w = portfolio.combine_books(
        {"mean_rev": mr_w, "momentum": mom_w, "defensive": def_w},
        rp_tilted,
    )

    # Risk overlay
    if apply_circuit_breaker:
        stock_w = portfolio.apply_vol_circuit_breaker(stock_w, simple_rets)

    # Weekly rebalance cadence
    stock_w = portfolio.resample_to_rebalance(stock_w)

    # Simulate
    sim = simulate(stock_w, simple_rets)

    return {
        "prices": prices,
        "stock_returns": simple_rets,
        "book_weights": {"mean_rev": mr_w, "momentum": mom_w, "defensive": def_w},
        "book_returns": book_rets,
        "rp_weights": rp,
        "rp_tilted": rp_tilted,
        "regime": reg,
        "stock_weights": stock_w,
        "simulation": sim,
    }
