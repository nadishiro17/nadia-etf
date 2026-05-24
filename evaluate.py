"""
evaluate.py — Performance metrics, attribution, and fee sensitivity.

All metrics take an equity curve (pd.Series indexed by date, starting at 1.0)
and produce scalars or short tables. Designed to be called both for the
historical backtest and for each Monte Carlo path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config


# ---------------------------------------------------------------------------
# Core scalar metrics
# ---------------------------------------------------------------------------
def cagr(equity: pd.Series) -> float:
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    return equity.iloc[-1] ** (1.0 / n_years) - 1.0


def ann_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def sharpe(daily_returns: pd.Series, rf_annual: float = 0.03) -> float:
    daily_rf = rf_annual / config.TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def sortino(daily_returns: pd.Series, rf_annual: float = 0.03) -> float:
    daily_rf = rf_annual / config.TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0].std()
    if downside == 0 or pd.isna(downside):
        return np.nan
    return excess.mean() / downside * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def max_drawdown(equity: pd.Series) -> float:
    running = equity.cummax()
    return (equity / running - 1.0).min()


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return np.nan
    return cagr(equity) / dd


def alpha_beta(fund_returns: pd.Series, bench_returns: pd.Series, rf_annual: float = 0.03):
    """
    OLS regression: fund_excess = alpha + beta * bench_excess + eps.
    Alpha is reported annualized.
    """
    daily_rf = rf_annual / config.TRADING_DAYS_PER_YEAR
    y = (fund_returns - daily_rf).dropna()
    x = (bench_returns - daily_rf).reindex(y.index).dropna()
    common = y.index.intersection(x.index)
    y = y.loc[common]; x = x.loc[common]
    if len(y) < 30:
        return np.nan, np.nan
    slope, intercept, _, _, _ = stats.linregress(x.values, y.values)
    ann_alpha = intercept * config.TRADING_DAYS_PER_YEAR
    return ann_alpha, slope


# ---------------------------------------------------------------------------
# Composite metrics table
# ---------------------------------------------------------------------------
def metrics_table(
    equity: pd.Series,
    daily_returns: pd.Series,
    bench_equity: pd.Series | None = None,
    bench_returns: pd.Series | None = None,
    rf_annual: float = 0.03,
    name: str = "Strategy",
) -> pd.Series:
    out = {
        "CAGR":         cagr(equity),
        "Ann Vol":      ann_vol(daily_returns),
        "Sharpe":       sharpe(daily_returns, rf_annual),
        "Sortino":      sortino(daily_returns, rf_annual),
        "Max Drawdown": max_drawdown(equity),
        "Calmar":       calmar(equity),
        "Final $":      equity.iloc[-1],
    }
    if bench_returns is not None:
        a, b = alpha_beta(daily_returns, bench_returns, rf_annual)
        out["Alpha (ann)"] = a
        out["Beta"]        = b
    return pd.Series(out, name=name)


# ---------------------------------------------------------------------------
# Drawdown table
# ---------------------------------------------------------------------------
def drawdown_table(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """List the worst N drawdowns by depth, with start/end dates and recovery."""
    running = equity.cummax()
    dd = equity / running - 1.0

    in_dd = dd < 0
    if not in_dd.any():
        return pd.DataFrame(columns=["start", "trough", "end", "depth", "duration_days"])

    # Identify drawdown segments
    segments = []
    start = None
    for i, val in enumerate(in_dd):
        if val and start is None:
            start = i
        elif not val and start is not None:
            seg = (start, i - 1)
            segments.append(seg)
            start = None
    if start is not None:
        segments.append((start, len(in_dd) - 1))

    rows = []
    for s, e in segments:
        seg_dd = dd.iloc[s : e + 1]
        trough_pos = seg_dd.idxmin()
        rows.append({
            "start": equity.index[s].date(),
            "trough": trough_pos.date(),
            "end": equity.index[e].date(),
            "depth": seg_dd.min(),
            "duration_days": (equity.index[e] - equity.index[s]).days,
        })

    table = pd.DataFrame(rows).sort_values("depth").head(top_n).reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Crisis-period sub-test
# ---------------------------------------------------------------------------
CRISIS_WINDOWS = {
    "Dot-com bust (2000-2002)": ("2000-03-01", "2002-12-31"),
    "Global Financial Crisis (2008-2009)": ("2007-10-01", "2009-06-30"),
    "COVID crash (2020)": ("2020-02-01", "2020-12-31"),
    "Rates bear (2022)": ("2022-01-01", "2022-12-31"),
}


def crisis_attribution(
    daily_returns: pd.Series,
    bench_returns: pd.Series,
) -> pd.DataFrame:
    """Return strategy vs benchmark performance across each crisis window."""
    rows = []
    for label, (s, e) in CRISIS_WINDOWS.items():
        fr = daily_returns.loc[s:e]
        br = bench_returns.loc[s:e]
        fund_total = (1 + fr).prod() - 1
        bench_total = (1 + br).prod() - 1
        fund_dd = max_drawdown((1 + fr).cumprod())
        bench_dd = max_drawdown((1 + br).cumprod())
        rows.append({
            "window": label,
            "fund_total": fund_total,
            "bench_total": bench_total,
            "excess": fund_total - bench_total,
            "fund_max_dd": fund_dd,
            "bench_max_dd": bench_dd,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fee sensitivity grid
# ---------------------------------------------------------------------------
def fee_sensitivity(
    gross_equity: pd.Series,
    bench_equity: pd.Series,
    mgmt_grid: list = config.MGMT_FEE_GRID,
    perf_grid: list = config.PERF_FEE_GRID,
) -> pd.DataFrame:
    """
    Build a (mgmt × perf) grid of net-of-fee CAGR.

    Performance fee: annual high-water style, charged on excess return
    above SPY (simple per-year implementation).
    """
    from backtest import apply_management_fee, apply_performance_fee

    rows = []
    for m in mgmt_grid:
        for p in perf_grid:
            net = apply_management_fee(gross_equity, m)
            net = apply_performance_fee(net, bench_equity, p)
            rows.append({"mgmt_fee": m, "perf_fee": p, "net_CAGR": cagr(net)})
    df = pd.DataFrame(rows)
    return df.pivot(index="mgmt_fee", columns="perf_fee", values="net_CAGR")


# ---------------------------------------------------------------------------
# Sleeve / book attribution
# ---------------------------------------------------------------------------
def book_attribution(
    book_returns: pd.DataFrame,
    book_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Contribution of each book to total return.

    contribution_b = sum_t (w_b(t-1) × r_b(t))
    """
    contrib = (book_weights.shift(1) * book_returns).sum(axis=0)
    total = contrib.sum()
    pct = contrib / total if total != 0 else contrib
    return pd.DataFrame({"contribution": contrib, "pct_of_total": pct})
