"""
explore_data.py — Initial data exploration and sanity checks.

Loads the adjusted-close panel, computes daily log returns, generates
per-ticker descriptive statistics, and produces a few diagnostic plots
to confirm the data is clean before any backtesting begins.

Outputs:
    reports/universe_stats.csv      per-ticker summary (return, vol, Sharpe, dd)
    reports/sector_stats.csv        per-sector aggregates
    plots/cumulative_returns.png    equal-weighted universe vs S&P 500
    plots/drawdown.png              drawdown of equal-weighted portfolio
    plots/return_distribution.png   histogram of daily log returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "adj_close.parquet")
    return df.sort_index()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns × 100 (per the assignment's specified formula)."""
    return 100.0 * np.log(prices / prices.shift(1))


def max_drawdown(equity: pd.Series) -> float:
    """Max peak-to-trough drawdown as a negative number, e.g. -0.43."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd.min()


def per_ticker_stats(prices: pd.DataFrame, rets_pct: pd.DataFrame) -> pd.DataFrame:
    """Build a per-ticker summary table."""
    # Convert log% returns back to simple decimal for cumulative product
    simple_rets = np.expm1(rets_pct / 100.0)
    equity = (1.0 + simple_rets.fillna(0)).cumprod()

    n_years = (prices.index[-1] - prices.index[0]).days / 365.25

    # rets_pct is in percent log-returns; convert to decimal for annualization
    daily_log = rets_pct / 100.0
    ann_log_return = daily_log.mean() * config.TRADING_DAYS_PER_YEAR
    ann_simple = np.expm1(ann_log_return)
    ann_vol = daily_log.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)

    sharpe = (ann_log_return - 0.03) / ann_vol  # using 3% as proxy r_f

    stats = pd.DataFrame({
        "Sector": [config.UNIVERSE.get(t, "Benchmark") for t in prices.columns],
        "Start": prices.apply(lambda s: s.first_valid_index().date()),
        "End": prices.apply(lambda s: s.last_valid_index().date()),
        "Total Return": equity.iloc[-1] - 1.0,
        "CAGR": (equity.iloc[-1]) ** (1 / n_years) - 1.0,
        "Ann Vol": ann_vol,
        "Sharpe (rf=3%)": sharpe,
        "Max Drawdown": equity.apply(max_drawdown),
    })
    return stats


def plot_cumulative(prices: pd.DataFrame, out_path):
    """Equal-weighted universe portfolio vs S&P 500 — buy-and-hold baseline."""
    fund_cols = [t for t in config.FUND_TICKERS if t in prices.columns]
    fund_prices = prices[fund_cols]

    # Normalize each ticker to start at 1.0 on its first valid date
    norm = fund_prices / fund_prices.iloc[0]
    ew = norm.mean(axis=1)                          # equal-weighted index
    spx = prices[config.BENCHMARK_TICKER]
    spx_norm = spx / spx.iloc[0]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(ew.index, ew.values, label="Equal-weighted 50-stock universe", lw=1.4)
    ax.plot(spx_norm.index, spx_norm.values,
            label="SPY (S&P 500 total return)", lw=1.2, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Buy-and-hold baseline, 1999–2024 (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return ew, spx_norm


def plot_drawdown(equity: pd.Series, out_path):
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.55)
    ax.set_title("Drawdown — equal-weighted universe")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_return_distribution(rets_pct: pd.DataFrame, out_path):
    """Daily log returns across all stocks pooled — sanity check on fat tails."""
    flat = rets_pct[config.FUND_TICKERS].stack().dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(flat.values, bins=200, density=True, alpha=0.7, color="steelblue")
    ax.set_xlim(-15, 15)
    ax.set_title(f"Daily log returns (×100), pooled across 50 stocks  "
                 f"(n={len(flat):,})")
    ax.set_xlabel("Daily log return (%)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    mu = flat.mean()
    sd = flat.std()
    sk = flat.skew()
    kt = flat.kurtosis()
    ax.text(0.02, 0.95,
            f"mean={mu:.3f}%\nstd={sd:.3f}%\nskew={sk:.2f}\nkurt={kt:.1f}",
            transform=ax.transAxes, va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    print("Loading prices ...")
    prices = load_prices()
    print(f"  panel shape: {prices.shape}")
    print(f"  date range:  {prices.index.min().date()} .. {prices.index.max().date()}")
    print(f"  tickers:     {list(prices.columns)}")

    n_missing = prices.isna().sum().sum()
    print(f"  total missing values: {n_missing}")

    print("\nComputing daily log returns ...")
    rets_pct = log_returns(prices)

    print("Building per-ticker stats ...")
    stats = per_ticker_stats(prices, rets_pct)
    stats_out = stats.sort_values("CAGR", ascending=False)

    out_csv = config.REPORTS_DIR / "universe_stats.csv"
    stats_out.to_csv(out_csv)
    print(f"  wrote {out_csv}")

    print("\nTop 10 by CAGR:")
    print(stats_out.head(10)
          [["Sector", "CAGR", "Ann Vol", "Sharpe (rf=3%)", "Max Drawdown"]]
          .round(3).to_string())

    print("\nBottom 5 by CAGR:")
    print(stats_out.tail(5)
          [["Sector", "CAGR", "Ann Vol", "Sharpe (rf=3%)", "Max Drawdown"]]
          .round(3).to_string())

    # Sector aggregates
    fund_stats = stats.loc[[t for t in config.FUND_TICKERS]]
    sec_stats = fund_stats.groupby("Sector").agg(
        n=("CAGR", "size"),
        median_CAGR=("CAGR", "median"),
        median_vol=("Ann Vol", "median"),
        median_sharpe=("Sharpe (rf=3%)", "median"),
        median_dd=("Max Drawdown", "median"),
    ).round(3)
    sec_csv = config.REPORTS_DIR / "sector_stats.csv"
    sec_stats.to_csv(sec_csv)
    print(f"\nSector aggregates -> {sec_csv}")
    print(sec_stats.to_string())

    print("\nBenchmark (S&P 500):")
    print(stats.loc[[config.BENCHMARK_TICKER]]
          [["CAGR", "Ann Vol", "Sharpe (rf=3%)", "Max Drawdown"]].round(3).to_string())

    print("\nGenerating plots ...")
    ew, spx_norm = plot_cumulative(prices, config.PLOTS_DIR / "cumulative_returns.png")
    plot_drawdown(ew, config.PLOTS_DIR / "drawdown.png")
    plot_return_distribution(rets_pct, config.PLOTS_DIR / "return_distribution.png")
    print(f"  wrote plots to {config.PLOTS_DIR}")

    # Quick buy-and-hold comparison
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    print(f"\nBuy-and-hold over {n_years:.1f} years:")
    print(f"  Equal-weighted 50-stock CAGR: "
          f"{(ew.iloc[-1])**(1/n_years) - 1:.2%}")
    print(f"  S&P 500 CAGR:                 "
          f"{(spx_norm.iloc[-1])**(1/n_years) - 1:.2%}")
    print(f"  EW final $1 -> ${ew.iloc[-1]:.2f}")
    print(f"  SPX final $1 -> ${spx_norm.iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
