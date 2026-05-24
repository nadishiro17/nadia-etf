"""
generate_report.py — Generate all final figures and tables for the
Checkpoint B + C report.

Outputs (all under plots/ and reports/):
    plots/equity_strategy_vs_spy.png
    plots/drawdown_strategy.png
    plots/book_allocations.png
    plots/mc_cagr_distribution.png
    plots/mc_drawdown_distribution.png
    plots/fee_sensitivity_heatmap.png
    plots/walk_forward_equity.png
    reports/metrics_summary.csv
    reports/crisis_attribution.csv
    reports/fee_sensitivity.csv
    reports/book_attribution.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import signals
import backtest
import walk_forward
import evaluate


def main():
    panel = pd.read_parquet(config.DATA_DIR / "adj_close.parquet")
    fund_prices = panel[config.FUND_TICKERS]
    spy = panel["SPY"]

    bench_rets_simple = signals.simple_returns(spy)
    spy_equity = spy / spy.iloc[0]

    print("=== Full historical backtest ===")
    result = backtest.run_strategy(
        fund_prices, spy, train_end=pd.Timestamp("2003-12-31")
    )
    sim = result["simulation"]
    eq = sim["equity"].dropna()
    daily = sim["net_return"].dropna()

    # ----- Metrics ----------------------------------------------------------
    strat_metrics = evaluate.metrics_table(
        eq, daily, bench_equity=spy_equity, bench_returns=bench_rets_simple,
        name="Strategy"
    )
    spy_metrics = evaluate.metrics_table(
        spy_equity, bench_rets_simple, bench_equity=spy_equity,
        bench_returns=bench_rets_simple, name="SPY"
    )
    metrics = pd.concat([strat_metrics, spy_metrics], axis=1)
    metrics.to_csv(config.REPORTS_DIR / "metrics_summary.csv")
    print("\n", metrics.round(4).to_string())

    # ----- Crisis attribution ----------------------------------------------
    crisis = evaluate.crisis_attribution(daily, bench_rets_simple)
    crisis.to_csv(config.REPORTS_DIR / "crisis_attribution.csv", index=False)
    print("\nCrisis attribution:")
    print(crisis.round(4).to_string(index=False))

    # ----- Book attribution ------------------------------------------------
    book_attr = evaluate.book_attribution(result["book_returns"], result["rp_tilted"])
    book_attr.to_csv(config.REPORTS_DIR / "book_attribution.csv")
    print("\nBook attribution:")
    print(book_attr.round(4).to_string())

    # ----- Drawdown table --------------------------------------------------
    dd_table = evaluate.drawdown_table(eq, top_n=5)
    dd_table.to_csv(config.REPORTS_DIR / "drawdown_table.csv", index=False)
    print("\nTop 5 drawdowns:")
    print(dd_table.round(4).to_string(index=False))

    # ----- Fee sensitivity -------------------------------------------------
    print("\nComputing fee sensitivity grid ...")
    fee_grid = evaluate.fee_sensitivity(eq, spy_equity)
    fee_grid.to_csv(config.REPORTS_DIR / "fee_sensitivity.csv")
    print(fee_grid.round(4).to_string())

    # ----- Walk-forward ----------------------------------------------------
    print("\n=== Walk-forward ===")
    wf = walk_forward.run_walk_forward(fund_prices, spy)
    wf_eq = wf["oos_equity"]
    wf_daily = wf["oos_returns"]

    spy_oos = spy.loc[wf_eq.index[0]:wf_eq.index[-1]]
    spy_oos_eq = spy_oos / spy_oos.iloc[0]
    spy_oos_daily = spy_oos.pct_change().dropna()

    wf_metrics = evaluate.metrics_table(
        wf_eq, wf_daily, bench_equity=spy_oos_eq, bench_returns=spy_oos_daily,
        name="Strategy (WF OOS)"
    )
    spy_oos_metrics = evaluate.metrics_table(
        spy_oos_eq, spy_oos_daily, bench_equity=spy_oos_eq,
        bench_returns=spy_oos_daily, name="SPY (same window)"
    )
    wf_combined = pd.concat([wf_metrics, spy_oos_metrics], axis=1)
    wf_combined.to_csv(config.REPORTS_DIR / "walk_forward_metrics.csv")
    print("\nWalk-forward OOS metrics:")
    print(wf_combined.round(4).to_string())

    # ----- Plots -----------------------------------------------------------
    print("\nGenerating plots ...")

    # 1. Strategy vs SPY equity
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(eq.index, eq.values, label="Strategy", lw=1.5, color="#1f4e79")
    ax.plot(spy_equity.index, spy_equity.values,
            label="SPY (total return)", lw=1.2, color="#c0504d", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Strategy vs SPY — growth of $1, 1999–2024 (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "equity_strategy_vs_spy.png", dpi=130)
    plt.close(fig)

    # 2. Drawdown comparison
    strat_dd = eq / eq.cummax() - 1
    spy_dd = spy_equity / spy_equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(strat_dd.index, strat_dd.values, 0, color="#1f4e79",
                    alpha=0.5, label="Strategy")
    ax.fill_between(spy_dd.index, spy_dd.values, 0, color="#c0504d",
                    alpha=0.3, label="SPY")
    ax.set_title("Drawdowns — Strategy vs SPY")
    ax.set_ylabel("Drawdown")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "drawdown_strategy.png", dpi=130)
    plt.close(fig)

    # 3. Book allocations over time
    rp = result["rp_tilted"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.stackplot(rp.index, rp["mean_rev"], rp["momentum"], rp["defensive"],
                 labels=["Mean-reversion", "Momentum", "Defensive"],
                 colors=["#2e75b6", "#5b9bd5", "#9dc3e6"], alpha=0.9)
    ax.set_title("Book allocation over time (risk-parity + regime tilt)")
    ax.set_ylabel("Allocation")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "book_allocations.png", dpi=130)
    plt.close(fig)

    # 4. Walk-forward equity vs SPY in same window
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wf_eq.index, wf_eq.values, label="Walk-forward OOS", lw=1.5, color="#1f4e79")
    ax.plot(spy_oos_eq.index, spy_oos_eq.values, label="SPY (same window)",
            lw=1.2, color="#c0504d", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Walk-forward out-of-sample equity (20 folds, 2004–2024)")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "walk_forward_equity.png", dpi=130)
    plt.close(fig)

    # 5. Monte Carlo distributions
    mc = pd.read_csv(config.REPORTS_DIR / "monte_carlo_paths.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(mc["CAGR"], bins=40, color="#1f4e79", alpha=0.7,
                 edgecolor="white", label="Strategy")
    axes[0].hist(mc["bench_CAGR"], bins=40, color="#c0504d", alpha=0.5,
                 edgecolor="white", label="SPY (bootstrap)")
    axes[0].axvline(0.0811, color="#c0504d", ls="--", lw=1.2,
                    label="SPY actual (8.11%)")
    axes[0].axvline(0.1202, color="#1f4e79", ls="--", lw=1.2,
                    label="Strategy actual (12.02%)")
    axes[0].set_xlabel("CAGR")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of 26-year CAGR across 500 MC paths")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].hist(mc["MaxDD"], bins=40, color="#1f4e79", alpha=0.7,
                 edgecolor="white", label="Strategy")
    axes[1].axvline(-0.5519, color="#c0504d", ls="--", lw=1.2,
                    label="SPY historical (−55%)")
    axes[1].axvline(-0.3614, color="#1f4e79", ls="--", lw=1.2,
                    label="Strategy historical (−36%)")
    axes[1].set_xlabel("Max drawdown")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of max drawdown across 500 MC paths")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "mc_distributions.png", dpi=130)
    plt.close(fig)

    # 6. Fee sensitivity heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(fee_grid * 100, annot=True, fmt=".2f",
                cmap="RdYlGn", center=8.11, ax=ax,
                cbar_kws={"label": "Net CAGR (%)"})
    ax.set_title("Net CAGR (%) — management fee × performance fee sensitivity")
    ax.set_xlabel("Performance fee (fraction of excess over SPY)")
    ax.set_ylabel("Management fee (annual)")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "fee_sensitivity_heatmap.png", dpi=130)
    plt.close(fig)

    print("\n=== ALL ARTIFACTS WRITTEN ===")
    print(f"  Tables in: {config.REPORTS_DIR}")
    print(f"  Plots in:  {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
