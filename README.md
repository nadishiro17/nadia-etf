# Term Project — Three-Book Barbell ETF

A rules-based, actively-managed ETF combining a **mean-reversion book**, a **momentum book**, and a **defensive low-volatility book**, drawn from 50 large-cap US stocks diversified across 10 GICS sectors. Risk-parity allocation between books with a rule-based regime tilt.

**Author:** Nadezhda Shiroglazova
**Course:** MSDS 451 — Financial Engineering
**Repository:** https://github.com/nadishiro17/nadia-etf

## Headline results (1999–2024)

| Metric | Strategy | SPY |
|---|---|---|
| CAGR | **12.02%** | 8.11% |
| Annual volatility | 15.51% | 19.34% |
| Sharpe (rf=3%) | **0.62** | 0.35 |
| Max drawdown | **−36.1%** | −55.2% |
| Annualized alpha | **+5.25%** | 0% |

Walk-forward OOS (20 folds, 2004–2024): **CAGR 11.03%, Sharpe 0.58, alpha +3.38%**.
Monte Carlo (500 paths): **97.4% probability of beating SPY**.

Full results in [CHECKPOINT_B_C_REPORT.md](CHECKPOINT_B_C_REPORT.md).

## Investment philosophy


Two uncorrelated engines run in parallel:

1. **Mean-reversion sleeve (50% of capital).** Buys stocks that have sold off sharply relative to their 20-day moving average (z-score < −1.5). Sells when they revert to neutral. Profits in choppy, sideways markets.
2. **Momentum sleeve (50% of capital).** Holds the top 5 stocks by 12-1 momentum (12-month return excluding the most recent month). Rebalanced weekly. Profits in trending markets.
3. **Risk overlay.** When portfolio 20-day realized volatility exceeds 30% annualized, positions scale to 50% and the remainder goes to cash. Per-name hard stop at −15%.

## Data

- **Universe:** 50 large-cap US stocks (see [config.py](config.py)) across 10 sectors, all with continuous 1999–2024 history.
- **Benchmark:** S&P 500 (`^GSPC`).
- **Risk-free rate:** 13-week T-bill (`^IRX`).
- **Source:** Yahoo Finance via `yfinance` (daily OHLCV + adjusted close for total return).

## Repo layout

```
term_project/
├── config.py              # all knobs: universe, dates, fees, strategy params
├── download_data.py       # one-shot data pull
├── data/                  # generated parquet files (gitignored)
├── plots/                 # generated charts (gitignored)
├── reports/               # generated tables + final report
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python download_data.py
```

## Checkpoint plan

- **Checkpoint A** (literature review): submitted.
- **Checkpoint B** (data + initial backtest): this checkpoint.
- **Checkpoint C** (Monte Carlo + walk-forward + fees): upcoming.

## Known limitations

- **Survivorship bias.** The universe contains only firms that exist today. Companies that went bankrupt during 1999–2024 (Lehman, Bear Stearns, Enron, etc.) are excluded, which inflates realized returns relative to an investable strategy at the time. Discussed honestly in the report.
- **M&A continuity.** Several tickers (JPM, BAC, C, VZ, XOM, CVX, COP) reflect entities that underwent major mergers in 1999–2002. yfinance returns the continuous adjusted ticker series, which is the standard treatment but not identical to a clean single-entity history.
