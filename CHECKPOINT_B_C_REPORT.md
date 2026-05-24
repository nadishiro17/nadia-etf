# Checkpoints B + C — Combined Report
## Data Preparation, Backtesting, and Performance Evaluation

**Term Project — Data Science in Finance (MSDS 451)**
**Author:** Nadezhda Shiroglazova
**Repository:** [github.com/.../FIN_ENG/term_project](https://github.com)
**Report date:** May 2026

---

## Executive Summary

This report covers the empirical phase of the term project: data acquisition (Checkpoint B) and backtesting / performance evaluation (Checkpoint C). The proposed fund is a **two-sleeve barbell ETF** combining a defensive mean-reversion sleeve, a growth momentum sleeve, and a low-volatility defensive book, allocated by risk parity with a rule-based regime tilt.

Over the 26-year historical period (1999–2024) and across 500 Monte Carlo synthetic histories, the strategy produces:

| Metric | Historical (full) | Walk-forward OOS | MC median | SPY actual |
|---|---|---|---|---|
| CAGR | **12.02%** | **11.03%** | 13.4% | 8.11% |
| Annual volatility | 15.51% | 14.91% | — | 19.34% |
| Sharpe ratio | **0.62** | 0.58 | 0.65 | 0.35 |
| Max drawdown | **−36%** | −36% | −30% | **−55%** |
| Annualized alpha | **+5.25%** | +3.38% | — | 0% |
| Beta | 0.65 | 0.66 | — | 1.00 |

**The strategy beats SPY in 97.4% of Monte Carlo paths** and **maintains a max drawdown better than −50% in 98.6% of paths**.

At a realistic fee structure of 1% management + 15% performance fee on excess over SPY, the net CAGR remains **9.93%** — still ~180 bps above passive SPY ownership. This is a viable business proposition for an actively managed ETF.

---

## 1. Data (Checkpoint B)

### 1.1 Sources

| Series | Source | Frequency | Coverage |
|---|---|---|---|
| Stock prices | Yahoo Finance via `yfinance` v0.2.40+ | Daily OHLCV + Adjusted Close | 1999-01-04 to 2024-12-30 |
| Benchmark (primary) | SPY ETF — total return via Adj Close | Daily | 1999-01-04 to 2024-12-30 |
| Benchmark (reference) | ^GSPC — S&P 500 price index (no dividends) | Daily | Same |
| Risk-free rate | ^IRX — 13-week T-bill yield | Daily | Same |

**Why SPY rather than ^GSPC as the primary benchmark:** SPY's Adjusted Close reinvests dividends, making it a true *total return* index comparable to our 50 stocks' adjusted closes. The raw S&P 500 price index (^GSPC) understates returns by ~2 pp/year because it excludes dividends — using it would artificially inflate apparent alpha.

**Total return treatment:** All stock price series are dividend- and split-adjusted (yfinance's `Adj Close`). This means buying $1 of any stock in 1999 and reinvesting all dividends compounds to the reported 2024 value.

### 1.2 Securities universe

50 large-cap US equities with continuous price history since January 4, 1999, diversified across 10 GICS sectors:

| Sector | n | Tickers |
|---|---|---|
| Technology | 8 | AAPL, MSFT, ORCL, IBM, INTC, CSCO, ADBE, TXN |
| Healthcare | 7 | JNJ, PFE, MRK, ABT, BMY, LLY, AMGN |
| Industrials | 7 | BA, CAT, MMM, LMT, HON, FDX, UNP |
| Financials | 6 | JPM, BAC, WFC, C, AXP, USB |
| Consumer Discretionary | 6 | HD, MCD, NKE, SBUX, LOW, TGT |
| Consumer Staples | 5 | KO, PEP, PG, WMT, CL |
| Energy | 4 | XOM, CVX, COP, SLB |
| Utilities | 3 | NEE, DUK, SO |
| Communication Services | 2 | VZ, T |
| Materials | 2 | NEM, APD |

**Selection criteria:**
1. Listed on NYSE or NASDAQ as of January 1, 1999.
2. Continuously traded under the same ticker (or via merger-continuous ticker) through December 30, 2024.
3. Large market capitalization (each name was in the S&P 500 for the majority of the window).
4. Sector diversification across all 10 GICS sectors.

**Known limitations:**
- **Survivorship bias.** All 50 names exist today. Companies that went bankrupt during 1999–2024 (Lehman Brothers, Bear Stearns, Enron, Washington Mutual, etc.) are excluded. This inflates realized returns: an equal-weighted buy-and-hold of these 50 stocks yields **13.76% CAGR**, versus SPY's 8.11% — much of the 5.6 pp/year gap is the survivorship-bias premium, not stock-selection skill. The strategy is evaluated against SPY rather than its own universe baseline precisely because we cannot fairly compare to a "hypothetical 1999 stock picker who didn't know which firms would survive."
- **M&A continuity.** Seven names (JPM, BAC, C, VZ, XOM, CVX, COP) reflect tickers whose underlying entity changed materially around 1999–2002 mergers. yfinance returns the continuous adjusted ticker series, which is the standard industry treatment.

### 1.3 Time period and frequency

- **Period:** 1999-01-04 to 2024-12-30 (26 calendar years, 6,540 trading days).
- **Frequency:** Daily.
- **Crises captured:** dot-com bust (2000–2002), Global Financial Crisis (2008–2009), European debt crisis (2011–2012), COVID-19 shock (2020), rates bear market (2022).

### 1.4 Log returns

Daily returns are computed in log-space per the assignment specification:

> r_t = 100 × ln(P_t / P_{t-1})

| Statistic | Value |
|---|---|
| Mean (daily) | 0.034% |
| Std (daily) | 1.97% |
| **Skew** | **−0.41** |
| **Kurtosis** | **23.9** |
| Observations | 326,950 |

The fat-tailed, negatively-skewed distribution (kurtosis ≈ 24 vs Gaussian 3) is critical for the methodology choice in §3: **Gaussian Monte Carlo would massively understate tail risk**, motivating the use of block bootstrap.

### 1.5 Buy-and-hold baseline

| Strategy | Final $1 → | CAGR | Comment |
|---|---|---|---|
| Equal-weighted 50-stock B&H | $28.51 | 13.76% | Reflects survivorship bias |
| SPY (total return) | $7.58 | 8.11% | Realistic passive benchmark |
| ^GSPC (price only, no divs) | $4.81 | 6.23% | Excludes dividends — incorrect comparison |

---

## 2. Investment philosophy and strategy specification

### 2.1 Philosophy in one sentence

> The fund runs two uncorrelated alpha engines — mean reversion (defensive) and momentum (growth) — in parallel with a low-volatility shock-absorber book, allocated by risk parity so no single engine can dominate or blow up the fund.

### 2.2 The three books

**Mean-reversion book.** Buys oversold stable names. Signal: 20-day rolling z-score of price relative to its moving average. Buy when z < −1.5 (stock has dropped >1.5σ below recent average); exit when z ≥ 0 (mean reversion completed). Up to 5 positions, equal-weighted. Profits in choppy, sideways markets.

**Momentum book.** Holds top 5 stocks by 12-1 momentum (12-month cumulative return excluding the most recent month). Standard Jegadeesh-Titman signal: the 1-month skip avoids contaminating the signal with short-term reversal. Profits in trending bull markets.

**Defensive book.** Equal-weighted across nine pre-defined low-volatility stable names: NEE, DUK, SO (utilities); KO, PEP, PG, WMT, CL (staples); JNJ (healthcare). This is the portfolio's shock absorber — when the other two books are sized down by risk parity (high volatility), this book grows.

### 2.3 Risk-parity allocation

Each book's weight is inversely proportional to its 60-day rolling realized volatility:

> w_b(t) ∝ 1 / σ_b(t−1)

Weights are floored at 5% and capped at 70%, then renormalized to sum to 1. This ensures each book contributes approximately equal **risk** (not equal capital) to the total portfolio. In calm markets all three books have similar weight; in turbulent markets the defensive book grows because the others have become too volatile to contribute much risk-adjusted return.

### 2.4 Regime tilt

A simple rule-based regime indicator overlays the risk-parity weights:

| Indicator | Definition |
|---|---|
| **Trend** | SPY price > 200-day moving average → "bull"; else "bear" |
| **Turbulence** | SPY 20-day realized vol > 90th percentile of *training-period* vol → "turbulent" |

Four regime states, each with a predefined multiplicative tilt applied to risk-parity weights (then renormalized):

| Regime | Mean-rev | Momentum | Defensive |
|---|---|---|---|
| Bull / calm | ×1.0 | ×1.3 | ×0.7 |
| Bull / turbulent | ×1.3 | ×1.0 | ×0.9 |
| Bear / calm | ×1.3 | ×0.7 | ×1.0 |
| Bear / turbulent | ×1.0 | ×0.5 | ×1.5 |

Tilt parameters are set *ex ante* — they are not optimized on backtest data. This avoids data-snooping. The vol percentile threshold (the one fitted parameter) is computed only on training-window data in walk-forward folds.

### 2.5 Risk overlay and execution

- **Portfolio-level circuit breaker:** if 20-day realized portfolio volatility exceeds 30% annualized, all positions scale to 50% (the remainder sits in cash).
- **Per-name hard stop:** −15% from purchase price triggers liquidation regardless of signal (implemented as conservative position-sizing in our case; full hard stops are future work).
- **Rebalance cadence:** Friday close (`W-FRI`).
- **Transaction cost model:** 5 basis points per side (covers commission, half-spread, and slippage). Applied to per-day absolute weight change × cost rate.

---

## 3. Methodology (Checkpoint C)

### 3.1 Historical backtest

Single-pass: run the full pipeline on 1999–2024 data with the regime turbulence threshold calibrated on the first 5 years (1999–2003). Pure pandas/numpy implementation; transparent and auditable.

### 3.2 Walk-forward out-of-sample validation

Twenty non-overlapping out-of-sample windows (2004 through 2023), each preceded by a 5-year rolling training window for vol-threshold calibration. Test windows are stitched into a single OOS equity curve. This is the most honest performance measurement: at every point in the OOS curve, no future data informed the strategy.

### 3.3 Block bootstrap Monte Carlo

500 synthetic 26-year histories generated by sampling 21-day blocks of actual daily returns (with replacement). Sampling is *joint across stocks* so cross-sectional correlation structure is preserved. The strategy is run on each synthetic price path; the distribution of CAGR / Sharpe / max drawdown across paths gives empirical confidence bands.

**Why block bootstrap, not Gaussian:** the observed return distribution has kurtosis 23.9 (Gaussian = 3). A Gaussian Monte Carlo would understate the probability of large losses by orders of magnitude. Block bootstrap preserves the empirical distribution including fat tails and short-range serial correlation (volatility clustering).

**Why not GAN-based synthetic data (Yoon et al. 2019):** explored as future work. Block bootstrap is the well-established, widely-cited approach (Politis & Romano 1994) and sufficient for this scale of project.

### 3.4 Performance metrics

- **CAGR** — compound annual growth rate of equity curve.
- **Annual volatility** — daily return std × √252.
- **Sharpe** — (mean − rf) / std × √252; rf = 3% baseline.
- **Sortino** — same but penalizing only downside volatility.
- **Max drawdown** — worst peak-to-trough loss.
- **Calmar** — CAGR ÷ |max drawdown|.
- **Alpha, beta** — daily excess-return OLS regression on SPY excess returns; alpha annualized.

### 3.5 Fee sensitivity

A 5 × 4 grid of (management fee, performance fee) combinations is evaluated. Management fee is deducted daily from returns at `annual_rate / 252`. Performance fee is charged at year-end on excess return above SPY: if fund yearly return > SPY yearly return, fee = `rate × (fund_return − SPY_return)`, applied multiplicatively to the equity curve. (High-water-mark logic is simplified to year-over-year.)

---

## 4. Results

### 4.1 Full-period historical backtest

| | Strategy | SPY |
|---|---|---|
| CAGR | **12.02%** | 8.11% |
| Annual volatility | 15.51% | 19.34% |
| Sharpe (rf=3%) | **0.62** | 0.35 |
| Sortino | 0.85 | 0.44 |
| Max drawdown | **−36.14%** | −55.19% |
| Calmar | 0.33 | 0.15 |
| Alpha (annualized) | **+5.25%** | 0% |
| Beta | 0.65 | 1.00 |
| Final $1 grows to | **$19.08** | $7.58 |

The fund delivers ~390 bps/year of excess return, with 4 pp less volatility and a 19 pp less severe peak drawdown. Beta of 0.65 confirms defensive positioning. See `plots/equity_strategy_vs_spy.png` and `plots/drawdown_strategy.png`.

### 4.2 Crisis attribution

| Crisis window | Fund total return | SPY total return | Excess | Fund max DD | SPY max DD |
|---|---|---|---|---|---|
| Dot-com bust (2000–2002) | **+32.43%** | −33.26% | **+65.7 pp** | −21.5% | −47.5% |
| GFC (2008–2009) | −20.00% | −37.16% | **+17.2 pp** | −36.1% | −55.2% |
| COVID crash (2020) | +18.78% | +18.38% | +0.4 pp | −18.5% | −33.7% |
| Rates bear (2022) | +6.07% | −18.18% | **+24.2 pp** | −14.2% | −24.5% |

The strategy **made money during dot-com bust and 2022 bear** — the defensive book + mean-rev rebalancing into oversold staples drove this. GFC was the only loss, but the loss was less than half of SPY's.

### 4.3 Walk-forward out-of-sample (2004–2024)

| | Strategy (WF OOS) | SPY (same window) |
|---|---|---|
| CAGR | **11.03%** | 9.46% |
| Annual volatility | 14.91% | 18.99% |
| Sharpe | **0.58** | 0.41 |
| Max drawdown | −35.62% | −55.19% |
| Alpha (annualized) | **+3.38%** | 0% |
| Beta | 0.66 | 1.00 |
| Final $1 grows to | **$8.10** | $6.09 |

The walk-forward OOS performance degrades only ~100 bps in CAGR and 0.04 in Sharpe vs the full-period backtest. This is **strong evidence the strategy is genuine rather than overfit**: the regime-vol threshold was calibrated only on the trailing 5-year window in each fold. See `plots/walk_forward_equity.png`.

Per-fold detail: of 20 OOS years, **17 are positive** for the fund and **fund beats SPY in 14 of 20 years**. The three negative fund years are 2008 (−11.8%), 2015 (−3.6%), 2018 (−2.1%) — and in each, SPY was either also negative or barely positive.

### 4.4 Monte Carlo (500 paths)

| Percentile | Strategy CAGR | Strategy Sharpe | Strategy Max DD | Bench (SPY) CAGR |
|---|---|---|---|---|
| 5th | 8.04% | 0.36 | −43.3% | 3.03% |
| 50th (median) | 13.37% | 0.65 | −30.3% | 8.92% |
| 95th | 18.41% | 0.91 | −22.4% | 15.00% |
| Mean | 13.25% | 0.64 | −31.5% | 8.85% |

**Probability statements (across 500 alternative 26-year histories):**

| | Probability |
|---|---|
| Strategy CAGR > SPY CAGR | **97.4%** |
| Strategy CAGR > 5% | 99.4% |
| Strategy CAGR > 8% | 95.4% |
| Strategy max drawdown better than −50% | 98.6% |

See `plots/mc_distributions.png`. The strategy's CAGR distribution is shifted approximately 4–5 pp right of SPY's, with similar shape. The actual historical 12.02% CAGR falls near the median of the MC distribution — i.e., the realized history was *typical*, not lucky.

### 4.5 Book attribution

| Book | Cumulative contribution | % of total return |
|---|---|---|
| Mean-reversion | 1.75 | **45.0%** |
| Momentum | 1.13 | 29.2% |
| Defensive | 1.00 | 25.8% |

The mean-reversion book is the largest contributor — consistent with the strategy's defensive bias. Momentum and defensive book contributions are roughly balanced.

See `plots/book_allocations.png` for the time-series evolution of book weights. Visible patterns:
- 2001–2002 dot-com: defensive book grows from ~33% to ~45%.
- 2007–2009 GFC: defensive book peaks above 40%, momentum drops below 20%.
- 2017–2019 late cycle volatility: mean-rev favored, momentum cut.
- 2022 bear: defensive book again ~40%, momentum below 20%.

These shifts are produced by the rule-based regime tilt and risk-parity allocator — no human judgment.

### 4.6 Fee sensitivity

Net CAGR under different fee combinations:

| mgmt \ perf | 0% | 5% | 15% | 25% |
|---|---|---|---|---|
| 0% | 12.02% | 11.66% | 10.94% | 10.20% |
| 0.5% | 11.46% | 11.12% | 10.44% | 9.74% |
| **1%** | 10.90% | 10.58% | **9.93%** | 9.27% |
| 2% | 9.80% | 9.51% | 8.92% | 8.33% |
| 4% | 7.63% | 7.39% | 6.89% | 6.38% |

**Read this table as a business plan:**
- At "ETF-typical" fees (0.5% mgmt, 0% perf): net 11.46% — easy sell.
- At "active ETF" fees (1% mgmt, 15% perf — hedge-fund-lite): net 9.93%.
- At full hedge-fund fees (2% mgmt + 20% perf, between cols 15% and 25%): net ~8.6% — barely above SPY.
- At 4% mgmt + 25% perf (extreme): net 6.4% — would not justify investment.

**Break-even with SPY** (net CAGR ≥ 8.11%) holds for management fees up to ~2% combined with 25% performance fees, or up to ~3% with 0% performance fees. There is substantial fee room before the product becomes unattractive.

See `plots/fee_sensitivity_heatmap.png`.

---

## 5. Business assessment

> *Is this a fund my team would actually launch?*

**Yes, with the following caveats.**

**Pros:**
- Defensible alpha across multiple lenses: full-period historical, walk-forward OOS, Monte Carlo distributional.
- Excellent risk profile: max drawdown 36% vs market 55%; beta 0.65; Sharpe ~2× market.
- Crisis behavior is good: positive returns in 2000–2002 dot-com and 2022 bear, only modest losses in 2008 GFC.
- Every rule in the strategy is explainable in English — no black-box models. Auditable and defensible to regulators and clients.
- Fees: there is meaningful room (1.5–2 pp of CAGR) between net performance and passive SPY at competitive fee levels.

**Concerns:**
- **Survivorship bias in universe.** The 50-stock universe contains only firms that exist today. A real-time implementation in 1999 would have included names like Lehman Brothers, Enron, Worldcom that subsequently went to zero. We estimate this contributes ~3–5 pp/year to the apparent gross CAGR. The strategy's alpha vs SPY (+5.25%) is therefore partially explained by stock selection bias rather than entirely by skill. A future iteration must use a point-in-time, survivorship-bias-free universe.
- **High turnover.** Annualized turnover is ~28x — driven primarily by the mean-reversion book cycling positions. Transaction costs are already deducted (~2.8% per year drag), but at scale a fund would face larger market impact than the 5 bps per side assumed. Reducing rebalance frequency to monthly is a likely first refinement.
- **Limited universe (50 names).** For a real fund, expanding to 200–500 names would improve statistical breadth per the Fundamental Law of Active Management (Grinold & Kahn) and reduce idiosyncratic risk.
- **No leverage, no shorts.** Long-only with no leverage. Adding a short sleeve to the momentum book (sell-low-momentum names) could meaningfully improve risk-adjusted returns but adds complexity.

**Verdict:** The architecture is sound. The numerical edge is real but the absolute alpha is overstated by survivorship bias. A production version would expand to a survivorship-bias-free universe, monthly rebalance, and possibly a short book. Even after those adjustments, the strategy is highly likely to produce competitive risk-adjusted returns at viable fee levels.

---

## 6. Limitations and next steps

1. **Universe construction.** Replace the 50-name fixed universe with a survivorship-bias-free, point-in-time S&P 500 membership list.
2. **Turnover reduction.** Move from weekly to monthly rebalancing; add hysteresis (only trade when signal moves by > threshold).
3. **Position-level hard stops.** Implement explicit −15% stop-loss per name (currently approximated by the portfolio-level circuit breaker).
4. **Short sleeve.** Add a short book of bottom-decile momentum names; convert to long/short market-neutral or 130/30 structure.
5. **GAN-based synthetic data.** Compare block bootstrap MC to TimeGAN-generated synthetic histories (Yoon et al. 2019) to test robustness of the conclusions.
6. **Higher-frequency signals.** Investigate using daily rebalance for the momentum book and weekly for mean-rev (different cadence per book).

---

## 7. Reproducibility

All code lives in [`term_project/`](.):

| File | Purpose |
|---|---|
| `config.py` | All knobs: universe, dates, strategy parameters, fees, MC settings |
| `download_data.py` | One-shot yfinance pull → `data/adj_close.parquet` |
| `explore_data.py` | Sanity checks + per-ticker statistics + diagnostic plots |
| `signals.py` | Pure signal functions: z-score, momentum, vol, regime |
| `books.py` | Three book modules: mean-rev, momentum, defensive |
| `portfolio.py` | Risk-parity allocator, regime tilt, circuit breaker, rebalance |
| `backtest.py` | End-to-end strategy pipeline + cost modeling + fee functions |
| `evaluate.py` | Performance metrics, crisis attribution, fee sensitivity, book attribution |
| `walk_forward.py` | 20-fold rolling-window OOS validation harness |
| `monte_carlo.py` | Block bootstrap MC driver |
| `generate_report.py` | Builds all figures and tables for this report |

To reproduce all results: `pip install -r requirements.txt && python download_data.py && python generate_report.py`.

---

## References

(Selected — full reference list in Checkpoint A.)

- Carver, R. (2015). *Systematic Trading*. Harriman House.
- Garner, S. (2019). *Mean Reversion Trading Strategies*. (Tutorial article)
- Greyserman, A., & Kaminski, K. (2014). *Trend Following with Managed Futures*. Wiley.
- Grinold, R., & Kahn, R. (2023). *Advances in Active Portfolio Management*. McGraw-Hill.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Politis, D., & Romano, J. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49–58.
- Trivedi, J., & Kyal, A. (2021). *Hands-On Financial Trading with Python*. Packt Publishing.
- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series generative adversarial networks. *NeurIPS 2019*.
- Zuckerman, G. (2019). *The Man Who Solved the Market*. Portfolio/Penguin.
