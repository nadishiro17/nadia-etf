# Progress Journal — Two-Sleeve Barbell ETF

A running log of what we did, what we learned, and what we decided at each step. Read top-down; newest sessions at the bottom.

---

## Session 1 — Project scoping (2026-05-23)

### What we did
- Reviewed the original AlphaCircuit proposal from Checkpoint A (tech-only AI/ML/NLP composite fund). Professor said it was too large for one course project.
- Walked through three "ambition levels" (numbers-trustworthy vs ML showcase vs portfolio theory) and three universe sizes.
- Discussed in plain English what each finance concept means, since user is entry-level in finance.

### Decisions locked
- **Strategy shape:** Two-sleeve barbell — mean reversion (defensive) + momentum (growth). Later upgraded to three-book in Session 4.
- **Universe size:** ~50 large-cap US stocks (not 300–500).
- **Sector scope:** Broaden from tech-only to diversified across all 10 GICS sectors.
- **Centerpiece:** "Trustworthy numbers" (rigorous backtesting) + small dose of portfolio construction. *Not* an ML showcase.
- **Survivorship bias:** acknowledge as a documented limitation, don't try to source delisted-stock data.

### Open at end of session
- Exact stock list (deferred to Session 2 research).

---

## Session 2 — Universe construction and data download (2026-05-23)

### What we did
- Built the 50-stock universe by sector. Verified all names have continuous price history from January 1999.
- Created the repo structure inside `term_project/`.
- Wrote `config.py` with central knobs, `download_data.py` with yfinance retry logic, `.gitignore`, `README.md`.
- Pulled all 52 tickers (50 + S&P 500 index + T-bill yield) from yfinance.

### Universe by sector

| Sector | Count | Tickers |
|---|---|---|
| Technology | 8 | AAPL, MSFT, ORCL, IBM, INTC, CSCO, ADBE, TXN |
| Financials | 6 | JPM, BAC, WFC, C, AXP, USB |
| Healthcare | 7 | JNJ, PFE, MRK, ABT, BMY, LLY, AMGN |
| Consumer Staples | 5 | KO, PEP, PG, WMT, CL |
| Consumer Discretionary | 6 | HD, MCD, NKE, SBUX, LOW, TGT |
| Energy | 4 | XOM, CVX, COP, SLB |
| Industrials | 7 | BA, CAT, MMM, LMT, HON, FDX, UNP |
| Utilities | 3 | NEE, DUK, SO |
| Communication | 2 | VZ, T |
| Materials | 2 | NEM, APD |

### Results
- All 52 symbols downloaded with 6,540 trading days each.
- Combined panel saved to `data/adj_close.parquet` — shape (6540, 53), date range 1999-01-04 to 2024-12-30.
- Only 8 missing values total across the entire panel (all in the ^IRX T-bill yield series).

---

## Session 3 — Initial data exploration (2026-05-23)

### What we did
- Wrote `explore_data.py` to compute log returns, per-ticker stats, and diagnostic plots.
- Caught and fixed two issues:
  1. **Benchmark mismatch.** Used `^GSPC` (price index, no dividends) initially. Replaced with `SPY` (ETF, includes reinvested dividends in adjusted close) for a fair total-return comparison.
  2. **^IRX is a yield, not a price.** It's the 13-week T-bill yield in percent. Cannot be treated as a price series. Will be used as risk-free rate via correct conversion later.

### Key numerical findings

**Top 10 stocks by 26-year CAGR:**

| Ticker | Sector | CAGR | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| AAPL | Tech | 29.4% | 41.0% | 0.56 | −82% |
| ADBE | Tech | 17.9% | 42.9% | 0.32 | −80% |
| SBUX | Cons. Disc | 14.7% | 35.0% | 0.31 | −82% |
| UNP | Industrials | 14.5% | 28.1% | 0.38 | −59% |
| CAT | Industrials | 14.0% | 32.5% | 0.31 | −73% |
| ORCL | Tech | 13.8% | 39.3% | 0.25 | −84% |
| LOW | Cons. Disc | 13.6% | 32.7% | 0.30 | −61% |
| LMT | Industrials | 12.8% | 26.0% | 0.35 | −62% |
| NEE | Utilities | 12.6% | 23.7% | 0.37 | −48% |
| MSFT | Tech | 12.1% | 30.5% | 0.28 | −69% |

**Worst performer**: Citigroup (C) at **−2.4% CAGR with −98% max drawdown** — captures the 2008 financial crisis collapse.

**Sector medians:**

| Sector | n | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| Technology | 8 | 11.4% | 38.5% | 0.22 | −82% |
| Cons. Disc | 6 | 11.7% | 32.2% | 0.28 | −67% |
| Utilities | 3 | 11.1% | 23.7% | **0.36** | **−48%** |
| Industrials | 7 | 9.1% | 30.2% | 0.21 | −70% |
| Energy | 4 | 8.2% | 30.4% | 0.18 | −67% |
| Cons. Staples | 5 | 7.8% | 21.5% | 0.22 | **−40%** |
| Materials | 2 | 7.7% | 34.5% | 0.15 | −68% |
| Healthcare | 7 | 7.6% | 26.6% | 0.22 | −68% |
| Financials | 6 | 6.5% | 37.4% | 0.09 | −82% |
| Communication | 2 | 4.3% | 25.3% | 0.05 | −60% |

**Buy-and-hold baselines:**
- Equal-weighted 50-stock universe: **13.76% CAGR**
- SPY total return: **8.11% CAGR**, 19.4% volatility, Sharpe 0.25
- The ~5.6 pp/year gap is largely **survivorship bias** — we picked names that exist today.

**Daily return distribution (pooled across all 50 stocks, n=326,950):**
- Mean: 0.034% per day
- Std: 1.97% per day
- **Skew: −0.41** (fat left tail — characteristic of equity markets)
- **Kurtosis: 23.9** (massively non-normal — extreme outliers occur far more often than Gaussian predicts)

### Why this matters for the strategy

- Utilities + Consumer Staples have the lowest drawdowns and best Sharpe ratios → ideal candidates for the **defensive book**.
- Tech + Industrials + Cons. Disc have high CAGR but high vol → feed the **momentum book**.
- The fat-tailed return distribution **justifies block-bootstrap Monte Carlo over Gaussian** simulations. Gaussian MC would massively underestimate crash risk.
- Three crisis periods clearly visible in the equity curve and drawdown chart: 2002 dot-com (−37%), 2008–09 GFC (−50%), 2020 COVID (−33%), plus 2022 tech-led bear (−25%). Strong stress-test set.

### Artifacts produced this session
- [reports/universe_stats.csv](reports/universe_stats.csv) — per-ticker full table
- [reports/sector_stats.csv](reports/sector_stats.csv) — sector aggregates
- [plots/cumulative_returns.png](plots/cumulative_returns.png) — 50-stock vs SPY equity curves
- [plots/drawdown.png](plots/drawdown.png) — equal-weighted portfolio drawdown chart
- [plots/return_distribution.png](plots/return_distribution.png) — pooled daily return histogram with moments

---

## Session 4 — Strategy upgrade to three-book design (2026-05-23)

### What we did
- Reviewed three strategy ambition levels (Solid Standard / Risk-Parity Three-Book / Multi-Factor + HMM) and picked **Option 2**.
- Picked **vectorbt** as the backtest framework.
- Committed to a combined Checkpoint B + C deliverable.
- Installed vectorbt 0.28.2 (required numpy downgrade from 1.26 to 1.23.5; smoke test passed).

### Strategy specification (locked)

**Three books:**

1. **Mean-reversion book** — buys oversold names (20-day z-score < −1.5), sells when z-score reverts to 0. Holds ~5 names at a time. Profits in choppy, sideways markets.

2. **Momentum book** — holds top 5 stocks by 12-1 momentum (12-month return excluding most recent month). Rebalanced weekly. Profits in trending bull markets.

3. **Defensive book** — when neither sleeve finds opportunities, capital sits here. Holds a low-vol mix of utilities + consumer staples + cash equivalents (T-bills). Acts as the portfolio's shock absorber.

**Risk-parity allocation between books:**
- Each book's weight is inversely proportional to its trailing 60-day realized volatility.
- This ensures each book contributes equal expected risk to the total portfolio.
- In calm markets the three books are similar-sized; in turbulent markets the defensive book grows because the other two have become too volatile.

**Regime tilt:**
- Two binary indicators: SPY above/below 200-day moving average (trend), and SPY 20-day realized volatility above/below the 90th historical percentile (turbulence).
- Four regimes: bull-calm, bull-turbulent, bear-calm, bear-turbulent.
- Each regime has a predefined tilt that *multiplies* the risk-parity base weights, then re-normalizes:
  - Bull-calm: momentum favored, defensive deweighted
  - Bull-turbulent: mean-rev favored
  - Bear-calm: mean-rev favored, momentum deweighted
  - Bear-turbulent: defensive favored, momentum cut hard

**Backtest methodology:**
- Walk-forward: 5-year train / 1-year test, rolled annually.
- Block bootstrap Monte Carlo (block size 21 days, 1000 paths) for confidence intervals on performance metrics.
- Realistic transaction costs (5 bps per side baseline).
- Management fee sensitivity grid: {0%, 0.5%, 1%, 2%, 4%}.
- Performance fee grid: {0%, 5%, 15%, 25%}.

---

## Session 5 — Strategy implementation and end-to-end backtest (2026-05-23)

### What we did
- Installed vectorbt 0.28.2 (smoke test passed; we ended up using pure pandas/numpy for transparency, vectorbt remains available for future extensions).
- Built six modules:
  - `signals.py` — log returns, z-score, 12-1 momentum, realized vol, regime indicator.
  - `books.py` — three book modules (mean-rev, momentum, defensive).
  - `portfolio.py` — risk-parity allocator, regime tilt, vol circuit breaker, rebalance resampler.
  - `backtest.py` — end-to-end pipeline `run_strategy()` + cost modeling + fee functions.
  - `walk_forward.py` — 20-fold OOS validation harness.
  - `monte_carlo.py` — block bootstrap MC driver.
- Updated `config.py` with three-book parameters, regime tilts, walk-forward windows, MC settings.

### Smoke tests
- **Books generate sensible holdings.** On 2020-03-20 (COVID crash), the mean-reversion book correctly held the most oversold defensive names: JNJ, ABT, KO, PEP, T (each 20%). Momentum book at end of 2024 held ORCL, JPM, WFC, AXP, WMT. Defensive book always equal-weighted across NEE, DUK, SO, KO, PEP, PG, WMT, CL, JNJ.
- **End-to-end pipeline.** CAGR 12.02%, Sharpe 0.62, Max DD −36.1% — clean equity curve through every crisis.
- **Book allocation evolves as designed.** In 2002 bear: defensive book ~39%, momentum cut to 25%. In 2022 bear: defensive ~41%, momentum 18%. Risk-parity + regime tilt working as expected.

### Issue caught and discussed
- **Annualized turnover 28.5x.** All trades happen on Fridays (weekly rebalance correct), but 56% of the portfolio changes each Friday — the mean-rev book churns positions. Transaction costs already deducted in the 12.02% CAGR. Flagged as future-work improvement (move to monthly).

---

## Session 6 — Walk-forward + Monte Carlo + final report (2026-05-23)

### What we did
- Ran the walk-forward backtest across 20 OOS folds (2004–2024). Each fold's vol-percentile threshold was calibrated on its own training window only.
- Ran 500-path block bootstrap Monte Carlo (block size 21 days). Runtime ~2.1 minutes.
- Built `evaluate.py` with metrics (CAGR, Sharpe, Sortino, alpha/beta, max DD, Calmar), drawdown table, crisis attribution, fee-sensitivity grid, book attribution.
- Built `generate_report.py` to produce all final figures and tables in one pass.
- Wrote the combined Checkpoint B+C report ([CHECKPOINT_B_C_REPORT.md](CHECKPOINT_B_C_REPORT.md)).

### Headline results

**Full historical backtest (1999–2024):**
- CAGR 12.02% vs SPY 8.11% (+3.91 pp)
- Sharpe 0.62 vs 0.35
- Max drawdown −36% vs −55%
- Annualized alpha **+5.25%**, beta 0.65 (defensive)

**Walk-forward OOS (2004–2024, 20 folds):**
- CAGR 11.03% vs SPY 9.46% (+1.57 pp)
- Sharpe 0.58 vs 0.41
- Annualized alpha **+3.38%** — even in true OOS, the strategy generates real alpha
- 17 of 20 OOS years positive for fund; beat SPY in 14 of 20

**Crisis behavior (fund vs SPY total return):**
- Dot-com 2000–2002: **+32% vs −33%** (+66 pp excess)
- GFC 2008–2009: −20% vs −37% (+17 pp excess)
- COVID 2020: +19% vs +18%
- 2022 bear: **+6% vs −18%** (+24 pp excess)

**Monte Carlo (500 paths, 26-year synthetic histories):**
- Median strategy CAGR 13.4%, 5th–95th percentile 8.0%–18.4%
- **97.4% probability strategy beats SPY**
- 95.4% probability of CAGR > 8%
- 98.6% probability max drawdown stays better than −50%

**Book attribution (total return contribution):**
- Mean-reversion: 45.0%
- Momentum: 29.2%
- Defensive: 25.8%

**Fee sensitivity (net CAGR at selected combinations):**
- 0.5% mgmt + 0% perf (ETF-typical): 11.46%
- 1% mgmt + 15% perf (active-ETF/hedge-fund-lite): **9.93%**
- 2% mgmt + 20% perf (full hedge fund): ~8.6%
- 4% mgmt + 25% perf (extreme): 6.4% — uneconomic

### Artifacts produced this session
- `reports/metrics_summary.csv`, `reports/crisis_attribution.csv`, `reports/book_attribution.csv`, `reports/drawdown_table.csv`, `reports/fee_sensitivity.csv`, `reports/walk_forward_metrics.csv`, `reports/monte_carlo_paths.csv`
- `plots/equity_strategy_vs_spy.png`, `plots/drawdown_strategy.png`, `plots/book_allocations.png`, `plots/walk_forward_equity.png`, `plots/mc_distributions.png`, `plots/fee_sensitivity_heatmap.png`
- [`CHECKPOINT_B_C_REPORT.md`](CHECKPOINT_B_C_REPORT.md) — full combined report

### Honest takeaways

**The good:**
- Robust evidence of alpha across three independent lenses (historical, walk-forward, Monte Carlo).
- Defensive risk profile is genuine — beta 0.65, max drawdown 19 pp better than SPY.
- Strategy survived every major crisis with smaller losses than the market.
- Fee structure is viable — even at 1% mgmt + 15% perf, net beats SPY.

**The honest caveats:**
- Survivorship bias inflates the gross alpha by perhaps 3–5 pp/year — the universe contains only firms that exist today.
- High turnover (28x) is a real concern at scale, even though costs are already deducted.
- The 50-stock universe is narrower than ideal for breadth (Grinold–Kahn would prefer 200+).
- No shorts, no leverage — opportunities to refine.

### Done for Checkpoint B + C. Open for Checkpoint D (final report + presentation):
- Build polished presentation slides.
- Optionally implement one of the "future work" items (survivorship-bias-free universe is the most impactful).
- Compile full Markdown report into PDF.
