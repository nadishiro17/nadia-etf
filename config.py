"""
config.py — Central configuration for the term project.

Two-sleeve barbell ETF: mean-reversion sleeve + momentum sleeve,
drawn from ~50 large-cap US stocks diversified across sectors.

All knobs (universe, dates, fees, parameters) live here so we don't
hunt through scripts to edit them.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
PLOTS_DIR    = PROJECT_ROOT / "plots"
REPORTS_DIR  = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time horizon (matches the 25-year window in the assignment spec)
# ---------------------------------------------------------------------------
START_DATE = "1999-01-01"
END_DATE   = "2024-12-31"

# ---------------------------------------------------------------------------
# Stock universe — 50 large-cap US names with full 1999–2024 history,
# diversified across 10 GICS sectors. Survivorship bias is noted as a
# limitation in the report (we only pick names that exist today).
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Technology (8)
    "AAPL": "Technology",
    "MSFT": "Technology",
    "ORCL": "Technology",
    "IBM":  "Technology",
    "INTC": "Technology",
    "CSCO": "Technology",
    "ADBE": "Technology",
    "TXN":  "Technology",

    # Financials (6)
    "JPM":  "Financials",
    "BAC":  "Financials",
    "WFC":  "Financials",
    "C":    "Financials",
    "AXP":  "Financials",
    "USB":  "Financials",

    # Healthcare (7)
    "JNJ":  "Healthcare",
    "PFE":  "Healthcare",
    "MRK":  "Healthcare",
    "ABT":  "Healthcare",
    "BMY":  "Healthcare",
    "LLY":  "Healthcare",
    "AMGN": "Healthcare",

    # Consumer Staples (5)
    "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples",
    "PG":   "Consumer Staples",
    "WMT":  "Consumer Staples",
    "CL":   "Consumer Staples",

    # Consumer Discretionary (6)
    "HD":   "Consumer Discretionary",
    "MCD":  "Consumer Discretionary",
    "NKE":  "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW":  "Consumer Discretionary",
    "TGT":  "Consumer Discretionary",

    # Energy (4)
    "XOM":  "Energy",
    "CVX":  "Energy",
    "COP":  "Energy",
    "SLB":  "Energy",

    # Industrials (7)
    "BA":   "Industrials",
    "CAT":  "Industrials",
    "MMM":  "Industrials",
    "LMT":  "Industrials",
    "HON":  "Industrials",
    "FDX":  "Industrials",
    "UNP":  "Industrials",

    # Utilities (3)
    "NEE":  "Utilities",
    "DUK":  "Utilities",
    "SO":   "Utilities",

    # Communication Services (2)
    "VZ":   "Communication Services",
    "T":    "Communication Services",

    # Materials (2)
    "NEM":  "Materials",
    "APD":  "Materials",
}

FUND_TICKERS = list(UNIVERSE.keys())
SECTORS = sorted(set(UNIVERSE.values()))

# Benchmarks and reference series.
# SPY is the primary benchmark because its Adj Close includes reinvested
# dividends — a fair "total return" comparison against our adj-close stock
# prices. ^GSPC is kept as a price-only reference.
BENCHMARK_TICKER       = "SPY"      # primary: total-return benchmark
BENCHMARK_PRICE_TICKER = "^GSPC"    # secondary: price-only S&P 500 index
RISK_FREE_TICKER       = "^IRX"     # 13-week T-bill yield (in % units)

# ---------------------------------------------------------------------------
# Trading-day assumptions
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Strategy parameters — three-book risk-parity design (Option 2)
# ---------------------------------------------------------------------------

# Mean-reversion book ------------------------------------------------------
MR_LOOKBACK_DAYS = 20          # window for moving average / z-score
MR_ENTRY_Z       = -1.5        # buy when z-score below this
MR_EXIT_Z        = 0.0         # sell when z-score recovers to here
MR_N_HOLDINGS    = 5           # how many oversold names to hold simultaneously

# Momentum book ------------------------------------------------------------
MOM_LOOKBACK_DAYS = 252        # 12-month return
MOM_SKIP_DAYS     = 21         # the "1" in 12-1 momentum (skip most recent month)
MOM_N_HOLDINGS    = 5          # top-N by momentum

# Defensive book -----------------------------------------------------------
# Low-vol cohort: utilities + consumer staples + selected healthcare.
# Equal-weighted within the book. "Cash" portion handled via the risk-parity
# allocator (when defensive book gets >100% of nominal allocation we cap it
# and route the excess to a flat risk-free return).
DEFENSIVE_TICKERS = ["NEE", "DUK", "SO", "KO", "PEP", "PG", "WMT", "CL", "JNJ"]

# Risk-parity allocator ----------------------------------------------------
RP_VOL_LOOKBACK = 60           # days of returns for book-vol estimate
RP_MIN_WEIGHT   = 0.05         # floor for any book's weight
RP_MAX_WEIGHT   = 0.70         # ceiling for any book's weight

# Regime indicator ---------------------------------------------------------
REGIME_TREND_LOOKBACK = 200    # SPY vs 200-day MA → trend regime
REGIME_VOL_LOOKBACK   = 20     # SPY 20-day realized vol
REGIME_VOL_PERCENTILE = 0.90   # > this percentile (in train data) = "turbulent"

# Regime tilt — multiplicative adjustments to the risk-parity base weights.
# Keys = (trend, turbulent). Values = (mean_rev_mult, momentum_mult, defensive_mult).
# After applying, weights are renormalized.
REGIME_TILTS = {
    ("bull",  False): (1.0, 1.3, 0.7),   # bull/calm   → favor momentum
    ("bull",  True):  (1.3, 1.0, 0.9),   # bull/turb   → favor mean-rev
    ("bear",  False): (1.3, 0.7, 1.0),   # bear/calm   → mean-rev, cut momentum
    ("bear",  True):  (1.0, 0.5, 1.5),   # bear/turb   → defensive dominates
}

# Risk overlay (still applied at portfolio level on top of the books) ------
VOL_LOOKBACK_DAYS   = 20
VOL_CIRCUIT_BREAKER = 0.30     # if portfolio realized vol > this, de-risk
DERISK_SCALE        = 0.50     # scale positions to this fraction when triggered

# Per-name hard stop loss (independent of signal) --------------------------
HARD_STOP_PCT = -0.15

# Rebalance cadence --------------------------------------------------------
REBALANCE_FREQ = "W-FRI"       # weekly, Friday close

# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
WF_TRAIN_YEARS = 5             # rolling window for parameter estimation
WF_TEST_YEARS  = 1             # out-of-sample test window
WF_STEP_YEARS  = 1             # advance by this much per fold

# ---------------------------------------------------------------------------
# Cost model (used in backtest)
# ---------------------------------------------------------------------------
COMMISSION_PER_TRADE   = 0.0005   # 5 bps each side (covers commission+slippage)
ANNUAL_MGMT_FEE        = 0.0075   # 75 bps annual management fee (baseline)
PERFORMANCE_FEE        = 0.0      # turn on later for sensitivity analysis

# Fee sensitivity grid (for the report)
MGMT_FEE_GRID         = [0.0, 0.005, 0.01, 0.02, 0.04]
PERF_FEE_GRID         = [0.0, 0.05, 0.15, 0.25]

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MC_N_SIMULATIONS  = 1000
MC_BLOCK_SIZE     = 21         # block-bootstrap block length (days)
MC_SEED           = 42
