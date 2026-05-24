"""
books.py — The three "books" (sub-portfolios) that make up the fund.

Each book takes a price panel and returns a DataFrame of target weights
(rows = dates, columns = tickers). Weights within a book sum to 1 on each
date (or 0 if the book is in cash). The risk-parity allocator in
portfolio.py later scales each book by an overall book weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import signals


# ---------------------------------------------------------------------------
# Book 1: Mean reversion
# ---------------------------------------------------------------------------
def mean_reversion_book(
    prices: pd.DataFrame,
    lookback: int = config.MR_LOOKBACK_DAYS,
    entry_z: float = config.MR_ENTRY_Z,
    exit_z: float = config.MR_EXIT_Z,
    n_holdings: int = config.MR_N_HOLDINGS,
) -> pd.DataFrame:
    """
    State-aware mean-reversion sleeve.

    Logic per ticker (run independently):
      - Enter long when z-score < entry_z (oversold)
      - Exit when z-score >= exit_z (mean reversion completed)

    Equal-weight across at most `n_holdings` active positions on each day.
    If more than `n_holdings` tickers are "in" simultaneously, keep the
    `n_holdings` with the most-negative z (most oversold) and exit the rest.
    """
    z = signals.zscore(prices, lookback=lookback)

    # Per-ticker state: 1 when in position, 0 when flat.
    # We can't vectorize the state transition (entry/exit are stateful),
    # so we loop per ticker. Universe is 50 stocks × 6540 days — fast enough.
    in_position = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)

    for tk in prices.columns:
        zt = z[tk].values
        state = 0
        col = np.zeros(len(zt), dtype=int)
        for i in range(len(zt)):
            if np.isnan(zt[i]):
                col[i] = 0
                continue
            if state == 0 and zt[i] < entry_z:
                state = 1
            elif state == 1 and zt[i] >= exit_z:
                state = 0
            col[i] = state
        in_position[tk] = col

    # When more than n_holdings are "in", keep the most oversold.
    # We zero-out the rest by ranking z (more negative = better rank).
    masked_z = z.where(in_position.astype(bool))
    rank = masked_z.rank(axis=1, ascending=True, method="first")  # 1 = most oversold
    keep = (rank <= n_holdings) & in_position.astype(bool)

    # Equal-weight across kept positions; 0 if no positions held that day.
    n_active = keep.sum(axis=1).replace(0, np.nan)
    weights = keep.astype(float).div(n_active, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# Book 2: Momentum (12-1)
# ---------------------------------------------------------------------------
def momentum_book(
    prices: pd.DataFrame,
    lookback: int = config.MOM_LOOKBACK_DAYS,
    skip: int = config.MOM_SKIP_DAYS,
    n_holdings: int = config.MOM_N_HOLDINGS,
) -> pd.DataFrame:
    """
    Hold the top-N stocks by 12-1 momentum.

    Rebalance is implicit (daily weights updated every day); the actual
    rebalance cadence is enforced at the portfolio level by sampling
    these weights weekly. This keeps the signal layer simple.
    """
    mom = signals.momentum(prices, lookback=lookback, skip=skip)
    keep = signals.top_n_mask(mom, n=n_holdings, ascending=False)
    n_active = keep.sum(axis=1).replace(0, np.nan)
    weights = keep.astype(float).div(n_active, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# Book 3: Defensive (low-vol cohort, equal-weighted)
# ---------------------------------------------------------------------------
def defensive_book(
    prices: pd.DataFrame,
    tickers: list[str] = config.DEFENSIVE_TICKERS,
) -> pd.DataFrame:
    """
    Equal-weighted across the predefined defensive cohort, whenever those
    prices are available. We rebalance to equal-weight on every date — this
    is the "anchor" that doesn't require any signal.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    available = prices[tickers].notna()
    n_avail = available.sum(axis=1).replace(0, np.nan)
    for tk in tickers:
        weights[tk] = available[tk].astype(float) / n_avail
    return weights.fillna(0.0)
