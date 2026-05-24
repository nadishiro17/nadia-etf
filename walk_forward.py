"""
walk_forward.py — Walk-forward validation harness.

Slides a (train, test) window through time. On each fold:
  1. The vol-percentile threshold for the regime detector is fit on train.
  2. The strategy runs across train+test, but only the test-window equity
     is kept and stitched into the walk-forward curve.

Why this matters: our strategy has one data-fit parameter (the 90th-pctl vol
threshold) and several predetermined parameters. Walk-forward gives an
honest out-of-sample equity curve where the threshold has never seen the
test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest
import config


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp


def build_folds(
    index: pd.DatetimeIndex,
    train_years: int = config.WF_TRAIN_YEARS,
    test_years: int = config.WF_TEST_YEARS,
    step_years: int = config.WF_STEP_YEARS,
) -> list[Fold]:
    """Create non-overlapping out-of-sample test windows."""
    start = index[0]
    end   = index[-1]
    folds = []
    current = start + pd.DateOffset(years=train_years)
    while current + pd.DateOffset(years=test_years) <= end:
        folds.append(Fold(
            train_start=current - pd.DateOffset(years=train_years),
            train_end=  current,
            test_start= current,
            test_end=   current + pd.DateOffset(years=test_years),
        ))
        current = current + pd.DateOffset(years=step_years)
    return folds


def run_walk_forward(
    prices: pd.DataFrame,
    benchmark: pd.Series,
) -> dict:
    """
    Run the strategy on each fold with the vol threshold set from the
    fold's training data only. Stitch test windows together.
    """
    folds = build_folds(prices.index)
    print(f"Walk-forward: {len(folds)} folds")

    oos_returns = []
    fold_summaries = []

    for i, f in enumerate(folds, 1):
        # We run the full pipeline up through test_end so signals can
        # use their full lookbacks, but we only KEEP returns from
        # test_start onward when stitching together the OOS curve.
        sub_prices = prices.loc[:f.test_end].copy()
        sub_bench  = benchmark.loc[:f.test_end].copy()

        result = backtest.run_strategy(sub_prices, sub_bench, train_end=f.train_end)
        sim = result["simulation"]

        # Slice out the test window
        oos = sim["net_return"].loc[f.test_start:f.test_end]
        oos_returns.append(oos)

        # Per-fold summary
        eq = (1 + oos).cumprod()
        if len(eq) >= 2:
            fold_summaries.append({
                "fold": i,
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
                "test_total_return": eq.iloc[-1] - 1,
                "test_max_dd": (eq / eq.cummax() - 1).min(),
                "n_days": len(eq),
            })

    # Concatenate
    oos_concat = pd.concat(oos_returns).sort_index()
    oos_concat = oos_concat[~oos_concat.index.duplicated(keep="first")]
    equity = (1 + oos_concat).cumprod()

    return {
        "folds": folds,
        "oos_returns": oos_concat,
        "oos_equity": equity,
        "fold_summary": pd.DataFrame(fold_summaries),
    }
