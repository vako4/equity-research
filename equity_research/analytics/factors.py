from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.analytics import ratios

_VALUATION_COLS = ["pe_ttm", "pb", "ev_ebit"]
_QUALITY_COLS = ["roe", "roce", "gross_margin", "debt_equity"]
_FCF_COLS = ["fcf_yield"]


def _nan_series(cols: list[str], name: str) -> pd.Series:
    return pd.Series(np.nan, index=cols, name=name)


def valuation_factor(ticker: str, as_of_date: pd.Timestamp) -> pd.Series:
    try:
        df = ratios.valuation_ratios(ticker)
    except ValueError:
        return _nan_series(_VALUATION_COLS, ticker)
    candidates = df[df.index <= as_of_date]
    if candidates.empty:
        return _nan_series(_VALUATION_COLS, ticker)
    return candidates.iloc[-1].rename(ticker)


def quality_factor(ticker: str, as_of_date: pd.Timestamp) -> pd.Series:
    try:
        df = ratios.quality_ratios(ticker)
    except ValueError:
        return _nan_series(_QUALITY_COLS, ticker)
    # Quarterly only: TTM uses rolling 4-quarter sums; annual rows are not equivalent
    quarterly = df[df["frequency"] == "quarterly"]
    prior = quarterly[quarterly.index <= as_of_date]
    # Fewer than 4 quarters before as_of_date means TTM is incomplete
    if len(prior) < 4:
        return _nan_series(_QUALITY_COLS, ticker)
    row = prior.iloc[-1].drop("frequency").rename(ticker)
    # roce is NaN when total_equity <= 0 — quality_ratios enforces this contract
    return row


def fcf_yield_factor(ticker: str, as_of_date: pd.Timestamp) -> pd.Series:
    try:
        df = ratios.fcf_ratios(ticker)
    except ValueError:
        return _nan_series(_FCF_COLS, ticker)
    candidates = df[df.index <= as_of_date]
    if candidates.empty:
        return _nan_series(_FCF_COLS, ticker)
    return candidates.iloc[-1].rename(ticker)
