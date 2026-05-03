from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def compute(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    ratio_fn: Callable[[str, pd.Timestamp], pd.Series],
) -> pd.DataFrame:
    results: dict[str, pd.Series] = {}
    for ticker in tickers:
        results[ticker] = ratio_fn(ticker, as_of_date)
    return pd.DataFrame(results).T
