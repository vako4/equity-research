from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from equity_research.analytics import cross_section, factors


# ---------------------------------------------------------------------------
# cross_section.compute
# ---------------------------------------------------------------------------

def test_compute_shape():
    fixed = pd.Series({"pe_ttm": 10.0, "pb": 1.5, "ev_ebit": 8.0})
    tickers = ["AAPL", "MSFT", "GOOG"]
    as_of = pd.Timestamp("2024-12-31")

    result = cross_section.compute(tickers, as_of, ratio_fn=lambda t, d: fixed)

    assert result.shape == (3, 3)
    assert list(result.index) == tickers
    assert list(result.columns) == ["pe_ttm", "pb", "ev_ebit"]


# ---------------------------------------------------------------------------
# factors.quality_factor — truncated TTM
# ---------------------------------------------------------------------------

def _quarterly_df(n: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return a quality_ratios-shaped DataFrame with n quarterly rows before as_of."""
    dates = pd.date_range(end=as_of - pd.Timedelta(days=1), periods=n, freq="QE")
    return pd.DataFrame(
        {
            "frequency": "quarterly",
            "roe": 0.15,
            "roce": 0.12,
            "gross_margin": 0.40,
            "debt_equity": 0.50,
        },
        index=dates,
    )


def test_quality_factor_truncated_ttm_returns_all_nan():
    as_of = pd.Timestamp("2024-12-31")
    # 3 quarterly rows before as_of — one short of the 4 required for a full TTM
    mock_df = _quarterly_df(n=3, as_of=as_of)

    with patch("equity_research.analytics.factors.ratios.quality_ratios", return_value=mock_df):
        result = factors.quality_factor("TEST", as_of)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["roe", "roce", "gross_margin", "debt_equity"]
    assert result.isna().all(), f"Expected all-NaN Series, got:\n{result}"


def test_quality_factor_four_quarters_returns_values():
    as_of = pd.Timestamp("2024-12-31")
    # 4 quarterly rows — exactly enough for a valid TTM
    mock_df = _quarterly_df(n=4, as_of=as_of)

    with patch("equity_research.analytics.factors.ratios.quality_ratios", return_value=mock_df):
        result = factors.quality_factor("TEST", as_of)

    assert not result.isna().any(), f"Expected non-NaN values, got:\n{result}"
