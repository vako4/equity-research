from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from equity_research.analytics import cross_section, factors
from equity_research.analytics.ratios import fcf_ratios, valuation_ratios


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


# ---------------------------------------------------------------------------
# valuation_ratios regression — guards the _compute_ev refactor
# ---------------------------------------------------------------------------

def test_valuation_ratios_aapl_2025_12_31():
    # Expected values derived from raw DB inputs in scripts/regression_setup.py.
    # 4 quarterly rows visible at 2025-12-31 (report_date <= 2025-12-31);
    # balance-sheet from 2025-11-14 row; TTM = sum of all 4 quarters.
    # adj_close=271.60583496, mktcap=4037.043B, ev=4099.766B,
    # ni_ttm=112.010B, oi_ttm=133.050B, equity=73.733B
    df = valuation_ratios("AAPL")
    # <= slice + iloc[-1] mirrors factors.py point-in-time selection;
    # 2025-12-31 may not be a trading day in the index.
    row = df.loc[:"2025-12-31"].iloc[-1]

    assert row["pe_ttm"]  == pytest.approx(36.041808, rel=1e-6)
    assert row["pb"]      == pytest.approx(54.752186, rel=1e-6)
    assert row["ev_ebit"] == pytest.approx(30.813724, rel=1e-6)


# ---------------------------------------------------------------------------
# fcf_ratios
# ---------------------------------------------------------------------------

def _fcf_mock_conn(rows):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def test_fcf_ratios_happy_path():
    # adj_close=100.0, shares=1_000_000 → market_cap=100e6
    # total_debt=0, cash=0 → EV=100e6
    # fcf per quarter=5e6, TTM=20e6 → fcf_yield = 20e6 / 100e6 = 0.20
    mock_prices = pd.Series(
        [100.0], index=pd.DatetimeIndex(["2025-06-30"])
    )
    mock_rows = [
        ("2024-07-01", 5e6, 1_000_000, 0.0, 0.0),
        ("2024-10-01", 5e6, 1_000_000, 0.0, 0.0),
        ("2025-01-01", 5e6, 1_000_000, 0.0, 0.0),
        ("2025-04-01", 5e6, 1_000_000, 0.0, 0.0),
    ]
    with patch("equity_research.analytics.ratios.get_connection", return_value=_fcf_mock_conn(mock_rows)), \
         patch("equity_research.analytics.ratios._load_prices", return_value=mock_prices):
        result = fcf_ratios("TEST")

    row = result.loc[:"2025-06-30"].iloc[-1]
    assert row["fcf_yield"] == pytest.approx(0.20, rel=1e-6)


def test_fcf_ratios_truncated_ttm_is_nan():
    # 3 quarterly rows — rolling(4, min_periods=4) returns NaN for all rows
    mock_prices = pd.Series(
        [100.0], index=pd.DatetimeIndex(["2025-06-30"])
    )
    mock_rows = [
        ("2024-10-01", 5e6, 1_000_000, 0.0, 0.0),
        ("2025-01-01", 5e6, 1_000_000, 0.0, 0.0),
        ("2025-04-01", 5e6, 1_000_000, 0.0, 0.0),
    ]
    with patch("equity_research.analytics.ratios.get_connection", return_value=_fcf_mock_conn(mock_rows)), \
         patch("equity_research.analytics.ratios._load_prices", return_value=mock_prices):
        result = fcf_ratios("TEST")

    row = result.loc[:"2025-06-30"].iloc[-1]
    assert pd.isna(row["fcf_yield"])


def test_fcf_ratios_negative_ev_is_nan():
    # market_cap=100e6, total_debt=0, cash=500e6 → EV=-400e6 < 0 → NaN
    # Positive FCF with negative EV should not produce a usable yield
    mock_prices = pd.Series(
        [100.0], index=pd.DatetimeIndex(["2025-06-30"])
    )
    mock_rows = [
        ("2024-07-01", 5e6, 1_000_000, 0.0, 500e6),
        ("2024-10-01", 5e6, 1_000_000, 0.0, 500e6),
        ("2025-01-01", 5e6, 1_000_000, 0.0, 500e6),
        ("2025-04-01", 5e6, 1_000_000, 0.0, 500e6),
    ]
    with patch("equity_research.analytics.ratios.get_connection", return_value=_fcf_mock_conn(mock_rows)), \
         patch("equity_research.analytics.ratios._load_prices", return_value=mock_prices):
        result = fcf_ratios("TEST")

    row = result.loc[:"2025-06-30"].iloc[-1]
    assert pd.isna(row["fcf_yield"])
