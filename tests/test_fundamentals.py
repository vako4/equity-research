from datetime import date, timedelta

import pandas as pd
import pytest

from equity_research.ingestion.fundamentals import (
    _closest_col,
    _process_periods,
    _upsert_period,
)


# Test 3
def test_fundamentals_upsert_idempotent(conn_with_ticker):
    conn = conn_with_ticker
    period_end = date(2023, 9, 30)

    _upsert_period("AAPL", period_end, 90, {"revenue": 100_000.0}, "financials_annual", conn)
    conn.commit()

    # Backdate ingested_at so the second upsert's refresh is unambiguous.
    conn.execute("UPDATE financials_annual SET ingested_at = '2000-01-01 00:00:00' WHERE ticker = 'AAPL'")
    conn.commit()

    _upsert_period("AAPL", period_end, 90, {"revenue": 200_000.0}, "financials_annual", conn)
    conn.commit()

    count, revenue, ingested_at = conn.execute(
        "SELECT COUNT(*), revenue, ingested_at FROM financials_annual WHERE ticker = 'AAPL'"
    ).fetchone()
    assert count == 1                               # no duplicate row
    assert revenue == 200_000.0                     # fields overwritten, not skipped
    assert ingested_at != "2000-01-01 00:00:00"     # ingested_at refreshed on UPDATE


def _stmt_df(col_dates: list[str]) -> pd.DataFrame:
    """DataFrame shaped like yfinance: financial labels as row index, dates as columns."""
    return pd.DataFrame(
        data=[[100_000.0] * len(col_dates)],
        index=["Total Revenue"],
        columns=pd.to_datetime(col_dates),
    )


# Test 4
def test_closest_col_within_tolerance():
    """Column 44 days from target is matched — one day inside the 45-day window."""
    target = pd.Timestamp("2023-09-30")
    col = target + pd.Timedelta(days=44)
    df = _stmt_df([col.isoformat()])
    result = _closest_col(df, target)
    assert result is not None
    assert abs((result - target).days) == 44


# Test 5
def test_closest_col_outside_tolerance():
    """Column 46 days from target returns None — one day outside the 45-day window."""
    target = pd.Timestamp("2023-09-30")
    col = target + pd.Timedelta(days=46)
    df = _stmt_df([col.isoformat()])
    result = _closest_col(df, target)
    assert result is None


# Test 6
def test_process_periods_skips_unrecognized_labels(conn_with_ticker):
    """A statement with plausible-but-unrecognized labels produces 0 upserted rows."""
    # "Total Sales" and "Net Profit" look real but match none of the expected maps.
    # Without the all-null guard, this would insert a junk row with every field NULL.
    inc = pd.DataFrame(
        data=[[500_000.0], [50_000.0]],
        index=["Total Sales", "Net Profit"],
        columns=pd.to_datetime(["2023-09-30"]),
    )
    conn = conn_with_ticker
    n = _process_periods(inc, None, None, "AAPL", 90, "financials_annual", conn)
    assert n == 0
    rows = conn.execute("SELECT COUNT(*) FROM financials_annual").fetchone()[0]
    assert rows == 0


# Test 8
def test_off_calendar_fiscal_year_alignment(conn_with_ticker):
    """September fiscal year end is stored as-is and aligns correctly with balance sheet."""
    # AAPL closes its fiscal year on the last Saturday of September.
    # A balance sheet dated 44 days later must align; one 46 days later must not.
    income_date = pd.Timestamp("2023-09-30")
    bal_close = income_date + pd.Timedelta(days=44)   # within tolerance — should align
    bal_far   = income_date + pd.Timedelta(days=46)   # outside tolerance — must not align

    inc = pd.DataFrame(
        data=[[100_000.0]],
        index=["Total Revenue"],
        columns=pd.DatetimeIndex([income_date]),
    )
    bal_aligned = pd.DataFrame(
        data=[[500_000.0]],
        index=["Total Assets"],
        columns=pd.DatetimeIndex([bal_close]),
    )
    bal_stale = pd.DataFrame(
        data=[[500_000.0]],
        index=["Total Assets"],
        columns=pd.DatetimeIndex([bal_far]),
    )

    conn = conn_with_ticker

    # Aligned balance sheet: 1 period upserted, fiscal_period_end stored as Sep 30.
    n = _process_periods(inc, bal_aligned, None, "AAPL", 90, "financials_annual", conn)
    assert n == 1
    row = conn.execute(
        "SELECT fiscal_period_end, total_assets FROM financials_annual WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row[0] == "2023-09-30"   # off-calendar date preserved exactly
    assert row[1] == 500_000.0      # balance sheet value aligned and stored

    # Stale balance sheet (46 days out): total_assets must be NULL — not silently picked up.
    conn.execute("DELETE FROM financials_annual WHERE ticker = 'AAPL'")
    conn.commit()
    _process_periods(inc, bal_stale, None, "AAPL", 90, "financials_annual", conn)
    row = conn.execute(
        "SELECT total_assets FROM financials_annual WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row[0] is None
